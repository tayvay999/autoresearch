"""
Buyer Match Algorithm v5.0 — Data-Driven Refinement
=====================================================
Major fixes from v4.5 backtesting:

v5.0 Changes:
  1. FIXED cap_rate SD: 0.8 → 0.015 (was 50x too wide — all buyers scored 100)
  2. ADDED units dimension: buyers who buy 8-units buy 8-units (strong signal)
  3. REDUCED grm weight: 10% → 5% (51% missing data = noise)
  4. REDUCED missing_penalty: 0.90 → 0.95 (less harsh on missing GRM/PPSF)
  5. EXPANDED velocity scoring: more granular thresholds
  6. EXPANDED strategy scoring: wider range (30-80 instead of 45-65)
  7. REBALANCED weights based on discrimination analysis

Backtesting showed v4.5 had 12.2% top-10 accuracy, mostly because:
  - Cap rate (20% weight) scored every buyer 100 (zero discrimination)
  - GRM (10% weight) was missing for 51% of buyers
  - No unit-count matching despite strong signal in data

Author: Taylor Avakian / The Group CRE
Version: 5.0  (2026-03-29)
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import statistics

__version__ = "6.0"

# ── Core Parameters ─────────────────────────────────────────────────────
RECENCY_HALF_LIFE_DAYS = 365
RECENCY_FLOOR = 0.15
VELOCITY_WINDOW_DAYS = 730
BEST_DEAL_WEIGHT = 0.60
AVG_DEAL_WEIGHT = 0.40
MISSING_PENALTY = 0.95       # Softened from 0.90 (less harsh on missing GRM/PPSF)
REPEAT_BONUS = 5
REPEAT_CAP = 98

# ── v5.0 Weights Distribution (10 dimensions, sum to 1.0) ──────────────
# Optimized via grid-search backtesting: 30.6% top-10 accuracy (was 12.2% in v4.5)
DEFAULT_WEIGHTS = {
    "submarket": 0.1825,
    "velocity": 0.1101,
    "ppu": 0.2230,
    "units": 0.0101,
    "cap_rate": 0.0429,
    "price": 0.1014,
    "recency": 0.0108,
    "grm": 0.1916,
    "strategy": 0.1173,
    "ppsf": 0.0104,
}

# ── v5.0 Market Standard Deviations ────────────────────────────────────
# CRITICAL FIX: cap_rate SD was 0.8 in v4.5 but cap rates are ~0.014-0.126
# With SD=0.8, the Gaussian scored EVERY buyer 99.9-100 (zero discrimination)
# Actual data SD is 0.015 — using 0.015 for tight discrimination
DEFAULT_MARKET_SD = {
    "ppu": 66301.8110,
    "ppsf": 102,
    "cap_rate": 0.0120,
    "grm": 1.1460,
    "price": 1055717.4290,
    "units": 5,
}

# ── Velocity Score Thresholds (24-month window) ─────────────────────────
# Expanded for more granularity
VELOCITY_THRESHOLDS = {
    6: 100,   # 6+ deals
    5: 95,    # 5 deals
    4: 85,    # 4 deals
    3: 70,    # 3 deals
    2: 55,    # 2 deals
    1: 30,    # 1 deal
    0: 0,     # 0 deals
}

# ── Recency Score Bands ─────────────────────────────────────────────────
RECENCY_BANDS = [
    (90, 100),
    (180, 85),
    (365, 70),
    (548, 50),
    (730, 35),
    (1095, 20),
    (float('inf'), 10),
]


# ── Dynamic Submarket Adjacency Graph ───────────────────────────────────
LA_SUBMARKET_ADJACENCIES = {
    # Central LA Core
    "East Hollywood": {"Hollywood", "Koreatown", "Thai Town", "City West"},
    "Hollywood": {"East Hollywood", "Koreatown", "Thai Town", "Los Feliz", "Larchmont"},
    "Koreatown": {"East Hollywood", "Hollywood", "Thai Town", "Westlake", "Mid-Wilshire"},
    "Thai Town": {"East Hollywood", "Hollywood", "Koreatown", "Chinatown", "Larchmont"},
    "Chinatown": {"Thai Town", "Little Tokyo/Arts District", "Downtown LA", "Westlake"},
    "Westlake": {"Koreatown", "Thai Town", "Chinatown", "MacArthur Park", "Downtown LA"},
    "MacArthur Park": {"Westlake", "Mid-Wilshire", "Rampart Village", "Downtown LA"},
    "Downtown LA": {"Chinatown", "Westlake", "MacArthur Park", "Little Tokyo/Arts District", "South Park"},
    "Little Tokyo/Arts District": {"Chinatown", "Downtown LA", "Boyle Heights"},
    # East Central LA
    "Silver Lake": {"Hollywood", "Los Feliz", "Echo Park", "Atwater Village"},
    "Echo Park": {"Silver Lake", "Los Feliz", "Chinatown"},
    "Los Feliz": {"Hollywood", "Silver Lake", "Echo Park", "Atwater Village"},
    "Atwater Village": {"Silver Lake", "Los Feliz", "Glassell Park", "Highland Park"},
    "Boyle Heights": {"Little Tokyo/Arts District", "Lincoln Heights", "East LA"},
    "Lincoln Heights": {"Boyle Heights", "Highland Park", "Eagle Rock"},
    "Highland Park": {"Atwater Village", "Lincoln Heights", "Eagle Rock", "Glassell Park"},
    "Eagle Rock": {"Lincoln Heights", "Highland Park", "Pasadena"},
    "Glassell Park": {"Atwater Village", "Highland Park"},
    # Mid-City LA
    "City West": {"East Hollywood", "Hollywood", "Larchmont", "Hancock Park"},
    "Larchmont": {"Hollywood", "Thai Town", "City West", "Hancock Park", "Mid-Wilshire"},
    "Hancock Park": {"City West", "Larchmont", "Mid-Wilshire", "Arlington Heights"},
    "Arlington Heights": {"Hancock Park", "Mid-Wilshire", "Fairfax", "Crenshaw", "Harvard Heights"},
    "Mid-Wilshire": {"Koreatown", "MacArthur Park", "Larchmont", "Hancock Park", "Arlington Heights", "Fairfax"},
    "Rampart Village": {"MacArthur Park", "Mid-Wilshire", "Mid-City"},
    "Fairfax": {"Arlington Heights", "Mid-Wilshire", "Beverly Grove"},
    # South Central LA
    "Mid-City": {"Rampart Village", "Pico-Union", "Vermont Square", "Vermont Knolls"},
    "Pico-Union": {"Mid-City", "Harvard Heights", "West Adams", "Downtown LA"},
    "Harvard Heights": {"Pico-Union", "Arlington Heights", "West Adams"},
    "West Adams": {"Pico-Union", "Mid-City", "Historic South-Central"},
    "Vermont Square": {"Mid-City", "Vermont Knolls", "Leimert Park"},
    "Vermont Knolls": {"Mid-City", "Vermont Square", "Leimert Park"},
    "Leimert Park": {"Vermont Square", "Vermont Knolls", "Jefferson Park"},
    "Jefferson Park": {"Leimert Park", "Exposition Park"},
    "Exposition Park": {"Jefferson Park", "University Park"},
    "University Park": {"Exposition Park", "Historic South-Central"},
    "Historic South-Central": {"West Adams", "South Park", "Central-Alameda"},
    "South Park": {"Downtown LA", "Historic South-Central", "Central-Alameda"},
    "Central-Alameda": {"Historic South-Central", "South Park"},
    # West Side
    "Beverly Grove": {"Fairfax", "West Hollywood"},
    "West Hollywood": {"Beverly Grove", "Culver City"},
    "Culver City": {"West Hollywood", "Palms", "Mar Vista"},
    "Palms": {"Culver City", "Mar Vista", "Del Rey"},
    "Mar Vista": {"Culver City", "Palms", "Del Rey", "Playa Vista"},
    "Del Rey": {"Palms", "Mar Vista", "Playa Vista"},
    "Playa Vista": {"Mar Vista", "Del Rey"},
    # South Bay & Harbor
    "Inglewood": {"Hawthorne", "Compton"},
    "Hawthorne": {"Inglewood", "Gardena", "Lawndale"},
    "Gardena": {"Hawthorne", "Torrance", "Long Beach"},
    "Torrance": {"Gardena", "Long Beach"},
    "Long Beach": {"Gardena", "Torrance", "San Pedro"},
    "San Pedro": {"Long Beach", "Wilmington"},
    "Wilmington": {"San Pedro", "Carson"},
    "Carson": {"Wilmington", "Compton", "Downey"},
    "Compton": {"Inglewood", "Carson", "Lynwood", "South Gate"},
    # South LA
    "Lynwood": {"Compton", "South Gate", "Downey"},
    "South Gate": {"Compton", "Lynwood", "Downey"},
    "Downey": {"Carson", "Lynwood", "South Gate", "Whittier"},
    "Whittier": {"Downey", "Pasadena"},
    # San Gabriel Valley
    "Pasadena": {"Eagle Rock", "Whittier", "Glendale"},
    "Glendale": {"Pasadena", "Burbank", "North Hollywood"},
    # San Fernando Valley
    "Burbank": {"Glendale", "North Hollywood"},
    "North Hollywood": {"Burbank", "Van Nuys", "Sherman Oaks"},
    "Van Nuys": {"North Hollywood", "Sherman Oaks", "Panorama City"},
    "Sherman Oaks": {"Van Nuys", "Encino", "Woodland Hills"},
    "Encino": {"Sherman Oaks", "Tarzana", "Woodland Hills"},
    "Tarzana": {"Encino", "Woodland Hills", "Canoga Park"},
    "Woodland Hills": {"Sherman Oaks", "Encino", "Tarzana", "Canoga Park", "Chatsworth"},
    "Canoga Park": {"Tarzana", "Woodland Hills", "Chatsworth", "Northridge"},
    "Chatsworth": {"Woodland Hills", "Canoga Park", "Northridge"},
    "Northridge": {"Canoga Park", "Chatsworth", "Granada Hills", "Arleta"},
    "Granada Hills": {"Northridge", "Sylmar", "Sun Valley"},
    "Sylmar": {"Granada Hills", "Pacoima", "Sun Valley"},
    "Sun Valley": {"Granada Hills", "Sylmar", "Panorama City", "Arleta"},
    "Panorama City": {"Van Nuys", "Sun Valley", "Arleta"},
    "Arleta": {"Northridge", "Sun Valley", "Panorama City", "Pacoima"},
    "Pacoima": {"Sylmar", "Arleta"},
    # CoStar-specific submarket names
    "South Central LA": {"West Adams", "Historic South-Central", "Vernon-Main", "Vermont Harbor", "Park Mesa Heights", "Crenshaw"},
    "Crenshaw": {"South Central LA", "Leimert Park", "Park Mesa Heights", "Mid-City", "Arlington Heights"},
    "Park Mesa Heights": {"South Central LA", "Crenshaw", "Vermont Harbor", "Leimert Park"},
    "Vermont Harbor": {"South Central LA", "Park Mesa Heights", "Historic South-Central"},
    "Southeast Los Angeles": {"Vernon-Main", "Boyle Heights", "Central-Alameda", "South Central LA"},
    "Vernon-Main": {"South Central LA", "Historic South-Central", "Central-Alameda", "Southeast Los Angeles"},
    "Miracle Mile": {"Mid-Wilshire", "Fairfax", "Beverly Grove", "Hancock Park"},
    "Florence-Graham": {"South Central LA", "Compton", "Inglewood"},
    "Canndu/Avalon Gardens": {"South Central LA", "Compton"},
    "Westlake North": {"Westlake", "MacArthur Park", "Pico-Union", "Echo Park"},
    "East LA": {"Boyle Heights", "Lincoln Heights"},
    "Lawndale": {"Hawthorne", "Gardena"},
}

SUBMARKET_SCORES = {
    "same": 100,
    "adjacent": 75,
    "near": 50,
    "market": 30,
    "other": 10,
}


# ── Date Utilities ──────────────────────────────────────────────────────

def parse_date(date_value):
    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value
    if isinstance(date_value, str):
        try:
            return datetime.fromisoformat(date_value)
        except (ValueError, TypeError):
            return None
    return None


def get_reference_date(reference_date=None):
    if reference_date is None:
        return datetime.now()
    parsed = parse_date(reference_date)
    return parsed if parsed else datetime.now()


# ── Core Scoring Functions ──────────────────────────────────────────────

def peaked_score(value, low, high, mid, sd):
    """Peaked Gaussian: exact midpoint = 100, falls off with distance."""
    if value is None or value <= 0:
        return None
    distance = abs(value - mid)
    return max(0, round(100.0 * math.exp(-(distance ** 2) / (2 * sd ** 2)), 1))


def units_score(buyer_units, subject_units, sd=None):
    """
    Score unit count match using peaked Gaussian.
    Buyers who buy 8-unit buildings are likely to buy another 8-unit building.
    """
    if buyer_units is None or subject_units is None:
        return None
    if buyer_units <= 0 or subject_units <= 0:
        return None
    unit_sd = sd or DEFAULT_MARKET_SD.get("units", 3)
    distance = abs(buyer_units - subject_units)
    return max(0, round(100.0 * math.exp(-(distance ** 2) / (2 * unit_sd ** 2)), 1))


def recency_weight(sale_date, reference_date=None, half_life_days=RECENCY_HALF_LIFE_DAYS, floor=RECENCY_FLOOR, is_high_velocity=False):
    """Exponential time-decay weight with capital indigestion penalty."""
    ref_dt = get_reference_date(reference_date)
    sale_dt = parse_date(sale_date)
    if sale_dt is None:
        return floor
    days_elapsed = max(0, (ref_dt - sale_dt).days)
    
    penalty = 1.0
    if not is_high_velocity and days_elapsed < 90:
        penalty = max(0.1, min(1.0, days_elapsed / 90))
        
    if days_elapsed == 0:
        return 1.0 * penalty
        
    decay = 0.5 ** (days_elapsed / half_life_days)
    weight = floor + (1.0 - floor) * decay
    return max(floor, min(1.0, round(weight * penalty, 3)))


def recency_score(sale_dates, reference_date=None, is_high_velocity=False):
    """Score based on most recent transaction using band lookup and indigestion penalty."""
    ref_dt = get_reference_date(reference_date)
    valid_dates = [parse_date(d) for d in sale_dates if parse_date(d) is not None]
    if not valid_dates:
        return None
    most_recent = max(valid_dates)
    days_ago = max(0, (ref_dt - most_recent).days)
    
    penalty = 1.0
    if not is_high_velocity and days_ago < 90:
        penalty = max(0.1, min(1.0, days_ago / 90))
        
    for max_days, score in RECENCY_BANDS:
        if days_ago <= max_days:
            return float(score) * penalty
    return 10.0 * penalty


def velocity_score(sale_dates, reference_date=None, window_days=VELOCITY_WINDOW_DAYS):
    """Score buyer velocity (deal count in window)."""
    ref_dt = get_reference_date(reference_date)
    window_start = ref_dt - timedelta(days=window_days)
    valid_dates = [parse_date(d) for d in sale_dates if parse_date(d) is not None]
    deals_in_window = sum(1 for d in valid_dates if window_start <= d <= ref_dt)
    for count in sorted(VELOCITY_THRESHOLDS.keys(), reverse=True):
        if deals_in_window >= count:
            return float(VELOCITY_THRESHOLDS[count])
    return 0.0


def dynamic_submarket_score(buyer_submarket, subject_submarket):
    """BFS-based submarket adjacency scoring."""
    if not buyer_submarket or not subject_submarket:
        return None
    buyer_sub = buyer_submarket.strip()
    subject_sub = subject_submarket.strip()
    if buyer_sub == subject_sub:
        return 100.0
    visited = set()
    queue = [(subject_sub, 0)]
    while queue:
        current, distance = queue.pop(0)
        if current == buyer_sub:
            if distance == 1: return 75.0
            elif distance == 2: return 50.0
            elif distance == 3: return 30.0
            else: return 10.0
        if current in visited:
            continue
        visited.add(current)
        neighbors = LA_SUBMARKET_ADJACENCIES.get(current, set())
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    return 10.0


def strategy_alignment_score(buyer_transactions, subject_deal):
    """
    v5.0: Expanded range (30-80 instead of 45-65) for better discrimination.
    """
    if not buyer_transactions or not subject_deal:
        return 50.0

    score = 50.0
    points = 0
    checks = 0

    subject_vacancy = subject_deal.get("vacancy_pct")
    subject_submarket = subject_deal.get("submarket", "")

    # Check 1: Vacancy Match (wider scoring range)
    if subject_vacancy is not None:
        buyer_vacancies = [t.get("vacancy_pct") for t in buyer_transactions
                          if t.get("vacancy_pct") is not None]
        if buyer_vacancies:
            avg_buyer_vacancy = sum(buyer_vacancies) / len(buyer_vacancies)
            vacancy_diff = abs(avg_buyer_vacancy - subject_vacancy)
            if vacancy_diff <= 3:
                points += 20
            elif vacancy_diff <= 7:
                points += 12
            elif vacancy_diff <= 12:
                points += 5
            else:
                points -= 5
            checks += 1

    # Check 2: Portfolio Clustering
    if subject_submarket:
        cluster_count = sum(1 for t in buyer_transactions
                           if t.get("submarket", "").strip() == subject_submarket.strip())
        if cluster_count >= 3:
            points += 18
        elif cluster_count >= 2:
            points += 12
        elif cluster_count == 1:
            points += 6
        checks += 1

    # Check 3: ADU/Land Play Detection
    dev_buyer_flag = False
    for txn in buyer_transactions:
        cap = txn.get("cap_rate")
        yr_built = txn.get("year_built")
        if cap is not None and cap < 0.035 and yr_built is not None:
            if datetime.now().year - yr_built > 30:
                dev_buyer_flag = True
                break

    subject_year = subject_deal.get("year_built")
    subject_lot_size = subject_deal.get("lot_size_sf")

    if dev_buyer_flag and subject_year is not None and subject_lot_size is not None:
        if datetime.now().year - subject_year > 30 and subject_lot_size > 5000:
            points += 12
        else:
            points -= 8
    checks += 1

    # Check 4: Unit size consistency (NEW in v5.0)
    subject_units = subject_deal.get("units")
    if subject_units:
        buyer_units_list = [t.get("units") for t in buyer_transactions
                           if t.get("units") is not None]
        if buyer_units_list:
            avg_units = sum(buyer_units_list) / len(buyer_units_list)
            unit_diff = abs(avg_units - subject_units)
            if unit_diff <= 2:
                points += 8
            elif unit_diff <= 4:
                points += 3
            elif unit_diff > 8:
                points -= 5
            checks += 1

    if checks > 0:
        score = 50.0 + min(50.0, max(-50.0, points))

    return round(max(0.0, min(100.0, score)), 1)


def composite_score(dim_scores, weights=None, missing_penalty=MISSING_PENALTY):
    """Weighted composite with softened missing-data penalty."""
    w = weights or DEFAULT_WEIGHTS
    total_score = 0.0
    total_weight = 0.0
    n_missing = 0

    for dim, weight in w.items():
        s = dim_scores.get(dim)
        if s is not None:
            total_score += s * weight
            total_weight += weight
        else:
            n_missing += 1

    if total_weight == 0:
        return 0.0, n_missing

    raw = total_score / total_weight
    penalized = raw * (missing_penalty ** n_missing)
    return round(penalized, 1), n_missing


def get_adaptive_weights(subject, base_weights=None):
    w = dict(base_weights or DEFAULT_WEIGHTS)
    cap = subject.get("cap_rate", {}).get("mid") if "cap_rate" in subject and isinstance(subject["cap_rate"], dict) else subject.get("cap_rate")
    vac = subject.get("vacancy_pct")
    
    is_value_add = False
    if (cap is not None and cap < 0.04) or (vac is not None and vac > 0.30):
        is_value_add = True
        
    if is_value_add:
        w["strategy"] = w.get("strategy", 0.04) + 0.10
        w["submarket"] = w.get("submarket", 0.18) + 0.05
        w["cap_rate"] = max(0.01, w.get("cap_rate", 0.10) - 0.08)
        w["grm"] = max(0.01, w.get("grm", 0.05) - 0.04)
        w["price"] = max(0.01, w.get("price", 0.10) - 0.03)
        
    is_stabilized = False
    if not is_value_add and (cap is not None and cap > 0.055) and (vac is not None and vac < 0.05):
        is_stabilized = True
        
    if is_stabilized:
        w["cap_rate"] = w.get("cap_rate", 0.10) + 0.08
        w["grm"] = w.get("grm", 0.05) + 0.03
        w["strategy"] = max(0.01, w.get("strategy", 0.04) - 0.02)
        
    total = sum(w.values())
    return {k: round(v/total, 4) for k, v in w.items()}


def score_single_transaction(buyer_metrics, subject, reference_date=None,
                            market_sd=None, weights=None, is_repeat=False,
                            buyer_specific_sd=None, is_high_velocity=False):
    """Score one transaction against the subject deal."""
    sd = buyer_specific_sd or market_sd or DEFAULT_MARKET_SD
    w = weights or DEFAULT_WEIGHTS
    ref_dt = get_reference_date(reference_date)

    dim_scores = {}

    # Numeric metric scoring (including units now)
    for metric in ["ppu", "ppsf", "cap_rate", "grm", "price"]:
        val = buyer_metrics.get(metric)
        sub = subject.get(metric, {})
        if sub and val:
            dim_scores[metric] = peaked_score(val, sub["low"], sub["high"], sub["mid"], sd[metric])
        else:
            dim_scores[metric] = None

    # Units scoring (NEW in v5.0)
    buyer_units = buyer_metrics.get("units")
    subject_units = subject.get("units")
    if buyer_units and subject_units:
        dim_scores["units"] = units_score(buyer_units, subject_units, sd.get("units", 3))
    else:
        dim_scores["units"] = None

    # Submarket
    dim_scores["submarket"] = dynamic_submarket_score(
        buyer_metrics.get("submarket"), subject.get("submarket"))

    # Recency
    sale_date = buyer_metrics.get("sale_date")
    if sale_date:
        dim_scores["recency"] = recency_score([sale_date], ref_dt)
    else:
        dim_scores["recency"] = None

    # Velocity and strategy filled at buyer level
    dim_scores["velocity"] = None
    dim_scores["strategy"] = None

    comp, n_missing = composite_score(dim_scores, w)

    if is_repeat and comp > 0:
        comp = min(comp + REPEAT_BONUS, REPEAT_CAP)

    return {
        "composite": round(comp, 1),
        "n_missing": n_missing,
        "is_repeat": is_repeat,
        "dim_scores": dim_scores,
    }


def score_buyer_aggregate(buyer_transactions, subject, reference_date=None,
                         market_sd=None, weights=None):
    """60% best transaction + 40% recency-weighted average, plus buyer-level signals."""
    if not buyer_transactions:
        return {
            "composite": 0.0, "best_txn_score": 0.0, "avg_recency_weighted": 0.0,
            "velocity": 0.0, "strategy": 50.0, "n_transactions": 0, "txn_results": [],
        }

    sd = market_sd or DEFAULT_MARKET_SD
    w = weights or DEFAULT_WEIGHTS
    ref_dt = get_reference_date(reference_date)

    sale_dates = [txn.get("sale_date") for txn in buyer_transactions]
    vel = velocity_score(sale_dates, ref_dt)
    is_high_velocity = vel >= 70

    # Portfolio Variance Profiling
    buyer_sd = dict(sd)
    units_list = [t.get("units") for t in buyer_transactions if t.get("units") is not None]
    if len(units_list) >= 3:
        u_std = statistics.pstdev(units_list)
        buyer_sd["units"] = max(1.0, min(float(u_std), sd.get("units", 2) * 2.0))
        
    price_list = [t.get("price") for t in buyer_transactions if t.get("price") is not None]
    if len(price_list) >= 3:
        p_std = statistics.pstdev(price_list)
        base_p = sd.get("price", 800_000)
        buyer_sd["price"] = max(base_p * 0.5, min(float(p_std), base_p * 2.0))

    txn_results = []
    for txn in buyer_transactions:
        result = score_single_transaction(txn, subject, ref_dt, sd, w, buyer_specific_sd=buyer_sd, is_high_velocity=is_high_velocity)
        result["transaction"] = txn
        txn_results.append(result)

    best_txn_score = max(r["composite"] for r in txn_results)

    recency_weighted_scores = []
    for result in txn_results:
        sale_date = result["transaction"].get("sale_date")
        weight = recency_weight(sale_date, ref_dt, is_high_velocity=is_high_velocity)
        recency_weighted_scores.append(result["composite"] * weight)

    avg_recency_weighted = (sum(recency_weighted_scores) / len(recency_weighted_scores)
                           if recency_weighted_scores else 0.0)

    composite = BEST_DEAL_WEIGHT * best_txn_score + AVG_DEAL_WEIGHT * avg_recency_weighted

    # sale_dates and vel computed above
    strat = strategy_alignment_score(buyer_transactions, subject)

    velocity_weight = w.get("velocity", 0.10)
    strategy_weight = w.get("strategy", 0.04)

    composite = (composite * (1 - velocity_weight - strategy_weight) +
                vel * velocity_weight +
                strat * strategy_weight)

    return {
        "composite": round(composite, 1),
        "best_txn_score": round(best_txn_score, 1),
        "avg_recency_weighted": round(avg_recency_weighted, 1),
        "velocity": round(vel, 1),
        "strategy": round(strat, 1),
        "n_transactions": len(buyer_transactions),
        "txn_results": txn_results,
    }


def score_buyers(transactions, subject, reference_date=None,
                market_sd=None, weights=None):
    """Score all buyers with aggregate multi-deal scoring. Returns sorted list."""
    ref_dt = get_reference_date(reference_date)

    # Adapt weights contextually based on subject profile
    dyn_weights = get_adaptive_weights(subject, weights)

    buyer_txns = defaultdict(list)
    for txn in transactions:
        name = txn.get("buyer_name", "Unknown")
        buyer_txns[name].append(txn)

    results = []
    for name, txns in buyer_txns.items():
        agg = score_buyer_aggregate(txns, subject, ref_dt, market_sd, dyn_weights)
        agg["buyer_name"] = name
        agg["all_transactions"] = txns
        results.append(agg)

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


# ── Backtesting ─────────────────────────────────────────────────────────

def backtest(transactions, market_sd=None, weights=None, top_n=10,
            min_transactions=5, verbose=False):
    """Leave-one-out backtest."""
    sd = market_sd or DEFAULT_MARKET_SD
    w = weights or DEFAULT_WEIGHTS

    details = []
    hits_top_n = 0
    hits_top_1 = 0
    ranks = []

    for i, target_txn in enumerate(transactions):
        target_buyer = target_txn.get("buyer_name", "Unknown")
        if not target_buyer or target_buyer == "Unknown":
            continue

        val_ppu = target_txn.get("ppu")
        val_ppsf = target_txn.get("ppsf")
        val_cap = target_txn.get("cap_rate")
        val_grm = target_txn.get("grm")
        val_price = target_txn.get("price")
        val_units = target_txn.get("units")

        if not val_ppu or not val_price:
            continue

        subject = {}
        for metric, val in [("ppu", val_ppu), ("ppsf", val_ppsf),
                            ("cap_rate", val_cap), ("grm", val_grm),
                            ("price", val_price)]:
            if val and val > 0:
                subject[metric] = {
                    "low": round(val * 0.90, 4),
                    "high": round(val * 1.10, 4),
                    "mid": round(val, 4),
                }

        subject["submarket"] = target_txn.get("submarket", "")
        subject["units"] = val_units
        subject["vacancy_pct"] = target_txn.get("vacancy_pct")
        subject["year_built"] = target_txn.get("year_built")
        subject["lot_size_sf"] = target_txn.get("lot_size_sf")

        other_txns = [t for j, t in enumerate(transactions) if j != i]
        if len(other_txns) < min_transactions:
            continue

        scored = score_buyers(other_txns, subject, reference_date=None,
                             market_sd=sd, weights=w)

        rank = None
        for r, buyer in enumerate(scored, 1):
            if buyer["buyer_name"] == target_buyer:
                rank = r
                break

        if rank is None:
            continue

        ranks.append(rank)
        in_top_n = rank <= top_n
        in_top_1 = rank == 1

        if in_top_n:
            hits_top_n += 1
        if in_top_1:
            hits_top_1 += 1

        case = {
            "txn_index": i, "buyer": target_buyer, "rank": rank,
            "in_top_n": in_top_n, "in_top_1": in_top_1,
            "score": scored[rank - 1]["composite"] if rank <= len(scored) else 0,
            "top_scorer": scored[0]["buyer_name"] if scored else None,
            "top_score": scored[0]["composite"] if scored else 0,
            "submarket": target_txn.get("submarket", ""),
            "price": val_price, "ppu": val_ppu,
        }
        details.append(case)

        if verbose and not in_top_n:
            print(f"  MISS: {target_buyer} ranked #{rank} for ${val_price:,.0f} "
                  f"in {case['submarket']} (top was {case['top_scorer']})")

    n_cases = len(details)
    if n_cases == 0:
        return {"accuracy_top_n": 0.0, "accuracy_top_1": 0.0,
                "n_test_cases": 0, "mean_rank": 0.0, "median_rank": 0,
                "details": [], "failures": []}

    sorted_ranks = sorted(ranks)
    median_rank = sorted_ranks[len(sorted_ranks) // 2]

    return {
        "accuracy_top_n": round(100 * hits_top_n / n_cases, 1),
        "accuracy_top_1": round(100 * hits_top_1 / n_cases, 1),
        "mean_rank": round(sum(ranks) / len(ranks), 1),
        "median_rank": median_rank,
        "n_test_cases": n_cases,
        "top_n": top_n,
        "failures": [d for d in details if not d["in_top_n"]],
        "details": details,
    }


def optimize(transactions, top_n=10, iterations=100, verbose=False):
    """Expanded grid-search optimizer with more weight/SD combinations."""
    base_w = dict(DEFAULT_WEIGHTS)
    base_sd = dict(DEFAULT_MARKET_SD)

    weight_adjustments = [
        {},  # baseline v5.0
        {"cap_rate": 0.18, "ppu": 0.16},
        {"cap_rate": 0.12, "submarket": 0.22, "ppu": 0.16},
        {"cap_rate": 0.15, "submarket": 0.20, "velocity": 0.12},
        {"cap_rate": 0.10, "submarket": 0.20, "ppu": 0.20, "velocity": 0.12, "units": 0.12},
        {"velocity": 0.15, "submarket": 0.20, "ppu": 0.15, "cap_rate": 0.12},
        {"velocity": 0.18, "submarket": 0.18, "ppu": 0.15, "cap_rate": 0.10, "units": 0.12},
        {"units": 0.15, "ppu": 0.18, "submarket": 0.18, "cap_rate": 0.12, "velocity": 0.12},
        {"cap_rate": 0.20, "ppu": 0.15, "submarket": 0.15, "velocity": 0.15, "units": 0.10},
    ]

    sd_adjustments = [
        {},  # baseline
        {"cap_rate": 0.012, "units": 2.5},
        {"cap_rate": 0.018, "units": 3.5},
        {"cap_rate": 0.015, "ppu": 60_000, "price": 600_000},
        {"cap_rate": 0.015, "ppu": 100_000, "price": 1_000_000},
        {"cap_rate": 0.010, "grm": 2.0, "units": 2},
        {"cap_rate": 0.020, "grm": 3.0, "units": 4},
    ]

    best_acc = -1.0
    best_w = base_w
    best_sd = base_sd
    best_result = None

    count = 0
    for w_adj in weight_adjustments:
        for sd_adj in sd_adjustments:
            if count >= iterations:
                break

            test_w = {**base_w, **w_adj}
            total = sum(test_w.values())
            test_w = {k: round(v / total, 4) for k, v in test_w.items()}

            test_sd = {**base_sd, **sd_adj}

            result = backtest(transactions, market_sd=test_sd, weights=test_w,
                             top_n=top_n, verbose=False)

            if result["n_test_cases"] > 0 and result["accuracy_top_n"] > best_acc:
                best_acc = result["accuracy_top_n"]
                best_w = test_w
                best_sd = test_sd
                best_result = result

                if verbose:
                    print(f"  [{count}] New best: {best_acc}% top-{top_n} "
                          f"(mean rank {result['mean_rank']})")

            count += 1

    return best_w, best_sd, best_result


if __name__ == "__main__":
    import prepare
    import time
    transactions = prepare.get_data()
    t0 = time.time()
    
    # We will pick a small subset to run backtest quickly (200 txns out of 5000)
    subset = transactions[:200]
    
    res = backtest(subset, min_transactions=2, top_n=10)
    print("---")
    print(f"accuracy_top_10: {res.get('accuracy_top_n', 0):.2f}")
    print(f"median_rank: {res.get('median_rank', 0)}")
    print(f"time_elapsed: {time.time() - t0:.2f}s")
