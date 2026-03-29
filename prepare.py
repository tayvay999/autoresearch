import os
import random
import datetime

# Fixed variables
NUM_FAKE_TRANSACTIONS = 5000
SUBMARKETS = ["Hollywood", "Koreatown", "East Hollywood", "Silver Lake", "Downtown LA", "Westlake", "Echo Park", "Los Feliz"]

def generate_synthetic_transactions():
    """Generates a stable list of realistic transactions to score the algorithm against."""
    random.seed(42)  # Deterministic dataset for comparable tests!
    transactions = []
    
    # Buyer archetypes
    buyer_profiles = {
        "K-Town Focused": {"submarkets": ["Koreatown"], "ppu_target": 250000, "units_target": 12},
        "Value-Add Dev": {"submarkets": SUBMARKETS, "ppu_target": 150000, "units_target": 8, "vac_pref": 0.40, "cap_pref": 0.03},
        "Core Hollywood": {"submarkets": ["Hollywood", "East Hollywood"], "ppu_target": 350000, "units_target": 20, "vac_pref": 0.05, "cap_pref": 0.055},
        "Mid-Size Any": {"submarkets": SUBMARKETS, "ppu_target": 200000, "units_target": 10},
        "Institutional": {"submarkets": ["Downtown LA", "Hollywood", "Koreatown"], "ppu_target": 400000, "units_target": 100},
    }
    
    buyer_names = []
    for profile_name, kwargs in buyer_profiles.items():
        for i in range(20):
            buyer_names.append((f"{profile_name} Buyer {i}", kwargs))
            
    base_date = datetime.datetime(2025, 1, 1)
    
    for buyer_name, profile in buyer_names:
        num_deals_for_buyer = random.randint(3, 15)
        for _ in range(num_deals_for_buyer):
            sub = random.choice(profile["submarkets"])
            ppu = profile["ppu_target"] * random.uniform(0.85, 1.15)
            units = max(2, int(profile["units_target"] * random.uniform(0.7, 1.3)))
            price = ppu * units
            cap = profile.get("cap_pref", 0.045) * random.uniform(0.9, 1.1)
            grm = 1 / cap * 0.6 if cap > 0 else 15
            vac = profile.get("vac_pref", 0.05) * random.uniform(0.8, 1.2)
            yr = 1970 + random.randint(-20, 30)
            
            days_ago = random.randint(10, 800)
            sale_dt = base_date - datetime.timedelta(days=days_ago)
            
            txn = {
                "buyer_name": buyer_name,
                "ppu": ppu,
                "ppsf": ppu / random.uniform(600, 1000),
                "cap_rate": cap,
                "grm": grm,
                "price": price,
                "units": units,
                "submarket": sub,
                "sale_date": sale_dt.isoformat(),
                "vacancy_pct": vac,
                "year_built": yr,
                "lot_size_sf": units * random.uniform(800, 1200)
            }
            transactions.append(txn)
    
    # Shuffle so that slicing doesn't grab just one buyer
    random.shuffle(transactions)
    return transactions

def get_data():
    return generate_synthetic_transactions()

if __name__ == "__main__":
    t = generate_synthetic_transactions()
    print(f"Generated {len(t)} synthetic transactions for evaluation.")
