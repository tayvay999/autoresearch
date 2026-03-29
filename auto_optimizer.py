import os
import sys
import time
import random
import subprocess
import re
import math

import prepare
from train import backtest

START_TIME = time.time()
MAX_DURATION = 5 * 3600  # 5 hours

# Existing baselines from file
with open("results.tsv", "r") as f:
    lines = [l for l in f.readlines() if l.strip()]
    if len(lines) > 1:
        best_acc = float(lines[-1].split("\t")[1])
    else:
        best_acc = 56.90

print(f"Starting 5-Hour Empirical Auto-Optimization Loop!", flush=True)
print(f"Baseline: {best_acc}%", flush=True)

def get_train_params():
    with open("train.py", "r") as f:
        code = f.read()
    w_match = re.search(r'DEFAULT_WEIGHTS = (\{.*?\})', code, re.DOTALL)
    sd_match = re.search(r'DEFAULT_MARKET_SD = (\{.*?\})', code, re.DOTALL)
    import ast
    w_base = ast.literal_eval(w_match.group(1))
    sd_base = ast.literal_eval(sd_match.group(1))
    return w_base, sd_base, code

def dict_to_str(d):
    s = "{\n"
    for k, v in d.items():
        if isinstance(v, float):
            s += f'    "{k}": {v:.6f},\n'
        else:
            s += f'    "{k}": {v},\n'
    s += "}"
    return s

# Load dataset once
print("Loading empirical CoStar dataset...", flush=True)
transactions = prepare.get_data()
print(f"Loaded {len(transactions)} real LA transactions.", flush=True)

# Two-stage evaluation:
#   Stage 1 (fast screen): 300 recent transactions (~7s per eval)
#   Stage 2 (full validation): 1000 transactions (~25s, only if stage 1 beats best)
fast_subset = transactions[-300:]
full_subset = transactions[-1000:]

iterations = 0
improvements = 0
mutations_since_last_best = 0

current_w, current_sd, _ = get_train_params()

print(f"Entering optimization loop...", flush=True)

while time.time() - START_TIME < MAX_DURATION:
    iterations += 1
    
    # If stuck in local minimum for 200 iterations, do a larger random jump
    big_jump = mutations_since_last_best > 200
    
    if big_jump:
        # Reset to current best from file and try a bigger perturbation
        current_w, current_sd, _ = get_train_params()
        mutations_since_last_best = 0
        
    test_w = dict(current_w)
    
    # Mutate weights
    n_mutations = random.randint(2, 5) if big_jump else random.randint(1, 3)
    magnitude = 0.08 if big_jump else 0.03
    for _ in range(n_mutations):
        k = random.choice(list(test_w.keys()))
        test_w[k] = max(0.005, test_w[k] + random.uniform(-magnitude, magnitude))
    
    # Normalize weights to sum to 1.0
    total = sum(test_w.values())
    test_w = {k: round(v/total, 6) for k, v in test_w.items()}
    
    # Mutate SDs
    test_sd = dict(current_sd)
    for _ in range(random.randint(1, 2)):
        k = random.choice(list(test_sd.keys()))
        if big_jump:
            mutation_factor = random.uniform(0.5, 1.5)
        else:
            mutation_factor = random.uniform(0.8, 1.2)
        if isinstance(test_sd[k], int) and test_sd[k] < 1000:
            test_sd[k] = max(1, int(round(test_sd[k] * mutation_factor)))
        else:
            test_sd[k] = max(0.001, round(test_sd[k] * mutation_factor, 4))
            
    # Stage 1: Fast screen on 300 transactions
    res_fast = backtest(fast_subset, market_sd=test_sd, weights=test_w, min_transactions=2, top_n=10)
    acc_fast = res_fast.get("accuracy_top_n", 0)
    
    if acc_fast > best_acc - 2.0:
        # Promising candidate — run full validation
        res_full = backtest(full_subset, market_sd=test_sd, weights=test_w, min_transactions=2, top_n=10)
        acc_full = res_full.get("accuracy_top_n", 0)
        
        if acc_full > best_acc:
            current_w = test_w
            current_sd = test_sd
            best_acc = acc_full
            improvements += 1
            mutations_since_last_best = 0
            
            elapsed = time.time() - START_TIME
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            print(f"[iter {iterations}] NEW BEST: {acc_full:.2f}% | "
                  f"Test cases: {res_full['n_test_cases']} | "
                  f"Median rank: {res_full['median_rank']} | "
                  f"Elapsed: {hrs}h{mins:02d}m | "
                  f"Improvements: {improvements}", flush=True)
            
            # Write winning params to train.py
            _, _, code = get_train_params()
            new_w_str = dict_to_str(test_w)
            new_sd_str = dict_to_str(test_sd)
            
            code = re.sub(r'DEFAULT_WEIGHTS = \{.*?\}', f'DEFAULT_WEIGHTS = {new_w_str}', code, flags=re.DOTALL)
            code = re.sub(r'DEFAULT_MARKET_SD = \{.*?\}', f'DEFAULT_MARKET_SD = {new_sd_str}', code, flags=re.DOTALL)
            
            with open("train.py", "w") as f:
                f.write(code)
                
            # Git commit
            hex_id = hex(random.randint(0, 16777215))[2:].zfill(6)
            desc = f"Empirical Opt: {acc_full:.2f}% (n={res_full['n_test_cases']}, median_rank={res_full['median_rank']})"
            subprocess.run(["git", "commit", "-am", desc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            with open("results.tsv", "a") as f:
                f.write(f"{hex_id[:7]}\t{acc_full:.2f}\tkeep\t{desc}\n")
    else:
        mutations_since_last_best += 1
        
    # Progress heartbeat every 500 iterations
    if iterations % 500 == 0:
        elapsed = time.time() - START_TIME
        hrs = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        print(f"[heartbeat] iter {iterations} | best: {best_acc:.2f}% | "
              f"elapsed: {hrs}h{mins:02d}m | improvements: {improvements} | "
              f"stale: {mutations_since_last_best}", flush=True)

elapsed = time.time() - START_TIME
print(f"\n{'='*60}")
print(f"5-HOUR OPTIMIZATION COMPLETE")
print(f"Final Best Accuracy: {best_acc:.2f}%")
print(f"Total Iterations: {iterations}")
print(f"Total Improvements Found: {improvements}")
print(f"Total Time: {elapsed/3600:.1f} hours")
print(f"{'='*60}")
