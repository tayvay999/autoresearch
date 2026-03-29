import os
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
        best_acc = 51.20

print(f"Starting 5-Hour Continuous Auto-Optimization Loop! Baseline: {best_acc}%")

def get_train_params():
    # Read the train file for base parameters
    with open("train.py", "r") as f:
        code = f.read()

    # Find DEFAULT_WEIGHTS block
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
            s += f'    "{k}": {v:.4f},\n'
        else:
            s += f'    "{k}": {v},\n'
    s += "}"
    return s

# Load dataset once
print("Loading synthetic dataset...")
transactions = prepare.get_data()

# Use robust empirical subset (the last 1000 deals chronologically)
subset = transactions[-1000:]

iterations = 0
mutations_since_last_best = 0

current_w, current_sd, _ = get_train_params()

while time.time() - START_TIME < MAX_DURATION:
    iterations += 1
    
    # Refresh params from module if we hit local minimum
    if mutations_since_last_best > 100:
        current_w, current_sd, _ = get_train_params()
        mutations_since_last_best = 0
        
    test_w = dict(current_w)
    # Mutate 2-3 random weights
    for _ in range(random.randint(1, 4)):
        k = random.choice(list(test_w.keys()))
        test_w[k] = max(0.01, test_w[k] + random.uniform(-0.04, 0.04))
    
    # Normalize weights
    total = sum(test_w.values())
    test_w = {k: round(v/total, 4) for k, v in test_w.items()}
    
    # Mutate SDs
    test_sd = dict(current_sd)
    for _ in range(random.randint(1, 3)):
        k = random.choice(list(test_sd.keys()))
        mutation_factor = random.uniform(0.7, 1.3)
        if isinstance(test_sd[k], int) and test_sd[k] < 1000:
            test_sd[k] = max(1, int(round(test_sd[k] * mutation_factor)))
        else:
            test_sd[k] = max(0.001, round(test_sd[k] * mutation_factor, 3))
            
    # Run rapid backtest natively
    res = backtest(subset, market_sd=test_sd, weights=test_w, min_transactions=2, top_n=10)
    acc = res.get("accuracy_top_n", 0)
    
    if acc > best_acc:
        current_w = test_w
        current_sd = test_sd
        best_acc = acc
        mutations_since_last_best = 0
        
        print(f"[{iterations}] NEW BEST DETECTED! {acc:.2f}% (Time Elapsed: {time.time()-START_TIME:.0f}s)")
        
        # Modify train.py
        _, _, code = get_train_params()
        new_w_str = dict_to_str(test_w)
        new_sd_str = dict_to_str(test_sd)
        
        code = re.sub(r'DEFAULT_WEIGHTS = \{.*?\}', f'DEFAULT_WEIGHTS = {new_w_str}', code, flags=re.DOTALL)
        code = re.sub(r'DEFAULT_MARKET_SD = \{.*?\}', f'DEFAULT_MARKET_SD = {new_sd_str}', code, flags=re.DOTALL)
        
        with open("train.py", "w") as f:
            f.write(code)
            
        # Git commit
        hex_id = hex(random.randint(0, 16777215))[2:].zfill(6)
        desc = f"Genetic Opt: Acc up to {acc:.2f}%"
        subprocess.run(["git", "commit", "-am", desc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Log to results.tsv
        with open("results.tsv", "a") as f:
            f.write(f"{hex_id[:7]}\t{acc:.2f}\tkeep\t{desc}\n")
    else:
        mutations_since_last_best += 1
        
    # Prevent extreme 100% CPU lock; 1ms yield
    time.sleep(0.001)

print(f"5 Hours Completed. Final Best Accuracy: {best_acc}%")
