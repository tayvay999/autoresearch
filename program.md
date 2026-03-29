# autoresearch (Buyer Match Algorithm Edition)

This repo tasks the LLM with autonomous empirical research to continuously optimize the Buyer Matching logic.

## Setup

1. **Agree on a run tag**: e.g., `git checkout -b optimize/mar29`.
2. **Read codebase**: The repo has:
    - `prepare.py` — generates a synthetic transaction dataset (fixed, do not modify).
    - `train.py` — the buyer match algorithm that ranks buyers against a subject building.
3. **Initialize `results.tsv`**: Header row: `commit\taccuracy_top_10\tstatus\tdescription`.

## Experimentation

The `train.py` file evaluates your algorithmic changes by backtesting historical matches. It prints standard outputs at the bottom of the script.

**What you CAN do:**
- Edit `train.py`. Modify weights, standard deviations, dimensions, gaussian equations, or any internal algorithms to create a smarter system.
- Radically alter logic if needed.

**What you CANNOT do:**
- Do not modify `prepare.py`.
- Do not bypass the `train.py` evaluation bounds.

**The goal is simple: maximize `accuracy_top_10`.** (Higher is better!)

## Output format
The script prints the accuracy summary at the very end:
```
---
accuracy_top_10: 55.40
median_rank: 5
```
You identify accuracy via:
```bash
grep "^accuracy_top_10:" run.log
```

## Logging results
Log to `results.tsv` after every hypothesis.
`status` must be `keep` or `discard`.

## The experiment loop

LOOP FOREVER:
1. `git status`
2. Formulate hypothesis and modify `train.py`
3. `git commit -am "test message"`
4. Execute: `python3 train.py > run.log 2>&1`
5. Read result: `grep "^accuracy_top_10:" run.log`
6. If the score is higher than baseline, **KEEP**. If worse or equal, **DISCARD** (`git reset HEAD~1 --hard`). If crashed, discard.
7. Append row to `results.tsv`
8. Loop. Never stop unless physically interrupted by the user.
