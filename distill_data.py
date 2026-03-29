import pandas as pd
import prepare
import statistics
from collections import Counter

data = prepare.get_data()
df = pd.DataFrame(data)

print(f"Total Transactions: {len(df)}")
print(f"Unique Buyers: {df['buyer_name'].nunique()}")

# Buyer Velocity
buyer_counts = Counter(df['buyer_name'])
repeat_buyers = {k: v for k, v in buyer_counts.items() if v >= 2}
print(f"Repeat Buyers (>=2 deals): {len(repeat_buyers)}")
print(f"Top 5 Most Active Buyers: {Counter(df['buyer_name']).most_common(5)}")

# Data Quality (Missingness)
metrics = ['price', 'units', 'ppu', 'cap_rate', 'grm', 'ppsf', 'year_built', 'vacancy_pct']
print("\nData Fill Rates:")
for m in metrics:
    filled = df[m].notna().sum()
    pct = (filled / len(df)) * 100
    print(f"  {m:12}: {pct:5.1f}% filled")

# Strategy Distribution
print("\nStrategy Distribution:")
print(df['strategy'].value_counts(normalize=True))

# Submarket Distribution
print("\nTop 5 Submarkets:")
print(df['submarket'].value_counts().head(5))

# Variance Analysis
print("\nCoefficient of Variation (SD / Mean):")
for m in metrics:
    if df[m].notna().sum() > 1:
        mean_val = df[m].mean()
        std_val = df[m].std()
        cv = std_val / mean_val if mean_val != 0 else 0
        print(f"  {m:12}: {cv:.4f}")
