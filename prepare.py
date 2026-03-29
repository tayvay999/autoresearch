import pandas as pd
import os
import glob
from math import isnan
import pickle
import warnings
warnings.filterwarnings('ignore') # Suppress openpyxl warnings

def get_data():
    cache_path = "emp_transactions_cache.pkl"
    
    # Fast load caching for the optimizer loop
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
            
    # If no cache, load from real empirical CoStar records
    folder_path = "/Users/thegroupcre/Desktop/Costar Sales Exports 2022-Current 2026 March"
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    dfs = []
    for file in files:
        dfs.append(pd.read_excel(file))
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Filter valid buyers (Try "True" company first, fallback to basic Company field)
    full_df['Final Buyer'] = full_df['Buyer (True) Company'].fillna(full_df['Buyer Company'])
    full_df = full_df.dropna(subset=['Final Buyer', 'Sale Price', 'Number Of Units'])
    
    # Exclude institutional blind trusts / useless aggregates
    exclusions = ['', 'unknown', 'various', 'private', 'individual']
    full_df = full_df[~full_df['Final Buyer'].astype(str).str.lower().isin(exclusions)]

    transactions = []
    for _, row in full_df.iterrows():
        buyer_id = str(row['Final Buyer']).strip()
        price = float(row['Sale Price'])
        units = float(row.get('Number Of Units', 0))
        
        # Must have units for multifamily matching
        if pd.isna(units) or units <= 0:
            continue
            
        ppu = price / units
        
        # Parse numeric edge cases
        ppsf = float(row['Price Per SF']) if 'Price Per SF' in row and pd.notna(row['Price Per SF']) else None
        
        # CoStar "Actual Cap Rate" is usually e.g., "5.4" rather than "0.054". Needs dividing by 100.
        cap_rate = float(row['Actual Cap Rate']) / 100.0 if 'Actual Cap Rate' in row and pd.notna(row['Actual Cap Rate']) else None
        grm = float(row['GRM']) if 'GRM' in row and pd.notna(row['GRM']) else None
        year_built = int(row['Year Built']) if 'Year Built' in row and pd.notna(row['Year Built']) else None
        
        vacancy = float(row['Vacancy']) / 100.0 if 'Vacancy' in row and pd.notna(row['Vacancy']) else 0.05
        
        date = pd.to_datetime(row.get('Sale Date'))
        date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "2024-01-01"
        
        submarket = str(row.get('Submarket Name', '')).strip()
        
        # Determine Value-Add or Stabilized Empirically
        is_value_add = False
        if pd.notna(vacancy) and vacancy > 0.15:
            is_value_add = True
        elif cap_rate is not None and 0.0 < cap_rate < 0.045:  # Buying sub-4.5 cap usually implies value-add
            is_value_add = True
        elif year_built is not None and year_built < 1980 and ppu < 250000:
            is_value_add = True
            
        strategy = "Value-Add" if is_value_add else "Stabilized"
        
        t = {
            "buyer_name": buyer_id,
            "sale_date": date_str,
            "price": price,
            "units": units,
            "submarket": submarket,
            "strategy": strategy,
            "cap_rate": cap_rate,
            "grm": grm,
            "ppu": ppu,
            "ppsf": ppsf,
            "year_built": year_built,
            "vacancy_pct": vacancy
        }
        transactions.append(t)
        
    # Sort chronologically 
    transactions.sort(key=lambda x: x["sale_date"])
    
    # Cache the payload
    with open(cache_path, "wb") as f:
        pickle.dump(transactions, f)
        
    return transactions
