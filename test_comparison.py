import pandas as pd
from app.utils import compare_dataframes

# Create test data matching your scenario
# File A: ID 46, 47, 48, 49, 50 appear ONCE each
df_a = pd.DataFrame({
    'id': [46, 47, 48, 49, 50],
    'name': ['A1', 'A2', 'A3', 'A4', 'A5'],
    'value': [100, 200, 300, 400, 500]
})

# File B: ID 46, 47, 48, 49, 50 appear TWICE each
df_b = pd.DataFrame({
    'id': [46, 46, 47, 47, 48, 48, 49, 49, 50, 50],
    'name': ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5', 'B5'],
    'value': [100, 999, 200, 888, 300, 777, 400, 666, 500, 555]
})

print("File A:")
print(df_a)
print("\nFile B:")
print(df_b)

# Run comparison
result = compare_dataframes(df_a, df_b, key_cols=['id'], numeric_tolerance=1e-9)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)

print(f"\nMatched (no differences): {result['stats']['matched_no_diff_count']}")
for i, row in enumerate(result['matched_records']):
    print(f"  {i+1}. {row}")

print(f"\nMatched (with differences): {result['stats']['matched_with_diff_count']}")
for i, diff in enumerate(result['matched_with_differences']):
    print(f"  {i+1}. Key: {diff['key']}")
    for col, vals in diff['columns'].items():
        if vals['is_different']:
            print(f"      {col}: {vals['value_a']} → {vals['value_b']}")

print(f"\nOnly in A: {result['stats']['only_in_a_count']}")
for i, row in enumerate(result['only_in_a']):
    print(f"  {i+1}. {row}")

print(f"\nOnly in B: {result['stats']['only_in_b_count']}")
for i, row in enumerate(result['only_in_b']):
    print(f"  {i+1}. {row}")

print("\n" + "="*80)
print("EXPECTED RESULTS:")
print("="*80)
print("Matched (no differences): 5 (ID 46-50, 1st occurrence each)")
print("Matched (with differences): 0")
print("Only in A: 0")
print("Only in B: 5 (ID 46-50, 2nd occurrence each)")
