import pandas as pd
from scipy import stats

# Load the cleaned dataset
df = pd.read_csv('student_depression_cleaned.csv')

# Z-score normalize all columns except Depression
normalized_df = pd.DataFrame()

for col in df.columns:
    if col == 'Depression':
        # Keep Depression as-is (not normalized)
        normalized_df[col] = df[col]
    else:
        # Calculate z-score: (x - mean) / std
        normalized_df[col] = stats.zscore(df[col], nan_policy='omit')

# Save the normalized dataset
normalized_df.to_csv('student_depression_normalized.csv', index=False)

print("Normalized dataset saved to 'student_depression_normalized.csv'")
print(f"\nDataset shape: {normalized_df.shape}")
print(f"\nColumns in normalized dataset:")
for col in normalized_df.columns:
    print(f"  - {col}")

print(f"\nFirst 5 rows of normalized dataset:")
print(normalized_df.head().to_string())

# Verify normalization (mean ≈ 0, std ≈ 1)
print(f"\nVerification (mean and std for each column):")
print("-" * 50)
for col in df.columns:
    mean = normalized_df[col].mean()
    std = normalized_df[col].std()
    print(f"{col:25s} mean: {mean:8.4f}  std: {std:.4f}")

print(f"\n{'Depression':25s} (not normalized - kept as reference)")
