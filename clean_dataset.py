import pandas as pd

# Load the dataset
df = pd.read_csv('student_depression_dataset.csv')

# Create a new dataframe with only the columns we want to keep
cleaned_df = pd.DataFrame()

# Keep continuous/numerical variables as-is
cleaned_df['Age'] = df['Age']
cleaned_df['Academic Pressure'] = df['Academic Pressure']
cleaned_df['Work Pressure'] = df['Work Pressure']
cleaned_df['CGPA'] = df['CGPA']
cleaned_df['Study Satisfaction'] = df['Study Satisfaction']
cleaned_df['Job Satisfaction'] = df['Job Satisfaction']
cleaned_df['Work/Study Hours'] = df['Work/Study Hours']
cleaned_df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')

# Convert Sleep Duration to mid-point hours
def convert_sleep_duration(value):
    if pd.isna(value):
        return None
    value = str(value).strip().strip("'\"")
    if 'Less than 5' in value:
        return 4.0  # Mid-point approximation for less than 5
    elif '5-6' in value or '5–6' in value:
        return 5.5
    elif '7-8' in value or '7–8' in value:
        return 7.5
    elif 'More than 8' in value:
        return 9.0  # Mid-point approximation for more than 8
    else:
        return None

cleaned_df['Sleep Duration'] = df['Sleep Duration'].apply(convert_sleep_duration)

# Convert Suicidal Thoughts: Yes = 1, No = 0
cleaned_df['Suicidal Thoughts'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# Convert Family History of Mental Illness: Yes = 1, No = 0
cleaned_df['Family History'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

# Convert Gender to single binary variable: Is Female? Yes = 1, No (Male) = 0
cleaned_df['Is Female'] = (df['Gender'] == 'Female').astype(int)

# Convert Dietary Habits: Unhealthy = 0, Moderate = 1, Healthy = 2
cleaned_df['Dietary Habits'] = df['Dietary Habits'].map({'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2})

# Include Depression column (target variable)
cleaned_df['Depression'] = df['Depression']

# Save the cleaned dataset
cleaned_df.to_csv('student_depression_cleaned.csv', index=False)

print("Cleaned dataset saved to 'student_depression_cleaned.csv'")
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {cleaned_df.shape}")
print(f"\nColumns in cleaned dataset:")
for col in cleaned_df.columns:
    print(f"  - {col}")

print(f"\nFirst 5 rows of cleaned dataset:")
print(cleaned_df.head().to_string())

# Check for any missing values
print(f"\nMissing values per column:")
print(cleaned_df.isnull().sum())
