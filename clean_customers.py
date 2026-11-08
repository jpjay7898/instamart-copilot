
import pandas as pd
from pathlib import Path

# Load raw customer data
customers = pd.read_csv("data/raw/customers.csv")

# 1. Standardize city names
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# 2. Standardize gender values
def clean_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"

    value = str(value).strip().lower()

    if value in ["m", "male"]:
        return "male"
    elif value in ["f", "female"]:
        return "female"
    else:
        return "unknown"

customers["gender"] = customers["gender"].apply(clean_gender)

# 3. Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Treat ages outside 10–100 as missing
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing ages with the median for each city
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# 4. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# 5. Save the cleaned customer data
output_path = Path("data/clean")
output_path.mkdir(parents=True, exist_ok=True)

customers.to_csv(
    output_path / "customers.csv",
    index=False,
    date_format="%Y-%m-%d"
)

# Display results for verification
print("Cleaned customer shape:", customers.shape)
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print(
    "Minimum age:", customers["age"].min(),
    "| Maximum age:", customers["age"].max()
)
print("\nSaved to data/clean/customers.csv")
