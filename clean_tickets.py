
import pandas as pd
from pathlib import Path

# Load tickets without converting empty strings into NaN
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

# Remove extra spaces from text columns
for column in tickets.columns:
    tickets[column] = tickets[column].astype("string").str.strip()

# Keep empty order IDs as missing values
tickets["order_id"] = tickets["order_id"].replace("", pd.NA)

# Keep empty ticket text as an empty string
tickets["text"] = tickets["text"].fillna("")

# Standardize category names
tickets["true_category"] = (
    tickets["true_category"].str.strip().str.lower()
)

# Check duplicates
print("Duplicate ticket IDs:", tickets["ticket_id"].duplicated().sum())

# Check blank values
print("\nBlank values:")
print(tickets.replace("", pd.NA).isna().sum())

# Category counts
print("\nTicket categories:")
print(tickets["true_category"].value_counts())

# Save cleaned tickets
output_path = Path("data/clean")
output_path.mkdir(parents=True, exist_ok=True)

tickets.to_csv(
    output_path / "tickets.csv",
    index=False
)

print("\nCleaned tickets shape:", tickets.shape)
print("Saved to data/clean/tickets.csv")
