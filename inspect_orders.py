
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")

print("Shape:", orders.shape)

print("\nColumn names:")
print(orders.columns.tolist())

print("\nFirst 10 rows:")
print(orders.head(10).to_string(index=False))

print("\nData types:")
print(orders.dtypes)

print("\nMissing values:")
print(orders.isna().sum())

print("\nOrder status counts:")
print(orders["status"].value_counts(dropna=False))

print("\nDuplicate rows:", orders.duplicated().sum())
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())
