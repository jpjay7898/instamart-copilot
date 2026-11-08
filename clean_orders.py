
import pandas as pd
from pathlib import Path

# Load raw orders and cleaned products
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/clean/products.csv")

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Convert columns to appropriate data types
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Convert invalid negative delivery times to missing
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Keep the latest record for each order ID
orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)

# Join product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Calculate expected order amount
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
)

# Fill missing order amounts
orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# Fill missing delivery times for completed orders
completed_mask = orders["status"] == "completed"

# Calculate median delivery time for each product category
category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)

# Fill missing completed delivery times using category median
orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"].fillna(
        orders.loc[completed_mask, "category"].map(
            category_medians
        )
    )
)

# Remove helper columns
orders = orders.drop(
    columns=["price_inr", "category"]
)

# Save cleaned data
output_path = Path("data/clean")
output_path.mkdir(parents=True, exist_ok=True)

orders.to_csv(
    output_path / "orders.csv",
    index=False,
    date_format="%Y-%m-%d %H:%M:%S"
)

# Display validation results
print("Cleaned orders shape:", orders.shape)

print("\nStatus counts:")
print(orders["status"].value_counts())

print("\nMissing delivery values by status:")
print(
    orders.groupby("status")["delivery_minutes"]
    .apply(lambda x: x.isna().sum())
)

print("\nMissing order amounts:")
print(orders["order_amount"].isna().sum())

print("\nDuplicate order IDs:")
print(orders["order_id"].duplicated().sum())

print("\nSaved to data/clean/orders.csv")
