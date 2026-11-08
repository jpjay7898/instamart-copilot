
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")

# Find order IDs that appear more than once
duplicate_ids = orders[
    orders.duplicated(subset="order_id", keep=False)
]

print("Rows with repeated order IDs:", len(duplicate_ids))

print("\nFirst 20 rows with repeated order IDs:")
print(
    duplicate_ids.sort_values("order_id")
    .head(20)
    .to_string(index=False)
)

# Check how many unique products appear per order ID
products_per_order = (
    orders.groupby("order_id")["product_id"].nunique()
)

print("\nOrder IDs with multiple products:")
print((products_per_order > 1).sum())

print("\nOrder IDs with only one product:")
print((products_per_order == 1).sum())
