
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/clean/products.csv")

# Convert order amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Join product prices with orders
check = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)

# Calculate expected amount
check["expected_amount"] = (
    check["price_inr"]
    * check["quantity"]
    * (1 - check["discount_pct"] / 100)
)

# Find differences
check["amount_difference"] = (
    check["order_amount"] - check["expected_amount"]
).abs()

print("Rows with missing amounts:", check["order_amount"].isna().sum())

print(
    "Rows where amount differs from calculated amount:",
    (check["amount_difference"] > 0.01).sum()
)

print("\nSample amount comparison:")
print(
    check[
        ["product_id", "quantity", "discount_pct",
         "order_amount", "expected_amount"]
    ].head(10).to_string(index=False)
)
