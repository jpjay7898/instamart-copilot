
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Sort records by order ID and timestamp
orders = orders.sort_values(
    ["order_id", "order_ts"]
)

# Show the status history for a few repeated orders
for order_id in orders["order_id"].value_counts().loc[
    lambda x: x > 1
].index[:10]:

    print("\nOrder ID:", order_id)

    print(
        orders.loc[
            orders["order_id"] == order_id,
            ["order_ts", "status", "order_amount", "delivery_minutes"]
        ].to_string(index=False)
    )

# Examine the final status distribution
latest_orders = orders.drop_duplicates(
    subset="order_id", keep="last"
)

print("\nLatest record status counts:")
print(latest_orders["status"].value_counts())

print("\nLatest records shape:", latest_orders.shape)
