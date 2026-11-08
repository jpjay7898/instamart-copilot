import sqlite3
import pandas as pd

# Connect to the Part A database
conn = sqlite3.connect("data/instamart.db")

# Read completed orders
query = """
SELECT
    customer_id,
    order_id,
    order_ts,
    order_amount
FROM orders
WHERE status = 'completed'
"""

orders = pd.read_sql_query(query, conn)
conn.close()

# Convert order timestamp into datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Calculate RFM values
rfm = orders.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency = days since the last completed order
rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days

# Keep required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display results
print("RFM Data Shape:")
print(rfm.shape)

print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

# Save RFM data
rfm.to_csv("outputs/customer_rfm.csv", index=False)

print("\nRFM data saved successfully!")