
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("data/instamart.db")

# Load cleaned customer information
customers = pd.read_sql_query(
    "SELECT customer_id, city, age, gender, signup_date FROM customers",
    conn
)

# Load final orders with product categories
orders = pd.read_sql_query("""
    SELECT
        o.order_id,
        o.customer_id,
        o.order_ts,
        o.status,
        o.order_amount,
        o.discount_pct,
        o.delivery_minutes,
        p.category
    FROM orders o
    LEFT JOIN products p
        ON o.product_id = p.product_id
""", conn)

conn.close()

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define cutoff and September observation window
cutoff = pd.Timestamp("2025-09-01 00:00:00")
september_end = pd.Timestamp("2025-10-01 00:00:00")

# Orders before September 1: used for features
past_orders = orders[orders["order_ts"] < cutoff].copy()

# Completed orders before September 1
past_completed = past_orders[
    past_orders["status"] == "completed"
].copy()

# Customers with at least one completed order before cutoff
cohort_ids = past_completed["customer_id"].unique()

# September completed orders: used ONLY for target label
september_completed = orders[
    (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < september_end)
    & (orders["status"] == "completed")
]

september_active_ids = set(
    september_completed["customer_id"].unique()
)

# Calculate RFM features using past completed orders
rfm = past_completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    cutoff - rfm["last_order_date"]
).dt.days

# Average discount using all final orders before cutoff
discount = past_orders.groupby("customer_id").agg(
    avg_discount_pct=("discount_pct", "mean")
).reset_index()

# Cancellation rate using all final orders before cutoff
status_counts = past_orders.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["total_orders"]
)

# Average delivery time using completed orders only
delivery = past_completed.groupby("customer_id").agg(
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# Most frequent product category from past completed orders
category_counts = (
    past_completed.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
)

most_frequent_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "most_frequent_category"})
)

# Combine all customer features
dataset = pd.DataFrame({"customer_id": cohort_ids})

dataset = dataset.merge(rfm, on="customer_id", how="left")
dataset = dataset.merge(discount, on="customer_id", how="left")
dataset = dataset.merge(status_counts, on="customer_id", how="left")
dataset = dataset.merge(delivery, on="customer_id", how="left")
dataset = dataset.merge(
    most_frequent_category,
    on="customer_id",
    how="left"
)
dataset = dataset.merge(customers, on="customer_id", how="left")

# Calculate customer tenure as of cutoff
dataset["tenure_days"] = (
    cutoff - dataset["signup_date"]
).dt.days

# Target: 1 if customer did not order in September, otherwise 0
dataset["lapsed"] = (
    ~dataset["customer_id"].isin(september_active_ids)
).astype(int)

# Remove helper columns
dataset = dataset.drop(
    columns=[
        "last_order_date",
        "total_orders",
        "cancelled_orders",
        "signup_date"
    ]
)

# Check missing values
print("\nDataset Shape:")
print(dataset.shape)

print("\nLapse Counts:")
print(dataset["lapsed"].value_counts().sort_index())

print("\nLapse Percentage:")
print(
    round(dataset["lapsed"].mean() * 100, 2),
    "%"
)

print("\nMissing Values:")
print(dataset.isnull().sum())

# Save dataset
dataset.to_csv(
    "outputs/lapse_dataset.csv",
    index=False
)

print("\nLapse dataset saved successfully!")
