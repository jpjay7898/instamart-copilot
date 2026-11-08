
import sqlite3
import pandas as pd

# Connect to database
connection = sqlite3.connect("data/instamart.db")

# Load tables
customers = pd.read_sql_query(
    "SELECT * FROM customers", connection
)

orders = pd.read_sql_query(
    "SELECT * FROM orders", connection
)

products = pd.read_sql_query(
    "SELECT * FROM products", connection
)

connection.close()

# Keep only completed orders
completed = orders[orders["status"] == "completed"].copy()

# Convert columns to numeric
completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

print("\n========== EDA 1: PIVOT TABLE ==========")

merged = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).merge(
    products[["product_id", "category"]],
    on="product_id"
)

pivot = pd.pivot_table(
    merged,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print(pivot.round(2))

print("\n========== EDA 2: CITY ANALYSIS ==========")

city_analysis = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_id", "count")
)

print(city_analysis.round(2))

print("\n========== EDA 3: ORDER STATISTICS ==========")

amount = completed["order_amount"]

print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Skewness:", round(amount.skew(), 2))

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))

print("Outliers below lower fence:",
      (amount < lower_fence).sum())

print("Outliers above upper fence:",
      (amount > upper_fence).sum())

print("\n========== EDA 4: CORRELATION ==========")

correlation_data = completed[
    ["order_amount", "discount_pct",
     "quantity", "delivery_minutes"]
].copy()

correlation = correlation_data.corr()

print(correlation.round(3))

print("\n========== EDA COMPLETED ==========")
