
import sqlite3
import pandas as pd

connection = sqlite3.connect("data/instamart.db")

try:
    customers = pd.read_sql_query(
        "SELECT * FROM customers", connection
    )
    products = pd.read_sql_query(
        "SELECT * FROM products", connection
    )
    orders = pd.read_sql_query(
        "SELECT * FROM orders", connection
    )

    # Checkpoint 1
    assert len(customers) == 1200
    print("Checkpoint 1 PASSED: 1200 customers")

    # Checkpoint 2
    assert len(products) == 40
    print("Checkpoint 2 PASSED: 40 products")

    # Checkpoint 3
    assert len(orders) == 12309
    print("Checkpoint 3 PASSED: 12309 orders")

    # Checkpoint 4
    assert orders["order_id"].is_unique
    print("Checkpoint 4 PASSED: Unique order IDs")

    # Checkpoint 5
    status_counts = orders["status"].value_counts().to_dict()

    assert status_counts == {
        "completed": 11110,
        "cancelled": 1199
    }
    print("Checkpoint 5 PASSED: Order status counts")

    # Checkpoint 6
    assert orders["order_amount"].notna().all()
    print("Checkpoint 6 PASSED: No missing order amounts")

    # Checkpoint 7
    completed = orders[orders["status"] == "completed"]
    cancelled = orders[orders["status"] == "cancelled"]

    assert completed["delivery_minutes"].notna().all()
    assert cancelled["delivery_minutes"].isna().all()

    print("Checkpoint 7 PASSED: Delivery times")

    # Checkpoint 8
    revenue = completed["order_amount"].sum()

    assert abs(revenue - 2998668.80) < 0.01
    print("Checkpoint 8 PASSED: Completed revenue")

    # Checkpoint 9
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        connection
    )["name"].tolist()

    required_tables = ["customers", "products", "orders"]

    assert all(table in tables for table in required_tables)

    print("Checkpoint 9 PASSED: Required database tables")

    print("\nALL CHECKPOINTS PASSED")

finally:
    connection.close()
