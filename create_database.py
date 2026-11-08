
import pandas as pd
import sqlite3
from pathlib import Path

# Database location
database_path = "data/instamart.db"

# Connect to SQLite database
connection = sqlite3.connect(database_path)

# List of cleaned datasets
tables = [
    "customers",
    "products",
    "orders",
    "tickets"
]

# Load each CSV into SQLite
for table in tables:
    csv_path = f"data/clean/{table}.csv"

    df = pd.read_csv(csv_path)

    df.to_sql(
        table,
        connection,
        if_exists="replace",
        index=False
    )

    print(f"{table} table created: {len(df)} rows")

# Verify tables
cursor = connection.cursor()

cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)

print("\nDatabase tables:")
print(cursor.fetchall())

connection.close()

print("\nDatabase created successfully!")
