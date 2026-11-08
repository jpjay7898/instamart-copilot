
import sqlite3
import pandas as pd

# Connect to the database
connection = sqlite3.connect("data/instamart.db")

try:
    # Read the SQL file
    with open("sql/queries.sql", "r") as file:
        sql_content = file.read()

    # Separate the seven SQL queries
    queries = [
        query.strip()
        for query in sql_content.split(";")
        if query.strip()
    ]

    # Check that there are exactly seven queries
    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, but found {len(queries)}"
        )

    # Execute each query
    for index, query in enumerate(queries, start=1):

        print(f"\n========== Q{index} ==========")

        result = pd.read_sql_query(query, connection)

        if index == 7:
            print("\nFirst 5 rows:")
            print(result.head(5).to_string(index=False))

            print("\nLast 5 rows:")
            print(result.tail(5).to_string(index=False))

        else:
            print(result.head(10).to_string(index=False))

        print("\nNumber of rows:", len(result))

        # Extra Q6 validation
        if index == 6:
            different_ranks = (
                result["rank_position"]
                != result["row_number_position"]
            ).sum()

            print(
                "RANK != ROW_NUMBER:",
                different_ranks
            )

            if different_ranks == 8:
                print("Q6 reference check PASSED")
            else:
                print("Q6 reference check needs investigation")

finally:
    connection.close()
