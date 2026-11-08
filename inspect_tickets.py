
import pandas as pd

tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

print("Shape:", tickets.shape)

print("\nColumn names:")
print(tickets.columns.tolist())

print("\nFirst 10 rows:")
print(tickets.head(10).to_string(index=False))

print("\nData types:")
print(tickets.dtypes)

print("\nMissing or blank values:")
print(
    tickets.apply(
        lambda column: (
            column.astype(str).str.strip() == ""
        ).sum()
    )
)

print("\nTicket categories:")
print(tickets["true_category"].value_counts(dropna=False))
