
import pandas as pd

products = pd.read_csv("data/raw/products.csv")

print("Shape:", products.shape)
print("\nColumn names:")
print(products.columns.tolist())

print("\nFirst 10 rows:")
print(products.head(10).to_string(index=False))

print("\nMissing values:")
print(products.isna().sum())
