
import pandas as pd
from pathlib import Path

# Load the original products dataset
products = pd.read_csv("data/raw/products.csv")

# Remove extra spaces from text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype("string").str.strip()

# Convert price to a numeric value
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Check for invalid or missing values
print("Missing values:")
print(products.isna().sum())

print("\nDuplicate product IDs:", products["product_id"].duplicated().sum())
print("Invalid prices:", (products["price_inr"] <= 0).sum())

# Save the cleaned dataset
output_path = Path("data/clean")
output_path.mkdir(parents=True, exist_ok=True)

products.to_csv(output_path / "products.csv", index=False)

print("\nCleaned products shape:", products.shape)
print("Saved to data/clean/products.csv")
