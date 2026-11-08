
import pandas as pd

# Load the raw CSV files
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Keep empty ticket text as an empty string
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

# Inspect each dataset
datasets = {
    "Customers": customers,
    "Products": products,
    "Orders": orders,
    "Tickets": tickets
}

for name, df in datasets.items():
    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    audit = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percent.round(2)
    })

    print(audit)

# Specific audit figures required by the assignment

print("\n" + "=" * 45)
print("SPECIFIC AUDIT FIGURES")
print("=" * 45)

# Order audit
numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

print("\nORDERS")
print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print(
    "Pending rows:",
    orders["status"].astype(str).str.strip().str.lower().eq("pending").sum()
)
print("Distinct order IDs:", orders["order_id"].nunique())
print("Amounts that cannot be converted to numbers:", numeric_amount.isna().sum())
print("Negative delivery times:", numeric_delivery.lt(0).sum())
print("Blank/missing delivery times:", numeric_delivery.isna().sum())

# Customer audit
numeric_age = pd.to_numeric(customers["age"], errors="coerce")
gender_text = customers["gender"].fillna("").astype(str).str.strip()
city_text = customers["city"].fillna("").astype(str)

print("\nCUSTOMERS")
print("Blank ages:", numeric_age.isna().sum())
print(
    "Ages outside 10-100:",
    numeric_age.lt(10).fillna(False).sum()
    + numeric_age.gt(100).fillna(False).sum()
)
print("Distinct raw city spellings:", city_text.nunique())
print(
    "Distinct non-blank raw gender spellings:",
    customers.loc[gender_text.ne(""), "gender"].nunique()
)
print("Blank genders:", gender_text.eq("").sum())
