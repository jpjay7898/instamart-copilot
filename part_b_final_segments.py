
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load RFM data
rfm = pd.read_csv("outputs/customer_rfm.csv")

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardize RFM values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train final K-means model
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# Assign business-friendly segment names
segment_names = {
    0: "Regular Customers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At-Risk Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Premium flag
rfm["is_premium"] = (
    rfm["segment_name"] == "Premium Customers"
)

# Display segment profiles
profiles = rfm.groupby(
    ["segment_id", "segment_name"]
).agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\nFinal Customer Segment Profiles:")
print(profiles)

# PCA analysis
pca = PCA(n_components=2)
pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratios:")
print(pca.explained_variance_ratio_.round(3))

print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

# Save final output
rfm = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

rfm.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nFinal customer segments saved successfully!")
print("Output shape:", rfm.shape)
print("\nPremium customer count:", rfm["is_premium"].sum())
