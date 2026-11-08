
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load RFM data
rfm = pd.read_csv("outputs/customer_rfm.csv")

# Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different K values
results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": round(inertia, 1),
        "Silhouette": round(silhouette, 3)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nK-Means Evaluation Results:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "outputs/kmeans_evaluation.csv",
    index=False
)

print("\nResults saved to outputs/kmeans_evaluation.csv")
