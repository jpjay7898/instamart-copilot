
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load lapse dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Keep customer IDs separately
customer_ids = df["customer_id"]

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

categorical_features = [
    "most_frequent_category", "city", "gender"
]

numerical_features = [
    "recency", "frequency", "monetary",
    "avg_discount_pct", "cancellation_rate",
    "avg_delivery_minutes", "age", "tenure_days"
]

# Split data
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, customer_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Oversample training data only
train_data = X_train.copy()
train_data["lapsed"] = y_train

not_lapsed = train_data[train_data["lapsed"] == 0]
lapsed = train_data[train_data["lapsed"] == 1]

lapsed_oversampled = lapsed.sample(
    n=len(not_lapsed),
    replace=True,
    random_state=42
)

balanced_train = pd.concat([
    not_lapsed,
    lapsed_oversampled
]).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("Balanced Training Counts:")
print(y_train_balanced.value_counts())

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Final Logistic Regression model
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        C=10
    ))
])

# Train model
print("\nTraining final model...")

final_model.fit(X_train_balanced, y_train_balanced)

# Save model
joblib.dump(final_model, "outputs/lapse_model.joblib")

print("\nModel saved successfully!")

# Reload model
loaded_model = joblib.load("outputs/lapse_model.joblib")

# Verify predictions
original_predictions = final_model.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

print(
    "\nPredictions identical:",
    (original_predictions == loaded_predictions).all()
)

# Calculate lapse probabilities
lapse_probabilities = loaded_model.predict_proba(X_test)[:, 1]

# Create test prediction dataframe
retention_df = pd.DataFrame({
    "customer_id": id_test.values,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.values
})

# Load customer segments
segments = pd.read_csv("outputs/customer_segments.csv")

# Keep Premium customers only
premium_customers = segments[
    segments["segment_name"] == "Premium Customers"
][["customer_id"]]

# Select Premium customers from test predictions
retention_df = retention_df.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)

# Sort by highest lapse probability
retention_df = retention_df.sort_values(
    by="lapse_probability",
    ascending=False
)

# Select top 10
retention_list = retention_df.head(10)

# Save retention list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nTop 10 Premium Customers for Retention:")
print(retention_list.to_string(index=False))

print("\nRetention list shape:", retention_list.shape)

print("\nStep 10 completed successfully!")
