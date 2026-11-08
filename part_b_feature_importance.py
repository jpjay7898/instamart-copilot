
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Train Random Forest using the best parameters
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_leaf=20
    ))
])

rf_pipeline.fit(X_train, y_train)

# Get feature names
feature_names = rf_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get Random Forest importance
rf_importance = rf_pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 5 Random Forest Features:")
print(importance_df.head(5).to_string(index=False))


# Train Logistic Regression using best C=10
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        C=10
    ))
])

lr_pipeline.fit(X_train, y_train)

# Get Logistic Regression coefficients
lr_coefficients = lr_pipeline.named_steps[
    "model"
].coef_[0]

coefficient_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_coefficients
})

coefficient_df["absolute_value"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    by="absolute_value",
    ascending=False
)

print("\nTop 3 Logistic Regression Coefficients:")
print(
    coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

# Save feature importance
importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
