
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

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

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

lr_search = GridSearchCV(
    lr_pipeline,
    lr_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Training Logistic Regression...")
lr_search.fit(X_train, y_train)

print("\nLogistic Regression Best Parameters:")
print(lr_search.best_params_)

print("Logistic Regression CV F1:")
print(round(lr_search.best_score_, 3))

lr_predictions = lr_search.predict(X_test)

print("\nLogistic Regression Test Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print(classification_report(
    y_test, lr_predictions, zero_division=0
))

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_search.fit(X_train, y_train)

print("\nRandom Forest Best Parameters:")
print(rf_search.best_params_)

print("Random Forest CV F1:")
print(round(rf_search.best_score_, 3))

rf_predictions = rf_search.predict(X_test)

print("\nRandom Forest Test Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print(classification_report(
    y_test, rf_predictions, zero_division=0
))

# Compare models
print("\nFinal Model Comparison:")
print("Logistic Regression F1:",
      round(f1_score(y_test, lr_predictions), 3))

print("Random Forest F1:",
      round(f1_score(y_test, rf_predictions), 3))
