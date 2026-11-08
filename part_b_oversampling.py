
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

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

# Combine training features and target
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate the two classes
not_lapsed = train_data[train_data["lapsed"] == 0]
lapsed = train_data[train_data["lapsed"] == 1]

# Oversample the minority class
lapsed_oversampled = lapsed.sample(
    n=len(not_lapsed),
    replace=True,
    random_state=42
)

# Combine balanced training data
balanced_train = pd.concat(
    [not_lapsed, lapsed_oversampled]
).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("Original Training Counts:")
print(y_train.value_counts())

print("\nBalanced Training Counts:")
print(y_train_balanced.value_counts())

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Train best Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        C=10
    ))
])

print("\nTraining Logistic Regression with Oversampled Data...")

model.fit(X_train_balanced, y_train_balanced)

# Predict on original, unchanged test data
predictions = model.predict(X_test)

print("\nOversampled Model Test Results:")
print("Precision:", round(
    precision_score(y_test, predictions), 3
))

print("Recall:", round(
    recall_score(y_test, predictions), 3
))

print("F1 Score:", round(
    f1_score(y_test, predictions), 3
))
