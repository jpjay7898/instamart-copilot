
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load lapse dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Separate features and target
X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

# Identify categorical and numerical columns
categorical_features = [
    "most_frequent_category",
    "city",
    "gender"
]

numerical_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target counts:")
print(y_train.value_counts().sort_index())

print("\nTesting target counts:")
print(y_test.value_counts().sort_index())

# Baseline: predict every customer as not lapsed
baseline_predictions = [0] * len(y_test)

print("\nBaseline Accuracy:")
print(round(accuracy_score(y_test, baseline_predictions), 3))

print("\nBaseline Lapsed F1:")
print(round(
    f1_score(y_test, baseline_predictions, zero_division=0),
    3
))

print("\nBaseline Confusion Matrix:")
print(confusion_matrix(y_test, baseline_predictions))
