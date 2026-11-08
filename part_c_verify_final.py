
import pandas as pd

df = pd.read_csv("outputs/triage_final.csv")

expected_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

assert list(df.columns) == expected_columns
assert len(df) == 164
assert df["ticket_id"].is_unique

# Fraud must always go to security
fraud = df[df["category"] == "fraud"]
assert (fraud["route"] == "security_team").all()
assert (fraud["priority"] == "HIGH").all()

# Human review statuses
human_statuses = [
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
]

human = df[df["status"].isin(human_statuses)]
assert (human["route"] == "human_review").all()
assert (human["priority"] == "NORMAL").all()

# Premium customers with valid non-fraud categories
premium = df[
    (df["segment_name"] == "Premium Customers") &
    (df["status"].isin(["classified_json", "classified_keyword"])) &
    (df["category"] != "fraud")
]

assert (premium["route"] == "premium_desk").all()
assert (premium["priority"] == "HIGH").all()

# Remaining valid tickets
support = df[df["route"] == "support_queue"]
assert (support["priority"] == "NORMAL").all()
assert (support["category"] != "fraud").all()

print("Final CSV verification passed!")
print("Total tickets:", len(df))
print("All required columns present.")
print("Fraud routing verified.")
print("Premium routing verified.")
print("Human review routing verified.")
print("Support queue routing verified.")
