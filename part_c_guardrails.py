
import pandas as pd
import re
from pathlib import Path
from mock_llm import est_tokens

# Load raw tickets
df = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

MAX_INPUT_TOKENS = 400

# Exact patterns required by the assignment
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

# Redaction counts
email_count = 0
phone_count = 0
order_count = 0

redacted_texts = []
statuses = []
token_counts = []

# Process tickets in their original order
for text in df["text"]:

    text = str(text)

    # Step 1: Redact email addresses
    redacted, count = re.subn(
        EMAIL_PATTERN, "[EMAIL]", text
    )
    email_count += count

    # Step 2: Redact phone numbers
    redacted, count = re.subn(
        PHONE_PATTERN, "[PHONE]", redacted
    )
    phone_count += count

    # Step 3: Redact order IDs
    redacted, count = re.subn(
        ORDER_PATTERN, "[ORDER_ID]", redacted
    )
    order_count += count

    redacted_texts.append(redacted)

    # Input guardrails
    estimated_tokens = est_tokens(text)
    token_counts.append(estimated_tokens)

    if not text.strip():
        statuses.append("skipped_empty")

    elif estimated_tokens > MAX_INPUT_TOKENS:
        statuses.append("flagged_long")

    else:
        statuses.append("ready_for_classification")


# Add results to dataframe
df["redacted_text"] = redacted_texts
df["estimated_tokens"] = token_counts
df["status"] = statuses

# Verify that no PII remains
remaining_emails = sum(
    len(re.findall(EMAIL_PATTERN, text))
    for text in redacted_texts
)

remaining_phones = sum(
    len(re.findall(PHONE_PATTERN, text))
    for text in redacted_texts
)

remaining_orders = sum(
    len(re.findall(ORDER_PATTERN, text))
    for text in redacted_texts
)

# Print guardrail results
print("\nTicket Count:", len(df))

print("\nGuardrail Status Counts:")
print(df["status"].value_counts())

print("\nLong Ticket Token Counts:")
print(
    df.loc[
        df["status"] == "flagged_long",
        "estimated_tokens"
    ].tolist()
)

# Print PII redaction results
print("\nPII Redaction Counts:")
print("Emails:", email_count)
print("Phone Numbers:", phone_count)
print("Order IDs:", order_count)
print(
    "Total Redactions:",
    email_count + phone_count + order_count
)

print("\nRemaining PII After Redaction:")
print("Emails:", remaining_emails)
print("Phones:", remaining_phones)
print("Order IDs:", remaining_orders)

# Assertions
assert len(df) == 164
assert (df["status"] == "skipped_empty").sum() == 4
assert (df["status"] == "flagged_long").sum() == 4

assert email_count == 55
assert phone_count == 91
assert order_count == 132

assert remaining_emails == 0
assert remaining_phones == 0
assert remaining_orders == 0

# Save intermediate output
Path("outputs").mkdir(exist_ok=True)

df.to_csv(
    "outputs/tickets_guardrails.csv",
    index=False
)

print("\nAll guardrail and PII checks passed!")
print("Saved: outputs/tickets_guardrails.csv")
