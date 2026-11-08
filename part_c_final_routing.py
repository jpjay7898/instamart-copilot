
import pandas as pd
import json
import re
import time
import random
import mock_llm

from part_c_prompts import FEW_SHOT_PROMPT


# Configuration
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 100000

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


# Retry function
def call_with_retry(ticket_id, system_prompt, redacted_text):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]

    for attempt in range(MAX_RETRIES + 1):

        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:

            if attempt == MAX_RETRIES:
                print(f"{ticket_id}: All retries exhausted!")
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"{ticket_id} | Retry {attempt + 1} "
                f"| Waiting {wait_time:.4f} seconds"
            )

            time.sleep(wait_time)


# Validate model response
def validate_response(content):

    try:
        result = json.loads(content)

        category = result.get("category", "").lower().strip()

        if category in ALLOWED_CATEGORIES:
            return category, "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError):

        words = re.findall(r"[a-z]+", content.lower())

        matches = [
            word for word in words
            if word in ALLOWED_CATEGORIES
        ]

        if len(matches) == 1:
            return matches[0], "classified_keyword"

        return "", "flagged_invalid_output"


# Reset before the full run
mock_llm.reset()


# Read input files
raw_tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

guardrails = pd.read_csv(
    "outputs/tickets_guardrails.csv",
    keep_default_na=False
)

segments = pd.read_csv(
    "outputs/customer_segments.csv"
)


# Merge ticket data with redacted text
tickets = raw_tickets[
    ["ticket_id", "customer_id"]
].merge(
    guardrails[
        ["ticket_id", "redacted_text", "status"]
    ],
    on="ticket_id",
    how="left"
)


# Keep customer segment information
tickets = tickets.merge(
    segments[
        ["customer_id", "segment_name", "is_premium"]
    ],
    on="customer_id",
    how="left"
)


results = []
running_tokens = 0


# Process all tickets
for _, ticket in tickets.iterrows():

    ticket_id = ticket["ticket_id"]
    customer_id = ticket["customer_id"]

    redacted_text = ticket["redacted_text"]
    original_status = ticket["status"]

    segment_name = ticket["segment_name"]

    if pd.isna(segment_name):
        segment_name = ""

    is_premium = (
        False if pd.isna(ticket["is_premium"])
        else bool(ticket["is_premium"])
    )

    category = ""
    tokens_used = 0

    # Guardrail tickets
    if original_status in ["skipped_empty", "flagged_long"]:

        final_status = original_status

    # Budget check
    elif running_tokens >= TOKEN_BUDGET:

        final_status = "budget_exhausted"

    else:

        response = call_with_retry(
            ticket_id,
            FEW_SHOT_PROMPT,
            redacted_text
        )

        content = response.choices[0].message.content

        tokens_used = response.usage.total_tokens
        running_tokens += tokens_used

        category, final_status = validate_response(content)


    # Determine routing
    if final_status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:

        route = "human_review"
        priority = "NORMAL"

    elif category == "fraud":

        route = "security_team"
        priority = "HIGH"

    elif is_premium:

        route = "premium_desk"
        priority = "HIGH"

    else:

        route = "support_queue"
        priority = "NORMAL"


    results.append({
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "status": final_status,
        "category": category,
        "segment_name": segment_name,
        "priority": priority,
        "route": route,
        "tokens_used": tokens_used
    })


# Create final DataFrame
final_df = pd.DataFrame(results)


# Save final output
final_df.to_csv(
    "outputs/triage_final.csv",
    index=False
)


# Display summary
print("\n========== FINAL ROUTING RESULTS ==========")

print("\nStatus counts:")
print(final_df["status"].value_counts())

print("\nRouting counts:")
print(final_df["route"].value_counts())

print("\nPriority counts:")
print(final_df["priority"].value_counts())

print("\nTotal tickets:", len(final_df))

print("\nTotal tokens used:", running_tokens)

print("\nConfigured token budget:", TOKEN_BUDGET)

print("\nSample final records:")
print(final_df.head(10).to_string(index=False))

print("\nSaved: outputs/triage_final.csv")
