
import pandas as pd
import json
import re
import time
import random
import mock_llm

from part_c_prompts import ZERO_SHOT_PROMPT

BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


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


# Reset immediately before the full experiment
mock_llm.reset()

raw_tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

guardrails = pd.read_csv(
    "outputs/tickets_guardrails.csv",
    keep_default_na=False
)

tickets = raw_tickets[
    ["ticket_id", "customer_id", "true_category"]
].merge(
    guardrails[
        ["ticket_id", "redacted_text", "status"]
    ],
    on="ticket_id",
    how="left"
)

results = []
running_tokens = 0

for _, ticket in tickets.iterrows():

    ticket_id = ticket["ticket_id"]
    redacted_text = ticket["redacted_text"]
    original_status = ticket["status"]

    category = ""
    tokens_used = 0

    if original_status in ["skipped_empty", "flagged_long"]:

        final_status = original_status

    elif running_tokens >= TOKEN_BUDGET:

        final_status = "budget_exhausted"

    else:

        response = call_with_retry(
            ticket_id,
            ZERO_SHOT_PROMPT,
            redacted_text
        )

        content = response.choices[0].message.content

        tokens_used = response.usage.total_tokens
        running_tokens += tokens_used

        category, final_status = validate_response(content)

    results.append({
        "ticket_id": ticket_id,
        "customer_id": ticket["customer_id"],
        "true_category": ticket["true_category"],
        "category": category,
        "status": final_status,
        "tokens_used": tokens_used
    })


result_df = pd.DataFrame(results)

result_df.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

print("\n========== TOKEN BUDGET RESULTS ==========")

print("\nStatus counts:")
print(result_df["status"].value_counts())

print("\nConfigured budget:", TOKEN_BUDGET)
print("Actual tokens used:", running_tokens)
print("Remaining budget:", TOKEN_BUDGET - running_tokens)

print(
    "\nBudget exhausted tickets:",
    (result_df["status"] == "budget_exhausted").sum()
)

print("\nTotal tickets:", len(result_df))

print("\nSaved: outputs/triage_budget_demo.csv")
