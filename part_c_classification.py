
import pandas as pd
import json
import re
import time
import random
import mock_llm

from part_c_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT


# Configuration
BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


# Retry function
def call_with_retry(ticket_id, system_prompt, redacted_text):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": redacted_text
        }
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


# Classify one model response
def validate_response(content):

    try:
        result = json.loads(content)

        category = result.get("category", "").lower().strip()

        if category in ALLOWED_CATEGORIES:
            return category, "classified_json"

        # Valid JSON but invalid category:
        # Do not use keyword fallback.
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


# Run one complete classification experiment
def run_classification(system_prompt, run_name):

    # Reset mock model before every full run
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

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        redacted_text = ticket["redacted_text"]
        status = ticket["status"]

        category = ""
        tokens_used = 0

        # Empty and long tickets never reach the model
        if status in ["skipped_empty", "flagged_long"]:

            final_status = status

        else:

            response = call_with_retry(
                ticket_id,
                system_prompt,
                redacted_text
            )

            content = response.choices[0].message.content

            tokens_used = response.usage.total_tokens

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

    # Save results
    output_path = f"outputs/triage_{run_name}.csv"

    result_df.to_csv(output_path, index=False)

    # Evaluation among classified tickets only
    classified = result_df[
        result_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    accuracy = (
        classified["category"] == classified["true_category"]
    ).mean() * 100

    total_tokens = result_df["tokens_used"].sum()

    classified_count = len(classified)

    tokens_per_ticket = (
        total_tokens / classified_count
        if classified_count > 0 else 0
    )

    print(f"\n========== {run_name.upper()} RESULTS ==========")

    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    print("\nClassified tickets:", classified_count)
    print(f"Accuracy: {accuracy:.2f}%")
    print("Total tokens:", total_tokens)
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")

    if run_name == "zero_shot":

        print("\nZero-shot confusion table:")

        print(
            pd.crosstab(
                classified["true_category"],
                classified["category"],
                rownames=["Actual"],
                colnames=["Predicted"]
            )
        )

    print("\nSaved:", output_path)

    return result_df


# Run zero-shot experiment
zero_shot_results = run_classification(
    ZERO_SHOT_PROMPT,
    "zero_shot"
)


# Run few-shot experiment
few_shot_results = run_classification(
    FEW_SHOT_PROMPT,
    "few_shot"
)


print("\nBoth classification experiments completed!")
