
import time
import random
import mock_llm

BASE_DELAY = 0.05
MAX_RETRIES = 3


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
            response = mock_llm.chat(messages)
            return response

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


print("Corrected retry function created successfully!")
