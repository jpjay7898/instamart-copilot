
import mock_llm
from part_c_prompts import FEW_SHOT_PROMPT

mock_llm.reset()

messages = [
    {
        "role": "system",
        "content": FEW_SHOT_PROMPT
    },
    {
        "role": "user",
        "content": "My order payment was deducted twice."
    }
]

response = mock_llm.chat(messages)

print("Response:")
print(response)

print("\nResponse type:")
print(type(response))

if isinstance(response, dict):
    print("\nResponse keys:")
    print(response.keys())
