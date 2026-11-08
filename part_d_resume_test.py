
import json

from pathlib import Path
from part_d_agent import run_agent, resume_agent


# Remove an old S1 checkpoint and ledger
checkpoint = Path("state/S1.json")
ledger = Path("outputs/refund_ledger.jsonl")

if checkpoint.exists():
    checkpoint.unlink()

if ledger.exists():
    ledger.unlink()


message = (
    "My order ORD-100048 reached me very late. "
    "I would like a refund."
)

print("Starting S1 and simulating a crash after 2 tools...")

try:
    run_agent(
        user_message=message,
        scenario_id="S1",
        crash_after_tools=2
    )
except RuntimeError as error:
    print("Simulated crash:", error)


# Checkpoint verification
print("Checkpoint exists:", checkpoint.exists())

with open(checkpoint, "r", encoding="utf-8") as file:
    saved_state = json.load(file)

print(
    "Tools completed before crash:",
    [step["tool"] for step in saved_state["steps"]]
)


# Resume the interrupted run
print("\nResuming S1...")

result = resume_agent("S1")

print("Final status:", result["status"])
print("Final answer:", result["final_answer"])
print("LLM calls:", result["llm_calls"])
print("Tokens:", result["tokens_used"])

print(
    "All tools across original run and resume:",
    [step["tool"] for step in result["steps"]]
)


# Verify refund ledger
print("\nRefund ledger:")

if ledger.exists():
    with open(ledger, "r", encoding="utf-8") as file:
        for line in file:
            print(json.loads(line))
