
import pandas as pd
import json

from pathlib import Path
from part_d_agent import run_agent


scenarios = pd.read_csv("data/raw/agent_scenarios.csv")

output_path = Path("outputs/agent_runs.jsonl")
output_path.parent.mkdir(exist_ok=True)

# Remove old test results
if output_path.exists():
    output_path.unlink()

# Clear the sample refund ledger
ledger_path = Path("outputs/refund_ledger.jsonl")
if ledger_path.exists():
    ledger_path.unlink()


for _, row in scenarios.iterrows():

    result = run_agent(
        user_message=row["message"],
        scenario_id=row["scenario_id"]
    )

    record = {
        "scenario_id": row["scenario_id"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"]
    }

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    print("\nScenario:", row["scenario_id"])
    print("Expected:", row["expected_outcome"])
    print("Actual:", result["status"])
    print("Answer:", result["final_answer"])
    print("Tools:", [s["tool"] for s in result["steps"]])
    print("LLM calls:", result["llm_calls"])
    print("Tokens:", result["tokens_used"])
