
import pandas as pd

results = [
    {
        "configuration": "10-word overlap",
        "document_retrieval": "12/12",
        "expected_facts": "12/12",
        "unanswerable_fallback": "2/2"
    },
    {
        "configuration": "Zero overlap",
        "document_retrieval": "12/12",
        "expected_facts": "11/12",
        "unanswerable_fallback": "2/2"
    }
]

df = pd.DataFrame(results)

df.to_csv(
    "outputs/policy_evaluation_summary.csv",
    index=False
)

print(df.to_string(index=False))
print("\nEvaluation summary saved successfully!")
