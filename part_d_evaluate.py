
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from part_d_rag import answer

# Load evaluation dataset
eval_df = pd.read_csv("data/raw/policy_eval.csv")

# Create zero-overlap chunks
zero_chunks = []

for file_path in sorted(Path("data/policies").glob("*.txt")):
    words = file_path.read_text(encoding="utf-8").split()

    for start in range(0, len(words), 40):
        chunk_words = words[start:start + 40]

        zero_chunks.append({
            "source": file_path.name,
            "text": " ".join(chunk_words)
        })

        if start + 40 >= len(words):
            break

zero_df = pd.DataFrame(zero_chunks)

# Build zero-overlap TF-IDF index
zero_vectorizer = TfidfVectorizer()
zero_matrix = zero_vectorizer.fit_transform(zero_df["text"])


def retrieve_zero(query, k=3):
    query_vector = zero_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        zero_matrix
    ).flatten()

    indices = scores.argsort()[::-1][:k]

    return [
        {
            "source": zero_df.iloc[i]["source"],
            "score": float(scores[i]),
            "text": zero_df.iloc[i]["text"]
        }
        for i in indices
    ]


def evaluate(retriever, label):
    retrieval_pass = 0
    fact_pass = 0
    fallback_pass = 0

    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)

    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        result = answer(question, retrieve_fn=retriever)
        response = result["answer"]

        retrieved_docs = [
            item["source"] for item in result["sources"]
        ]

        if expected_doc == "NONE":
            fallback_ok = response.startswith("I could not find")
            fallback_pass += int(fallback_ok)

            print("\nQuestion:", question)
            print("Answer:", response)
            print("Fallback:", fallback_ok)

        else:
            doc_ok = expected_doc in retrieved_docs
            fact_ok = (
                pd.notna(expected_fact)
                and str(expected_fact) in response
            )

            retrieval_pass += int(doc_ok)
            fact_pass += int(fact_ok)

            print("\nQuestion:", question)
            print("Expected document:", expected_doc)
            print("Retrieved documents:", retrieved_docs)
            print("Expected fact:", expected_fact)
            print("Answer:", response)
            print("Document found:", doc_ok)
            print("Fact found:", fact_ok)

    print("\nRESULTS")
    print("Document retrieval:", retrieval_pass, "/ 12")
    print("Expected facts:", fact_pass, "/ 12")
    print("Unanswerable fallback:", fallback_pass, "/ 2")


# Evaluate required 10-word overlap version
from part_d_retrieval import retrieve

evaluate(retrieve, "10-WORD OVERLAP")

# Evaluate zero-overlap version
evaluate(retrieve_zero, "ZERO OVERLAP")
