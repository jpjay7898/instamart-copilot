
from pathlib import Path
from part_d_retrieval import retrieve
from mock_llm import chat


def answer(question, retrieve_fn=retrieve):
    # Retrieve the top 3 chunks
    results = retrieve_fn(question, k=3)

    # Sort chunks into their original order in the policy
    policy_texts = {}

    for result in results:
        source = result["source"]
        original_text = Path("data/policies", source).read_text(
            encoding="utf-8"
        )

        position = original_text.find(result["text"])

        policy_texts.setdefault(source, []).append(
            (position, result["text"])
        )

    context = "CONTEXT:\n"

    for source, chunks in policy_texts.items():
        chunks.sort(key=lambda item: item[0])

        previous_words = []

        for _, chunk_text in chunks:
            words = chunk_text.split()

            # Remove repeated overlap with the previous chunk
            overlap = 0
            for size in range(min(10, len(previous_words), len(words)), 0, -1):
                if previous_words[-size:] == words[:size]:
                    overlap = size
                    break

            clean_words = words[overlap:]

            context += f"[{source}]\n"
            context += " ".join(clean_words) + "\n"

            previous_words.extend(clean_words)

    context += f"\nQUESTION:\n{question}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. "
                "Answer using only the supplied policy context. "
                "If the answer is not available, say you could not find it."
            )
        },
        {
            "role": "user",
            "content": context
        }
    ]

    response = chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": results
    }


if __name__ == "__main__":
    question = "What happens if a completed order is delivered in more than 45 minutes?"
    
    result = answer(question)

    print("Question:", question)
    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source["source"], "| Score:", source["score"])
