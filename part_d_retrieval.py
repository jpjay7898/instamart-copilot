
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load policy chunks
chunks_df = pd.read_csv("outputs/policy_chunks.csv")

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks_df["text"])

def retrieve(query, k=3):
    # Convert question into TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get top k matching chunks
    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": chunks_df.iloc[index]["source"],
            "score": round(float(similarities[index]), 4),
            "text": chunks_df.iloc[index]["text"]
        })

    return results


# Test retrieval
if __name__ == "__main__":
    query = "refund policy late delivery"

    results = retrieve(query, k=3)

    print("Query:", query)
    print("\nTop 3 Retrieved Policy Chunks:\n")

    for result in results:
        print("Source:", result["source"])
        print("Score:", result["score"])
        print("Text:", result["text"])
        print("-" * 60)
