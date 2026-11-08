
import os
import csv
from pathlib import Path

POLICY_FOLDER = Path("data/policies")
OUTPUT_FILE = Path("outputs/policy_chunks.csv")

CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP

chunks = []

for file_path in sorted(POLICY_FOLDER.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    words = text.split()

    start = 0

    while start < len(words):
        chunk_words = words[start:start + CHUNK_SIZE]

        chunks.append({
            "source": file_path.name,
            "text": " ".join(chunk_words)
        })

        if start + CHUNK_SIZE >= len(words):
            break

        start += STEP_SIZE

os.makedirs("outputs", exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["source", "text"])
    writer.writeheader()
    writer.writerows(chunks)

print("Policy chunking completed!")
print("\nChunks per policy file:")

for file_path in sorted(POLICY_FOLDER.glob("*.txt")):
    count = sum(1 for chunk in chunks if chunk["source"] == file_path.name)
    print(f"{file_path.name}: {count}")

print(f"\nTotal chunks: {len(chunks)}")
print(f"Saved to: {OUTPUT_FILE}")
