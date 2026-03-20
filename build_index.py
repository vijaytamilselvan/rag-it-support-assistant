import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your dataset
with open("kb_dataset.json", "r") as f:
    data = json.load(f)

# Chunking
chunks = []

for item in data:
    text = item.get("content", "")
    
    # simple chunking (you can improve later)
    chunk_size = 200
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append({
            "content": chunk
        })

with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Total chunks created: {len(chunks)}")

# Create embeddings
texts = [chunk["content"] for chunk in chunks]
embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

#  FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")

print("FAISS index created and saved!")