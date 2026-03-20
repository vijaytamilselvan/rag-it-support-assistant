import pandas as pd
import json
import faiss
import numpy as np
from transformers import pipeline

from sentence_transformers import SentenceTransformer

#Loading models

model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

#Loading index

index=faiss.read_index("faiss_index.bin")

#Loading chunks

with open("chunks.json","r") as f:
    chunked_data=json.load(f)


#Search function

def search(query,k=3):

    query_vector = model.encode([query])
    distances,indices=index.search(query_vector,k)

    return [chunked_data[i] for i in indices[0]]


#Prompt- building

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])
    
    prompt = f"""
You are an IT support assistant.

Use the following context to answer the question clearly and accurately.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


#Genearting answer

def generate_answer(query, retrieved_chunks):
    prompt = build_prompt(query, retrieved_chunks)
    
    response = generator(prompt, max_new_tokens=100, do_sample=False)
    
    return response[0]["generated_text"]


def main():
    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        retrieved = search(query)
        answer = generate_answer(query, retrieved)

        print("\nFinal Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()


