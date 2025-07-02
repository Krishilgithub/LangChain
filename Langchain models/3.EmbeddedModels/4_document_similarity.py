# [AI GENERATED] This file was written by the assistant.
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", # Using a local Hugging Face model
)

documents = [
  "Virat Kohli is a famous indian cricketer known for his aggressive batting style. He has played for the Royal Challengers Bangalore in the Indian Premier League. He is also the former captain of the Indian national cricket team. He has numerous records to his name, including being one of the fastest players to score 8000 runs in One Day Internationals (ODIs). Kohli is known for his fitness regime and has been a role model for many aspiring cricketers. He has also been involved in various philanthropic activities.",
]

query = "Who is Virat Kohli?"

document_embd = embedding.embed_documents(documents)
query_embd = embedding.embed_query(query)

query_embd = np.array(query_embd).reshape(1, -1)
document_embd = np.array(document_embd)
print(cosine_similarity(query_embd, document_embd))
