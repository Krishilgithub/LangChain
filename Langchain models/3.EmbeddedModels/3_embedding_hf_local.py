# [AI GENERATED] This file was written by the assistant.
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2",
)

text = "Delhi is the capital of India."

vector = embedding.embed_query(text)

print("Embedding for query:", str(vector))
