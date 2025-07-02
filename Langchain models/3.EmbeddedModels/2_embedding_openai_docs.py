# [AI GENERATED] This file was written by the assistant.
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
]
vectors = embedding.embed_documents(documents)
print("Embeddings for documents:", vectors)
