# [AI GENERATED] This file was written by the assistant.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="google/gemma-7b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    timeout=60,
)

model = ChatHuggingFace(llm=llm)

prompt = "What is the difference between supervised and unsupervised learning?"
result = model.invoke(prompt)
print("HuggingFace API Model Response:", result)
