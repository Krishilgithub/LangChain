from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
# print('HF API Key:', hf_api_key)  # Debug print

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Chat-capable model
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    timeout=60,
)

model = ChatHuggingFace(llm=llm)

prompt = "What is the difference between supervised and unsupervised learning?"
result = model.invoke(prompt)
print("HuggingFace API Model Response:", result.content)