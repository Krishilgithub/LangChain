# [AI GENERATED] This file was written by the assistant.
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

model = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192"
)

prompt = "Explain the concept of transfer learning."
result = model.invoke(prompt)
print("Groq Model Response:", result.content)
