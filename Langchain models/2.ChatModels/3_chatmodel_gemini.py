# [AI GENERATED] This file was written by the assistant.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')

model = ChatGoogleGenerativeAI(
    google_api_key=gemini_api_key,
    model="gemini-pro"
)

prompt = "Describe the applications of generative AI in education."
result = model.invoke(prompt)
print("Gemini Model Response:", result.content)
