# [AI GENERATED] This file was written by the assistant.
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="google/gemma-7b-it",
    task="text-generation",
    huggingfacehub_api_token="hf_api_key",
    timeout=60,
)

# model = ChatHuggingFace(
#     llm=llm,
#     model_kwargs={
#         "temperature": 0.5,
#     }
# )

prompt = "What is LangChain and how does it help with LLM applications?"

result = llm.invoke(prompt)
print("LLM Response:", result)

#* For using OpenAI API
# llm = OpenAI(model='gpt-3.5-turbo-instruct')

# result = llm.invoke("What is the capital of India")