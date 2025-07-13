from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="google/gemma-7b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    timeout=60,
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
  template="Suggest a catchy blog title about the {topic}",
  input_variables=["topic"],
)

topic = input("Enter a topic for the blog title: ")
# Example: topic = "Artificial Intelligence"

formatted_prompt = prompt.format(topic= topic)

blog_title = model.predict(formatted_prompt)

print("Generated Blog Title:", blog_title)