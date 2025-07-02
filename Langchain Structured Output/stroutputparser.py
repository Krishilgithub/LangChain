from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
  model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task="text-generation",
)

model = ChatHuggingFace(
  llm = llm,
)

#* 1st prompt -> detailed report
template1 = PromptTemplate(
  template="write a detailed report on the following topic: {topic}",
  input_variables=["topic"],
  # output_parser=None,  # No structured output parser for this prompt
)

#* 2nd prompt -> concise report
template2 = PromptTemplate(
  template="write a concise report on the following text: {text}",
  input_variables=["text"],
  # output_parser=None,  # No structured output parser for this prompt
)

prompt1 = template1.format(topic="Artificial Intelligence in Healthcare") #* can also use .invoke() method

result = model.invoke(prompt1)
print("Detailed Report:", result.content)

prompt2 = template2.format(text=result.content)  # Use the content from the first prompt
result2 = model.invoke(prompt2)