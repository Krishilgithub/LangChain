from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key
)

model = ChatHuggingFace(
    llm=llm,
    model_kwargs={
        "temperature": 0.5,
    }
)

prompt = PromptTemplate(
  template="Suggest a catchy blog title about {topic}",
  input_variables=['topic']
)

# chain = LLMChain(llm = llm, prompt=prompt)

topic = input('Enter a topic')
# output = chain.run(topic)

# print("Generated blog title: ", output)