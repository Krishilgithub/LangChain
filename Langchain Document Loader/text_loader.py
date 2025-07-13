from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
  template="write a summary for the following poem - \n {poem}",
  input_variables=["poem"]
)

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

# print(docs)

# print(type(docs))

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)


parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})

print(result)