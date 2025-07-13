from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
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

prompt1 = PromptTemplate(
  template="Generate a tweet about {topic}",
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template="Generate a blog post about {topic}",
  input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'tweet': prompt1 | model | parser,
  'blog': prompt2 | model | parser
})

output = parallel_chain.invoke({'topic': 'AI'})

print("Tweet: ", output['tweet'])
print("Blog: ", output['blog'])