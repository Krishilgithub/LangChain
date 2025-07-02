from typing import TypedDict, Annotated, Literal, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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

schema = [
  ResponseSchema(
    name="fact_1",
    description="The first fact about the topic",
  ),
  ResponseSchema(
    name="fact_2",
    description="The second fact about the topic",
  ),
  ResponseSchema(
    name="fact_3",
    description="The third fact about the topic",
  ),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
  template="Give me 3 facts about {topic}:\n{format_instructions}",
  input_variables=['topic'],
  partial_variables={'format_instructions': parser.get_format_instructions() }
)

chain = template | model | parser

result = chain.invoke({'topic': 'black holes'})

print("Parsed Result:", result)
print("Type: ", type(result))