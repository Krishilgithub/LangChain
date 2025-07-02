from typing import TypedDict, Annotated, Literal, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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

class Person(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(..., gt=18, description="The age of the person")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person:\n{format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'Indian'})
print("Prompt:", prompt)

# result = model.invoke(prompt)
# final_result = parser.parse(str(result))
# print("Final Result:", final_result)

chain = template | model | parser
chain_result = chain.invoke({'place': 'Indian'})

print("Parsed Result:", chain_result)
print("Type: ", type(chain_result))