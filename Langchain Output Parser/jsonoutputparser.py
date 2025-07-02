from typing import TypedDict, Annotated, Literal, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    timeout=60,
)

model = ChatHuggingFace(
    llm=llm,
    model_kwargs={
        "temperature": 0.5,
    }
)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me 5 facts about {topic}:\n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions() }
)

# prompt = template1.format()    #* we can also use model.invoke({})

# result = model.invoke(prompt)

# final_result = parser.parse(str(result.content))

chain = template1 | model | parser

result = chain.invoke({'topic': 'black holes'})

print("Parsed Result:", result)
print("Type: ", type(result))