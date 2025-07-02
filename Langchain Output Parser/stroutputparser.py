
#* 

from typing import TypedDict, Annotated, Literal, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

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

#* 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Generate a detailed report on the topic: {topic}",
    input_variables=["topic"],
    # partial_variables={
    #     "format_instruction": "The report should be comprehensive and well-structured."
    # }
)

#* 2nd prompt -> concise summary
template2 = PromptTemplate(
  template="Generate a 5 line summary on the text: {text}",
  input_variables=["text"],
)

prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})
result2 = model.invoke(prompt2)

print("Detailed Report:\n", result.content)
print("\nConcise Summary:\n", result2.content)