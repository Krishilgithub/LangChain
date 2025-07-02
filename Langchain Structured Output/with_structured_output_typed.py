# [AI GENERATED] This file was written by the assistant.
from typing import TypedDict
from langchain_core.prompts import PromptTemplate

class PersonInfo(TypedDict):
    name: str
    age: int
    occupation: str

template = PromptTemplate(
    template="Name: {name}\nAge: {age}\nOccupation: {occupation}",
    input_variables=["name", "age", "occupation"],
    validate_template=True
)

info = PersonInfo(name="Alice", age=30, occupation="Engineer")
prompt = template.invoke(dict(info))
print(prompt)
