# [AI GENERATED] This file was written by the assistant.
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

class User(BaseModel):
    name: str
    email: str

user = User(name="Bob", email="bob@example.com")

template = PromptTemplate(
    template="User: {name}\nEmail: {email}",
    input_variables=["name", "email"],
    validate_template=True
)

prompt = template.invoke(user.dict())
print(prompt)
