# [AI GENERATED] This file was written by the assistant.
import json
from langchain_core.prompts import PromptTemplate

# Load JSON schema (for demonstration, hardcoded here)
schema = {
    "name": "John Doe",
    "age": 28,
    "occupation": "Developer"
}

template = PromptTemplate(
    template="Name: {name}\nAge: {age}\nOccupation: {occupation}",
    input_variables=["name", "age", "occupation"],
    validate_template=True
)

prompt = template.invoke(schema)
print(prompt)
print("As JSON:")
print(json.dumps(schema, indent=2))
