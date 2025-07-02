# [AI GENERATED] This file was written by the assistant.
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

class Car(BaseModel):
    make: str
    model: str

car = Car(make="Tesla", model="Model S")

template = PromptTemplate(
    template="Car: {make}\nModel: {model}",
    input_variables=["make", "model"],
    validate_template=True
)

prompt = template.invoke(car.dict())
print(prompt)
