# [AI GENERATED] This file was written by the assistant.
from typing import TypedDict
from langchain_core.prompts import PromptTemplate

class Book(TypedDict):
    title: str
    author: str

book = Book(title="1984", author="George Orwell")

template = PromptTemplate(
    template="Book: {title}\nAuthor: {author}",
    input_variables=["title", "author"],
    validate_template=True
)

prompt = template.invoke(book)
print(prompt)
