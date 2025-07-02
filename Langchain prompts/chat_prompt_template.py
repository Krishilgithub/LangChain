# [AI GENERATED] This file was written by the assistant.
from langchain_core.prompts import PromptTemplate

chat_template = PromptTemplate(
    template="You are a helpful assistant. Answer the following question: {question}",
    input_variables=["question"],
    validate_template=True
)

# Example usage
prompt = chat_template.invoke({"question": "What is the capital of India?"})
print(prompt)
