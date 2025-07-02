# [AI GENERATED] This file was written by the assistant.
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate

# Example of using MessagesPlaceholder in a prompt template
prompt = PromptTemplate(
    template="{history}\nHuman: {input}\nAI:",
    input_variables=["history", "input"],
    partial_variables={"history": MessagesPlaceholder(variable_name="history")},
    validate_template=True
)

# Example usage
filled_prompt = prompt.invoke({"input": "What is the weather today?", "history": "Human: Hi\nAI: Hello!"})
print(filled_prompt)
