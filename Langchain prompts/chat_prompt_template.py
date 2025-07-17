from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# chat_template = PromptTemplate(
#     template="You are a helpful assistant. Answer the following question: {question}",
#     input_variables=["question"],
#     validate_template=True
# )

# # Example usage
# prompt = chat_template.invoke({"question": "What is the capital of India?"})
# print(prompt)

#* Chat Prompt Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)