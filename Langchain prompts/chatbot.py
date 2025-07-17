from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Chat-capable model
    task="conversational",
    huggingfacehub_api_token=hf_api_key,
    timeout=60
)

from typing import List, Union

chat_history: List[Union[SystemMessage, AIMessage, HumanMessage]] = [
    SystemMessage(content="You are a helpful AI Assitant")
]

chatbot = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot.invoke(user_input)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    print("Chatbot:", response.content)


# model = ChatOpenAI()

# chat_history = [
#     SystemMessage(content='You are a helpful AI assistant')
# ]

# while True:
#     user_input = input('You: ')
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input == 'exit':
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content))
#     print("AI: ",result.content)

# print(chat_history)