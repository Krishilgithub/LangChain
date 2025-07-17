from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

hf_api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    timeout=60,
    huggingfacehub_api_token=hf_api_token
)

model = ChatHuggingFace(llm=llm)

from typing import List, Union

# Example messages for a conversation
messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    query = input("Ask your question: ")
    if query.lower() in ['exit', 'quit']:
        break
    messages.append(HumanMessage(content=query))    
    result = model.invoke(messages)
    messages.append(AIMessage(result.content))
    print(result.content)

for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")
