# [AI GENERATED] This file was written by the assistant.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Example messages for a conversation
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the 2011 Cricket World Cup?"),
    AIMessage(content="India won the 2011 Cricket World Cup.")
]

for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")
