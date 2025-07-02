# [AI GENERATED] This file was written by the assistant.
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

model = ChatAnthropic(
    anthropic_api_key=anthropic_api_key,
    model="claude-3-opus-20240229"
)

prompt = "What are the main differences between Claude and GPT-4?"
result = model.invoke(prompt)
print("Anthropic Model Response:", result)
