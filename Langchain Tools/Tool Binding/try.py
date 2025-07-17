import os
from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

# Define a custom ChatOpenRouter class for OpenRouter integration
class ChatOpenRouter(ChatOpenAI):
    OPENAI_API_KEY: str = Field(alias="api_key")

    def __init__(self, OPENAI_API_KEY: str = None, **kwargs):
        # Set the API key from environment variable or provided argument
        OPENAI_API_KEY = OPENAI_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass it directly.")
        
        super().__init__(
            base_url="https://openrouter.ai/api/v1",  # OpenRouter's API base URL
            OPENAI_API_KEY=OPENAI_API_KEY,
            **kwargs,
        )

# Define your tool (function)
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# Instantiate the OpenRouter-powered LangChain model
# Replace "anthropic/claude-3.7-sonnet:thinking" with your desired OpenRouter model
llm = ChatOpenRouter(model_name="anthropic/claude-3.7-sonnet:thinking")  #

# Bind the tool to the model
llm_with_tools = llm.bind_tools([multiply])  #

# Invoke the model with a relevant query
result = llm_with_tools.invoke(HumanMessage(content="What is 5 multiplied by 42?"))  #

# Access the tool calls from the model's response
if result.tool_calls:
    for tool_call in result.tool_calls:
        if tool_call.name == "multiply":
            args = tool_call.args
            output = multiply(a=args["a"], b=args["b"])
            print(f"Tool Call: {tool_call.name} with arguments {args}")
            print(f"Tool Output: {output}")
        else:
            print(f"Model called an unknown tool: {tool_call.name}")
else:
    print(f"Model response: {result.content}")
