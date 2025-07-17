from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os

load_dotenv()

hf_api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

load_dotenv()

llm = HuggingFaceEndpoint(
  model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    timeout=60,
    huggingfacehub_api_token=hf_api_token
)

model = ChatHuggingFace(
  llm = llm
)

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

# --- Pydantic structured output (NOT SUPPORTED for HuggingFace, will raise NotImplementedError) ---
# structured_model = model.with_structured_output(Review)
# result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
# The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
# However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
# Pros:
# Insanely powerful processor (great for gaming and productivity)
# Stunning 200MP camera with incredible zoom capabilities
# Long battery life with fast charging
# S-Pen support is unique and useful
# Review by Nitish Singh
# """)
# print(result)

# --- StrOutputParser + manual JSON parsing fallback ---
from langchain_core.output_parsers import StrOutputParser
import json, re

parser = StrOutputParser()

prompt = """Return a JSON object with the following fields: key_themes (list), summary (string), sentiment (pos/neg), pros (list), cons (list), name (string or null). Review: I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver. The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality. However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow. Pros: Insanely powerful processor (great for gaming and productivity) Stunning 200MP camera with incredible zoom capabilities Long battery life with fast charging S-Pen support is unique and useful Review by Nitish Singh"""

output_content = model.invoke(prompt).content
if isinstance(output_content, str):
    parsed_input = output_content
else:
    parsed_input = str(output_content)
raw_output = parser.parse(parsed_input)
print("Raw output:\n", raw_output)
try:
    # Try to extract JSON from the output
    if isinstance(raw_output, str):
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            json_str = match.group()
            parsed = json.loads(json_str)
            print("Parsed JSON:", parsed)
        else:
            print("No JSON object found in the output.")
    else:
        print("Output is not a string, cannot parse as JSON.")
except Exception as e:
    print("Could not parse output as JSON:", e)