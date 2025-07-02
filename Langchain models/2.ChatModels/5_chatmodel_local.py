# [AI GENERATED] This file was written by the assistant.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="text-generation",
    huggingfacehub_api_token=None,  # Not needed for local models
    timeout=60,
    local=True
)

model = ChatHuggingFace(llm=llm)

prompt = "Explain the importance of embeddings in NLP."
result = model.invoke(prompt)
print("Local Model Response:", result)
