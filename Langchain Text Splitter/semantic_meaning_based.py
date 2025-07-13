from langchain_text_splitters import SemanticChunker
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
  model="meta-llama/Llama-3.1-8B-Instruct",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text = """
  Space exploration has led to incredible scientific discoveries, but it also raises important ethical questions about the future of our planet and the universe. This mission is a significant step towards understanding the mysteries of the cosmos and expanding our knowledge of the universe. Satellite Communication is a critical technology for connecting people and enabling global communication. It has revolutionized how we stay connected and access information, but also raises concerns about privacy and security. The future of space exploration and satellite communication is full of possibilities and challenges, and it is important to continue to explore the mysteries of the cosmos while also addressing the ethical questions that arise from our interactions with the universe.
"""

text_splitter = SemanticChunker(
  llm=llm,
  embeddings=embeddings,
  chunk_size=150,
  chunk_overlap=20,
  breakpoint_threshold_type="standard deviation",
  breakpoint_threshold_amount=1
)

chunks = text_splitter.split_text(text)

print(len(chunks))

print(chunks[0])
print(chunks[1])