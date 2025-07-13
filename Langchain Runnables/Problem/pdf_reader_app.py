from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="google/gemma-7b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
)

model = ChatHuggingFace(llm=llm)

#* Load the document
loader = TextLoader("sample_text.txt")
documents = loader.load()

#* Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,
  chunk_overlap=50,
)
docs = text_splitter.split_documents(documents)

#* Convert the documents into embeddings and store in FAISS vector store
vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())

#* Create a retriever ()fetches relevant documents
retriever = vectorstore.as_retriever()

#* Manually retrieve relevant documents
query = retriever.invoke("What is the key takeaways of the document?")
retrieved_docs = retriever.get_relevant_documents(query)

#* Combine retrieve text into a single prompt 
retreived_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

#* Initialize the llm
prompt = f"Based on the following text, answer the question{query} :\n\n{retreived_text}"
answer = model.predict(prompt)

print("Answer", answer)