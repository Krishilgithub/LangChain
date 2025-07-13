from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    timeout=60,
)

model1 = ChatHuggingFace(
    llm=llm,
    model_kwargs={
        "temperature": 0.5,
    }
)

model2 = ChatHuggingFace(
  llm=llm,
  model_kwargs={
    "temperature": 0.7,
  }
)

prompt1 = PromptTemplate(
  template = "Generate a short and simple notes from the following text \n {text}",
  input_variables=['text'],
)

prompt2 = PromptTemplate(
  template="Generate 5 short question answer from the following text \n {text}",
  input_variables=['text'],
)

prompt3 = PromptTemplate(
  template="Merge the provided notes and quiz into  a single document. \n Notes: {notes} \n Quiz: {quiz}",
  input_variables=['notes', 'quiz'],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'notes': prompt1 | model1 | parser,
  'quiz': prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Quantum computing is a rapidly evolving field that leverages the principles of quantum mechanics to perform computations. Unlike classical computers, which use bits as the smallest unit of data, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, allowing quantum computers to process vast amounts of information at once. This parallelism enables quantum computers to solve certain problems much faster than classical computers. Quantum computing has the potential to revolutionize various industries, including cryptography, drug discovery, and optimization problems. However, building practical quantum computers is still a significant challenge due to issues like qubit coherence and error rates. Researchers are actively working on developing more stable qubits and error correction techniques to make quantum computing a viable technology for real-world applications."""

result = chain.invoke({'text': text})
print("Final Result:", result)
chain.get_graph().print_ascii()