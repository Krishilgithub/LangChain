from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

# Create the endpoint with the correct task
llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",  # <-- Use 'conversational'
    huggingfacehub_api_token=hf_api_key,
    temperature=0.5
)

# Pass the endpoint to ChatHuggingFace
chat_model = ChatHuggingFace(
    llm=llm
)

def word_counter(text):
    return len(text.split())

prompt1 = ChatPromptTemplate.from_template(
    "write a Joke about \n {topic}"
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1 | chat_model | parser)

parallel_chain = RunnableParallel({
  "joke": RunnablePassthrough(),
  "word_count": RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "AI"}))