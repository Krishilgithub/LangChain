from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key
)

model = ChatHuggingFace(
    llm=llm,
    model_kwargs={
        "temperature": 0.5,
    }
)

prompt1 = PromptTemplate(
    template= "write a Joke about \n {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="explain the joke in detail \n {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1 | llm | parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explain": RunnableSequence(prompt2 | llm | parser)
})

final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)
print(final_chain.invoke({"topic":"csk and rcb"}))