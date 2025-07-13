from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

hf_api_key = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key,
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
  template="Answer the following question \n {question} from the following text \n {text}",
  input_variables=["question", "text"]
)

parser = StrOutputParser()

#* we can also pass a list of urls
url = "https://www.amazon.in/Acer-Predator-Processor-Windows-PH16-72/dp/B0CXPXT4XD/ref=sr_1_3?adgrpid=67997053628&dib=eyJ2IjoiMSJ9.kgXW62mqIj7a-tRUqQMd5cokawsLN_BIawobPjsHfQt82iNmtPwxbwxoyI-GJDzEpAtIvgy9BcZYpTeJJfBTLna6-aaiAMDGg9oki35kLFJ7bW_KdqKgqE1qCLenEzG3ZWb7ijKIAJWD3tMs7zTh5LH9Xa1E0-vvbdYEfptFmHBUaZp81SnKpDaNWadAF6GOgRWrfwF7k-QO2VIitu3inV9jHBpYmgIkbfmDe1Ifvdc.JqtbJ1LI0nnPSgDTv9J8JHZEiHJ5dth_xbIITcl83vc&dib_tag=se&ext_vrnc=hi&hvadid=590736327772&hvdev=c&hvlocphy=9302006&hvnetw=g&hvqmt=e&hvrand=10966701693944854770&hvtargid=kwd-901732557320&hydadcr=16067_2268610&keywords=predator%2Bgaming%2Blaptop%2Bamazon&mcid=e831b051483038b6993ad5b87eafad51&nsdOptOutParam=true&qid=1751604990&sr=8-3&th=1"

loader = WebBaseLoader([url])

docs = loader.load()

# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

question = "What is the price of the laptop?"

response = chain.invoke({"question": question, "text": docs[0].page_content})

print(response)