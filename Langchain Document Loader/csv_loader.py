from langchain_community.document_loaders import CSVLoader
import csv

loader = CSVLoader(
  file_path="Social_Network_Ads.csv",
  # csv_args={
  #   "delimiter": ",",
  #   "quotechar": '"',
  #   "quoting": csv.QUOTE_MINIMAL
  # }
)

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
print(len(docs))