
#* this method is slow and takes a lot of time to give output 

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader #type: ignore

loader = DirectoryLoader(
  path="books",
  glob="*.pdf",
  loader_cls=PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load()

for document in docs:
  print(document.page_content)
  print(document.metadata)
  print("--------------------------------")

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)