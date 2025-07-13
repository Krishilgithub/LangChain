from langchain.text_splitter import RecursiveCharacterTextSplitter

text = 'Space exploration has led to incredible scientific discoveries, but it also raises important ethical questions about the future of our planet and the universe. This mission is a significant step towards understanding the mysteries of the cosmos and expanding our knowledge of the universe. Satellite Communication is a critical technology for connecting people and enabling global communication. It has revolutionized how we stay connected and access information, but also raises concerns about privacy and security. The future of space exploration and satellite communication is full of possibilities and challenges, and it is important to continue to explore the mysteries of the cosmos while also addressing the ethical questions that arise from our interactions with the universe.'

splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  chunk_overlap=50,
  length_function=len
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks[0])