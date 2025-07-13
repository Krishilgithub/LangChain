from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def get_name(self):
    return self.name


  def get_age(self):
    return self.age


student = Student("John", 20)
print(student.get_name())
print(student.get_age())

"""
#* we can also use Language.PYTHON to split python code
splitter = RecursiveCharacterTextSplitter.from_language(
  language=Language.PYTHON,
  chunk_size=100,
  chunk_overlap=20,
  # length_function=len
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks[0])
print(chunks[1])
print(chunks[2])