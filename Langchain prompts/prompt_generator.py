# [AI GENERATED] This file was written by the assistant.
from langchain_core.prompts import PromptTemplate

def generate_prompt(task: str, variable: str) -> str:
    if task == "summarize":
        template = PromptTemplate(
            template="Summarize the following text: {text}",
            input_variables=["text"],
            validate_template=True
        )
        template.save('template.json')
        return str(template.invoke({"text": variable}))
    elif task == "question":
        template = PromptTemplate(
            template="Answer the following question: {question}",
            input_variables=["question"],
            validate_template=True
        )
        template.save('template.json')
        return str(template.invoke({"question": variable}))
    else:
        return "Unknown task."

# Example usage
if __name__ == "__main__":
    print(generate_prompt("summarize", "LangChain is a framework for developing LLM applications."))
    print(generate_prompt("question", "What is LangChain?"))

# template
# template = PromptTemplate(
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}  
# Explanation Length: {length_input}  
# 1. Mathematical Details:  
#    - Include relevant mathematical equations if present in the paper.  
#    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
# 2. Analogies:  
#    - Use relatable analogies to simplify complex ideas.  
# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=['paper_input', 'style_input','length_input'],
# validate_template=True
# )

# template.save('template.json')