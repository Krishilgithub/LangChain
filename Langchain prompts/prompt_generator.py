# [AI GENERATED] This file was written by the assistant.
from langchain_core.prompts import PromptTemplate

def generate_prompt(task: str, variable: str) -> str:
    if task == "summarize":
        template = PromptTemplate(
            template="Summarize the following text: {text}",
            input_variables=["text"],
            validate_template=True
        )
        return template.invoke({"text": variable})
    elif task == "question":
        template = PromptTemplate(
            template="Answer the following question: {question}",
            input_variables=["question"],
            validate_template=True
        )
        return template.invoke({"question": variable})
    else:
        return "Unknown task."

# Example usage
if __name__ == "__main__":
    print(generate_prompt("summarize", "LangChain is a framework for developing LLM applications."))
    print(generate_prompt("question", "What is LangChain?"))
