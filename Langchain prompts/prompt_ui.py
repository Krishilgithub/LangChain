# [AI GENERATED] This file was written by the assistant.
import streamlit as st
from langchain_core.prompts import PromptTemplate

st.title("Prompt Generator UI")

prompt_type = st.selectbox("Choose prompt type:", ["Summarize", "Question"])
user_input = st.text_area("Enter your text or question:")

if prompt_type == "Summarize":
    template = PromptTemplate(
        template="Summarize the following text: {text}",
        input_variables=["text"],
        validate_template=True
    )
    prompt = template.invoke({"text": user_input})
elif prompt_type == "Question":
    template = PromptTemplate(
        template="Answer the following question: {question}",
        input_variables=["question"],
        validate_template=True
    )
    prompt = template.invoke({"question": user_input})
else:
    prompt = ""

if st.button("Generate Prompt"):
    st.code(prompt)
