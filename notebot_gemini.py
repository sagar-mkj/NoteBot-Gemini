import asyncio
import nest_asyncio
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("Please set your GOOGLE_API_KEY environment variable.")
    st.stop()

st.header("NoteBot (using Gemini API)")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query, k=4)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_output_tokens=512)
        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context and
            if you did not get the context simply say "I don't know Jenny":
            {context}
            Question: {input}"""
        )
        chain = create_stuff_documents_chain(llm, customized_prompt)
        output = chain.invoke({"input": user_query, "input_documents": matching_chunks})
        st.write(output)
