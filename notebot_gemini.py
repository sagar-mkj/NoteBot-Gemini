import asyncio
import nest_asyncio

# Patch asyncio so Streamlit works with async code
nest_asyncio.apply()

# Ensure event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)















import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Gemini API (LangChain integration)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Paste your Gemini API key here ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBMFewGCPk-MyBrDSpm_7N7CbxX9tUjQag"

st.header("NoteBot (using Gemini API)")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    # extract text
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)

    # create embeddings using Gemini API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # build FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # user query
    user_query = st.text_input("Type your query here")

    if user_query:
        # semantic search
        matching_chunks = vector_store.similarity_search(user_query, k=4)

        # define Gemini chat model
        # you can switch "gemini-1.5-flash" → "gemini-1.5-pro" if your key supports it
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_output_tokens=512,
        )

        # custom prompt
        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context and
            if you did not get the context simply say "I don't know Jenny":
            {context}
            Question: {input}
            """
        )

        # create chain
        chain = create_stuff_documents_chain(llm, customized_prompt)

        # run chain
        output = chain.invoke({"input": user_query, "input_documents": matching_chunks})
        st.write(output)
