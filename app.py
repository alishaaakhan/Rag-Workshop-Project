import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import os

st.set_page_config(page_title="Medical FAQ Bot", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
    }
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1d1d1;
    margin-bottom: 25px;
}
.custom-box {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    margin-top: 40px;
    color: #d1d1d1;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🩺 Medical FAQ Bot using RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Health Assistant using Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="custom-box">📄 Upload WHO guidelines, medical encyclopedias, FAQ documents, or health awareness PDFs and ask intelligent questions.</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-box">⚡ Gemini API + LangChain + FAISS + Streamlit</div>', unsafe_allow_html=True)

GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks
    def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = [Document(page_content=chunk) for chunk in text_chunks]

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    chain = load_qa_chain(model, chain_type="stuff")
    return chain



def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.run(
        input_documents=docs,
        question=user_question
    )

    st.markdown('## 🤖 AI Response')
    st.write(response)


st.markdown('## 📤 Upload Documents')

pdf_docs = st.file_uploader(
    "Upload Health FAQ Documents",
    accept_multiple_files=True
)
if st.button("Submit & Process"):
    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("PDF processed successfully!")


st.markdown('## 💬 Ask Questions')

user_question = st.text_input("Ask a Health-related Question")

if user_question:
    user_input(user_question)

st.markdown("---")
st.warning(
    "This project is for educational purposes only and does not replace professional medical advice."
)

st.markdown(
    '<div class="footer">Developed for Build with RAG: AI Workshop & Competition 🚀</div>',
    unsafe_allow_html=True
)
"
)
