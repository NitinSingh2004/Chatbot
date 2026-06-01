import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================
load_dotenv()

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Document Chatbot",
    layout="wide"
)

# =====================================================
# EMBEDDING MODEL (LOAD ONLY ONCE)
# =====================================================


@st.cache_resource
def load_embedding_model():
    # Download/load model only once
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


Settings.embed_model = load_embedding_model()

# =====================================================
# LLM CONFIG
# =====================================================
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# STORAGE CONFIG
# =====================================================
# Render Persistent Disk
BASE_DIR = "/var/data"

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# =====================================================
# UI
# =====================================================
st.title("📚 Document Chatbot")
st.caption("LlamaIndex + HuggingFace Embeddings + Groq")

# =====================================================
# BUSINESS ID & INITIALIZATION
# =====================================================
business_id = st.text_input("Business ID", value="demo_business")

upload_path = os.path.join(UPLOADS_DIR, business_id)
index_path = os.path.join(INDEX_DIR, business_id)

os.makedirs(upload_path, exist_ok=True)
os.makedirs(index_path, exist_ok=True)

# Clear chat engine memory if the business ID changes
if "current_business" not in st.session_state:
    st.session_state.current_business = business_id

if st.session_state.current_business != business_id:
    st.session_state.current_business = business_id
    if "chat_engine" in st.session_state:
        del st.session_state.chat_engine
    st.session_state.messages = []

# =====================================================
# FILE UPLOAD
# =====================================================
st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, DOCX Files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

# =====================================================
# CREATE EMBEDDINGS
# =====================================================
if st.button("Create Embeddings"):
    try:
        with st.spinner("Reading documents..."):
            documents = SimpleDirectoryReader(upload_path).load_data()

        with st.spinner("Creating vector index..."):
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=index_path)

        # Instantiate and cache the chat engine immediately after creation
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            verbose=False
        )
        st.success("Embeddings created successfully.")
    except Exception as e:
        st.error(str(e))

# =====================================================
# LAZY LOAD INDEX INTO SESSION STATE
# =====================================================
if "chat_engine" not in st.session_state:
    try:
        if os.path.exists(index_path) and len(os.listdir(index_path)) > 0:
            storage_context = StorageContext.from_defaults(
                persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                verbose=False
            )
    except Exception as e:
        st.warning(f"Could not load index: {e}")

# =====================================================
# CHAT HISTORY
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.divider()
st.subheader("💬 Chat With Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================
# CHAT INPUT
# =====================================================
prompt = st.chat_input("Ask something about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the chat engine is available in session state
    if "chat_engine" not in st.session_state:
        answer = "Please upload documents and create embeddings first."
    else:
        try:
            with st.spinner("Thinking..."):
                # Call chat on the persistent state instance to preserve history
                response = st.session_state.chat_engine.chat(prompt)
                answer = str(response)
        except Exception as e:
            answer = f"Error generating response: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# =====================================================
# DELETE DATA
# =====================================================
st.divider()

if st.button("Delete Business Data"):
    try:
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        os.makedirs(upload_path, exist_ok=True)
        os.makedirs(index_path, exist_ok=True)

        # Wipe state tracking for this business
        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            del st.session_state.chat_engine

        st.success("Business data deleted successfully.")
        st.rerun()  # Force page refresh to update UI state immediately
    except Exception as e:
        st.error(str(e))
