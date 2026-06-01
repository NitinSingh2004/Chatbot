import os
import shutil
import streamlit as st
from dotenv import load_dotenv

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
# ENVIRONMENT CONFIGURATION
# =====================================================
load_dotenv()

# Prevent Hugging Face from creating symlinks or cluttering your local working directory
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Production Chatbot",
    layout="wide"
)

# =====================================================
# EMBEDDING MODEL (Zero Local Folder Storage)
# =====================================================
@st.cache_resource
def load_embedding_model():
    # This automatically uses the global system cache, completely bypassing your local folder
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = load_embedding_model()

# =====================================================
# LLM CONFIG (Cloud Execution via Groq)
# =====================================================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY environment variable. Please check your .env file or Render dashboard configuration.")

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

# =====================================================
# DATA MANAGEMENT CONFIG
# =====================================================
LOCAL_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(LOCAL_PROJECT_DIR, "uploads")
INDEX_DIR = os.path.join(LOCAL_PROJECT_DIR, "indexes")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# =====================================================
# UI HEADER
# =====================================================
st.title("📚 Cloud-Native Document Chatbot")
st.caption("LlamaIndex + Isolated Embeddings + Groq Integration")

# =====================================================
# BUSINESS WORKFLOW INITIALIZATION
# =====================================================
business_id = st.text_input("Business ID", value="demo_business")

upload_path = os.path.join(UPLOADS_DIR, business_id)
index_path = os.path.join(INDEX_DIR, business_id)

os.makedirs(upload_path, exist_ok=True)
os.makedirs(index_path, exist_ok=True)

if "current_business" not in st.session_state:
    st.session_state.current_business = business_id

if st.session_state.current_business != business_id:
    st.session_state.current_business = business_id
    if "chat_engine" in st.session_state:
        del st.session_state.chat_engine
    st.session_state.messages = []

# =====================================================
# FILE MANAGEMENT SYSTEM
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
# RAG INDEX GENERATION
# =====================================================
if st.button("Create Embeddings"):
    try:
        with st.spinner("Processing files..."):
            documents = SimpleDirectoryReader(upload_path).load_data()

        with st.spinner("Indexing vector database..."):
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=index_path)

        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            verbose=False
        )
        st.success("Vector database created successfully.")
    except Exception as e:
        st.error(str(e))

# =====================================================
# RE-LOAD INDEX PIPELINE
# =====================================================
if "chat_engine" not in st.session_state:
    try:
        if os.path.exists(index_path) and len(os.listdir(index_path)) > 0:
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                verbose=False
            )
    except Exception as e:
        st.warning(f"Could not load index: {e}")

# =====================================================
# CONVERSATIONAL INTERFACE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.divider()
st.subheader("💬 Interactive Query Hub")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "chat_engine" not in st.session_state:
        answer = "Please upload documents and process them first."
    else:
        try:
            with st.spinner("Streaming response..."):
                response = st.session_state.chat_engine.chat(prompt)
                answer = str(response)
        except Exception as e:
            answer = f"Error generating response: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# =====================================================
# DATA WIPE MECHANISM
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

        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            del st.session_state.chat_engine

        st.success("Business data wiped clean.")
        st.rerun()
    except Exception as e:
        st.error(str(e))
