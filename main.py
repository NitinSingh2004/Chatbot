import os
import json
import shutil
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

CUSTOM_PROMPT = """
You are a helpful AI assistant.

Rules:
- Respond naturally to greetings such as hi, hello, good morning, thank you, goodbye, etc.
- For data-related questions, use ONLY the provided context.
- Convert retrieved records into natural human-friendly language.
- Do not expose raw JSON unless explicitly requested.
- Do not make up information.
- If the answer is not present in the context, respond exactly:
  "I could not find that information in the uploaded data."

Context:
---------------------
{context_str}
---------------------

Question:
{query_str}

Answer:
"""

QA_PROMPT = PromptTemplate(CUSTOM_PROMPT)

app = FastAPI(title="CSV JSON RAG API")

UPLOAD_DIR = "uploads"
INDEX_DIR = "index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

query_engine = None


class ChatRequest(BaseModel):
    question: str


def load_documents():
    documents = []

    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)

        if filename.lower().endswith(".csv"):
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                documents.append(
                    Document(
                        text=json.dumps(
                            row.to_dict(),
                            ensure_ascii=False,
                            default=str
                        )
                    )
                )

        elif filename.lower().endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    documents.append(
                        Document(
                            text=json.dumps(
                                item,
                                ensure_ascii=False,
                                default=str
                            )
                        )
                    )

            elif isinstance(data, dict):
                documents.append(
                    Document(
                        text=json.dumps(
                            data,
                            ensure_ascii=False,
                            default=str
                        )
                    )
                )

    return documents


def build_query_engine(index):
    return index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=5,
        text_qa_template=QA_PROMPT,
        streaming=True,
    )


def load_query_engine():
    global query_engine

    if not os.path.exists(INDEX_DIR):
        return

    if len(os.listdir(INDEX_DIR)) == 0:
        return

    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_DIR
    )

    index = load_index_from_storage(
        storage_context
    )

    query_engine = build_query_engine(index)


@app.on_event("startup")
async def startup():
    load_query_engine()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):
    if not (
        file.filename.lower().endswith(".csv")
        or file.filename.lower().endswith(".json")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and JSON files are allowed"
        )

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filepath = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    return {
        "status": "success",
        "filename": file.filename
    }


@app.post("/Create_Embeddings")
async def create_embeddings():
    global query_engine

    documents = load_documents()

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No CSV or JSON data found"
        )

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    os.makedirs(INDEX_DIR, exist_ok=True)

    index = VectorStoreIndex.from_documents(
        documents
    )

    index.storage_context.persist(
        persist_dir=INDEX_DIR
    )

    query_engine = build_query_engine(index)

    return {
        "status": "success",
        "documents": len(documents)
    }


@app.post("/chat")
async def chat(data: ChatRequest):
    global query_engine

    if query_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Create embeddings first"
        )

    def generate():
        response = query_engine.query(
            data.question
        )

        if hasattr(response, "response_gen"):
            for token in response.response_gen:
                yield token
        else:
            yield str(response)

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.delete("/reset")
async def reset():
    global query_engine

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    os.makedirs(INDEX_DIR, exist_ok=True)

    query_engine = None

    return {
        "status": "success",
        "message": "Index deleted"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
