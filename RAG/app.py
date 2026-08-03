import json
import shutil
import uuid
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage, BaseMessage
from langchain_ollama import  ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import ingest_pdf

load_dotenv()

persistant_directory ="dbv2/chroma_db"

#Load embeddings and vectory store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory=persistant_directory
db=Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

llm = ChatOllama(model="llava", temperature=0)
sessions: dict[str, list[BaseMessage]] = {}


def rewrite_standalone_question(user_question: str, chat_history: list) -> str:
    if not chat_history:
        return user_question
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]
    result = llm.invoke(messages)
    return result.content.strip()

def build_evidence_and_message(scored_chunks, query):
    """scored_chunks: list of (Document, score) tuples from similarity_search_with_score.
    Builds the multimodal HumanMessage AND a UI-friendly evidence list in one pass."""
    prompt_text = f"Based on the following documents, answer this question: {query}\n\nCONTENT TO ANALYZE:\n"
    message_images: list[str] = []
    evidence: list[dict[str, Any]] = []

    for i, (chunk, distance) in enumerate(scored_chunks):
        # Chroma returns a distance (lower = more similar); convert to a 0-1 "relevance" score for the UI
        relevance = max(0.0, 1 - distance)
        prompt_text += f"--- Document {i+1} ---\n"

        if "original_content" in chunk.metadata:
            original_data = json.loads(chunk.metadata["original_content"])

            raw_text = original_data.get("raw_text", "")
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"
                evidence.append({
                    "type": "text",
                    "label": f"Document {i+1} — text",
                    "content": raw_text[:400],
                    "score": round(relevance, 2)
                })

            for j, table in enumerate(original_data.get("tables_html", [])):
                prompt_text += f"Table {j+1}:\n{table}\n\n"
                evidence.append({
                    "type": "table",
                    "label": f"Document {i+1} — table {j+1}",
                    "content": table,
                    "score": round(relevance, 2)
                })

            for img_b64 in original_data.get("images_base64", []):
                message_images.append(img_b64)
                evidence.append({
                    "type": "image",
                    "label": f"Document {i+1} — diagram",
                    "content": f"data:image/jpeg;base64,{img_b64}",
                    "score": round(relevance, 2)
                })
        else:
            prompt_text += chunk.page_content + "\n\n"
            evidence.append({
                "type": "text",
                "label": f"Document {i+1}",
                "content": chunk.page_content[:400],
                "score": round(relevance, 2)
            })

    prompt_text += "\nProvide a clear answer using the text, tables, and images above, and consider our earlier conversation. If you can't find the answer, say so.\n\nANSWER:"

    message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for img in message_images:
        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    return HumanMessage(content=message_content), evidence



def ask_question(user_question: str, chat_history: list):
    search_question = rewrite_standalone_question(user_question, chat_history)

    scored_chunks = db.similarity_search_with_score(search_question, k=3)
    human_message, evidence = build_evidence_and_message(scored_chunks, user_question)

    messages = [
        SystemMessage(content="You are a helpful assistant answering questions using documents, tables, images, and conversation history.")
    ] + chat_history + [human_message]

    result = llm.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer, evidence




#--------------FASTAPI---------------------------#

app = FastAPI()

class AskRequest(BaseModel):
    question:str
    session_id:str | None=None

class AskResponse(BaseModel):
    answer : str
    evidence : list[dict[str, Any]]
    session_id : str

@app.post("/ask", response_model=AskResponse)
def ask (req:AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    chat_history = sessions.setdefault(session_id, [])

    answer,evidence = ask_question(req.question, chat_history)

    return AskResponse(answer=answer, evidence = evidence, session_id = session_id)

@app.post("/reset")
def reset (req:AskRequest):
    if req.session_id and req.session_id in sessions:
        sessions[req.session_id] = []
    return{"status":"ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{Path(file.filename).name}"
    dest_path = docs_dir / safe_name

    # Save the uploaded file to disk first
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        file.file.close()


    # Run ingestion synchronously and add straight into the already-connected db.
    # NOTE: this blocks the request until ingestion finishes — on CPU-only hardware
    # with a vision model, this can take several minutes for PDFs with many
    # tables/diagrams. See the note below the endpoint for a background-task version.
    try:
        chunks_added = ingest_pdf(str(dest_path), db)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
        status_code=500,
        detail=f"Ingestion failed: {e}"
    )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks_added": chunks_added
    }

@app.get("/stats")
def stats():
    collection = db._collection
    chunk_count = collection.count()

    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    sources = {m.get("source", "unknown") for m in all_metadata if m}

    total_bytes = sum(
        f.stat().st_size for f in Path(persist_directory).rglob("*") if f.is_file()
    )
    return {
        "documents": len(sources),
        "chunks": chunk_count,
        "total_size_mb": round(total_bytes / (1024 * 1024), 1)
    }


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

app.mount("/static",StaticFiles(directory="frontend"), name="static")






