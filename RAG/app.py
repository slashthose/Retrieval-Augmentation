import json
import shutil
import uuid
import traceback
from collections import deque
from datetime import datetime, timezone
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
from image_ingest import get_image_store
from image_retriever import rerank_multimodal_results
import base64

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

# Load the separate image vector store used by SigLIP
image_db = get_image_store()

llm = ChatOllama(model="llava", temperature=0)
sessions: dict[str, list[BaseMessage]] = {}

# Per-session metadata for the History tab — title, timestamps, message
# count. Kept in memory alongside `sessions`; not persisted across restarts.
session_meta: dict[str, dict[str, Any]] = {}

# Live state for the Knowledge Base tab's Indexing Status card, updated by
# /upload before/after ingestion so a concurrent GET /indexing-status
# (polled from the frontend) can see "processing" while /upload is still
# running.
indexing_status: dict[str, Any] = {
    "status": "idle",  # idle | processing | completed | error
    "filename": None,
    "started_at": None,
    "finished_at": None,
    "chunks_added": None,
    "error": None,
}

# Rolling log of recent retrievals for the Knowledge Base tab's relevance
# table — most recent first, capped so it never grows unbounded.
retrieval_log: deque = deque(maxlen=20)


def _format_date(iso_str: str | None) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
    return dt.strftime("%b %d, %Y \u00b7 %I:%M %p")


def _human_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / 1024:.0f} KB"


def _display_name(source: str) -> str:
    """Strip the uuid4 prefix /upload adds to stored filenames
    (format: '<36-char-uuid>_<original-filename>') for display."""
    name = Path(source).name
    if len(name) > 37 and name[36] == "_":
        try:
            uuid.UUID(name[:36])
            return name[37:]
        except ValueError:
            pass
    return name


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

def _load_image_base64(asset_path: str) -> str | None:
    """Load an extracted image asset for LLaVA and the frontend."""
    try:
        return base64.b64encode(Path(asset_path).read_bytes()).decode("utf-8")
    except (FileNotFoundError, OSError) as e:
        print(f"Could not load image asset {asset_path}: {e}")
        return None


def build_evidence_and_message(ranked_docs: list, query: str):
    """
    ranked_docs:
        List of (Document, relevance_score) pairs returned by
        rerank_multimodal_results(), ordered best-first.

    Text/table chunks come from the normal Chroma store.
    Image chunks come from the dedicated image store and point to
    extracted files through metadata['asset_path'].
    """
    prompt_text = (
        f"Based on the following documents, answer this question: {query}\n\n"
        "CONTENT TO ANALYZE:\n"
    )

    message_images: list[str] = []
    evidence: list[dict[str, Any]] = []

    for i, (chunk, raw_relevance) in enumerate(ranked_docs):
        relevance = round(max(0.0, min(1.0, float(raw_relevance))), 2)
        modality = chunk.metadata.get("modality", "text")

        prompt_text += f"--- Retrieved source {i + 1} ---\n"

        if modality == "table":
            table_html = chunk.metadata.get("table_html", "")

            # Some older chunks may not have table_html. Fall back to
            # page_content so retrieval still contributes useful context.
            if table_html:
                prompt_text += f"TABLE:\n{table_html}\n\n"
            else:
                prompt_text += (
                    f"TABLE TEXT:\n{chunk.page_content}\n\n"
                )

            evidence.append({
                "type": "table",
                "label": f"Document {i + 1} — table",
                "content": table_html or chunk.page_content,
                "score": relevance,
                "page_number": chunk.metadata.get("page_number")
                    or chunk.metadata.get("page_start"),
                "source": chunk.metadata.get("source"),
            })

        elif modality == "image":
            asset_path = chunk.metadata.get("asset_path", "")
            img_b64 = _load_image_base64(asset_path) if asset_path else None

            if img_b64:
                message_images.append(img_b64)

                prompt_text += (
                    "[RETRIEVED IMAGE/DIAGRAM ATTACHED BELOW. "
                    "Use the actual visual as evidence. "
                    "Do not invent details that are not visible in the image. "
                    f"Caption/description: {chunk.page_content}\n\n"
                )

                evidence.append({
                    "type": "image",
                    "label": f"Document {i + 1} — diagram",
                    "content": f"data:image/jpeg;base64,{img_b64}",
                    "asset_path": asset_path,
                    "score": relevance,
                    "page_number": chunk.metadata.get("page_number"),
                    "source": chunk.metadata.get("source"),
                })
            else:
                # Keep the retrieval result useful even if an asset was
                # deleted/moved after ingestion.
                prompt_text += (
                    f"IMAGE DESCRIPTION:\n{chunk.page_content}\n\n"
                )

                evidence.append({
                    "type": "text",
                    "label": f"Document {i + 1} — diagram (image unavailable)",
                    "content": chunk.page_content[:400],
                    "score": relevance,
                    "page_number": chunk.metadata.get("page_number"),
                    "source": chunk.metadata.get("source"),
                })

        else:
            prompt_text += f"TEXT:\n{chunk.page_content}\n\n"

            evidence.append({
                "type": "text",
                "label": f"Document {i + 1}",
                "content": chunk.page_content[:400],
                "score": relevance,
                "page_number": chunk.metadata.get("page_number")
                    or chunk.metadata.get("page_start"),
                "source": chunk.metadata.get("source"),
            })

    prompt_text += (
        "\nAnswer using the retrieved evidence above. "
        "Ground factual claims in the evidence. "
        "When a table or image is present, use it as evidence rather than "
        "inventing values or visual details. "
        "If the evidence does not contain the answer, say so.\n\nANSWER:"
    )

    message_content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_text}
    ]

    for img in message_images:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            },
        })

    return HumanMessage(content=message_content), evidence



def ask_question(user_question: str, chat_history: list):
    # Rewrite follow-up questions into a standalone query before retrieval.
    search_question = rewrite_standalone_question(
        user_question,
        chat_history,
    )

    # IMPORTANT: this is now the real multimodal retrieval path:
    #   text  -> shared Chroma store, modality=text
    #   table -> shared Chroma store, modality=table
    #   image -> dedicated SigLIP image store
    #   all  -> real relevance scores + reranking
    ranked_docs = rerank_multimodal_results(
        search_question,
        db,
        image_db,
    )

    human_message, evidence = build_evidence_and_message(
        ranked_docs,
        user_question,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant answering questions using "
                "retrieved text, tables, images, and conversation history. "
                "Use only the retrieved evidence for document-specific facts."
            )
        )
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

    # A fresh session, or one just cleared by /reset, starts a new
    # history entry titled after this question.
    if not chat_history:
        session_meta[session_id] = {
            "title": req.question[:60] + ("\u2026" if len(req.question) > 60 else ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    answer,evidence = ask_question(req.question, chat_history)

    top = evidence[0] if evidence else None
    retrieval_log.appendleft({
        "question": req.question,
        "top_source": top["label"] if top else "No sources retrieved",
        "top_source_type": top["type"] if top else "text",
        "relevance": top["score"] if top else 0,
        "timestamp": _format_date(datetime.now(timezone.utc).isoformat()),
    })

    meta = session_meta.setdefault(session_id, {
        "title": req.question[:60],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    meta["last_active"] = datetime.now(timezone.utc).isoformat()
    meta["message_count"] = len(chat_history)

    return AskResponse(answer=answer, evidence = evidence, session_id = session_id)

@app.post("/reset")
def reset (req:AskRequest):
    if req.session_id and req.session_id in sessions:
        sessions[req.session_id] = []
        session_meta.pop(req.session_id, None)
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
    # indexing_status is updated before/after so GET /indexing-status (polled from
    # a different request thread) can show live progress for the Knowledge Base tab
    # while this request is still in flight.
    indexing_status.update({
        "status": "processing",
        "filename": file.filename,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "chunks_added": None,
        "error": None,
    })

    try:
        chunks_added = ingest_pdf(str(dest_path), db)
    except Exception as e:
        traceback.print_exc()
        indexing_status.update({
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
        raise HTTPException(
        status_code=500,
        detail=f"Ingestion failed: {e}"
    )

    indexing_status.update({
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "chunks_added": chunks_added,
    })

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


@app.get("/documents")
def list_documents():
    """Backs the Documents tab: one card per uploaded PDF, aggregated
    from the chunk metadata already stored in the vector store."""
    collection = db._collection
    all_metadata = collection.get(include=["metadatas"])["metadatas"]

    by_source: dict[str, int] = {}
    for m in all_metadata:
        if not m:
            continue
        source = m.get("source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1

    docs = []
    for source, chunk_count in by_source.items():
        path = Path(source)
        size = _human_size(path.stat().st_size) if path.exists() else None
        docs.append({
            "name": _display_name(source),
            "type": "PDF",
            "size": size,
            "chunks": chunk_count,
        })

    docs.sort(key=lambda d: d["name"].lower())
    return docs


@app.get("/history")
def list_history():
    """Backs the History tab: past chat sessions with at least one
    exchanged message, most recently active first."""
    entries = [
        {
            "session_id": sid,
            "title": meta.get("title") or "Untitled session",
            "date": _format_date(meta.get("last_active") or meta.get("created_at")),
            "messages": meta.get("message_count", 0),
            "_sort_key": meta.get("last_active") or meta.get("created_at") or "",
        }
        for sid, meta in session_meta.items()
        if meta.get("message_count", 0) > 0
    ]
    entries.sort(key=lambda e: e["_sort_key"], reverse=True)
    for e in entries:
        del e["_sort_key"]
    return entries


@app.get("/indexing-status")
def get_indexing_status():
    """Backs the Knowledge Base tab's Indexing Status card. The frontend
    polls this while a file is being ingested so the status reads as
    live — 'idle' the rest of the time."""
    return indexing_status


@app.get("/relevance-log")
def get_relevance_log():
    """Backs the Knowledge Base tab's relevance table — the top-scoring
    source retrieved for each of the last 20 questions asked."""
    return list(retrieval_log)


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

app.mount("/static",StaticFiles(directory="frontend"), name="static")



# Serve extracted document images for direct browser access when needed.
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
