import json
import os
from typing import List
from typing import cast, Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

#Unstructured for documment parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()

llm=ChatOllama(model="llava", temperature=0)

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, List


ASSETS_DIR = Path("assets")


def document_id_for(pdf_path: str) -> str:
    """Stable ID that also prevents accidental duplicate ingestion."""
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def page_number_for(element) -> int | None:
    return getattr(getattr(element, "metadata", None), "page_number", None)


def save_image_asset(
    image_base64: str,
    document_id: str,
    page_number: int | None,
    element_index: int,
) -> str:
    """Decode an extracted image and store it outside Chroma."""
    output_dir = ASSETS_DIR / document_id
    output_dir.mkdir(parents=True, exist_ok=True)

    page_label = page_number if page_number is not None else "unknown"
    filename = f"page_{page_label}_image_{element_index}.jpg"
    output_path = output_dir / filename

    image_bytes = base64.b64decode(image_base64)
    output_path.write_bytes(image_bytes)

    # Use forward slashes so the value is portable in API responses.
    return output_path.as_posix()



def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image","Table"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )

    print(f" Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("Creating smart chunks...")

    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )

    print(f" Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk) -> dict[str, Any]:
    """Extract text, tables, and images with page-level source metadata."""
    content_data: dict[str, Any] = {
        "text": chunk.text or "",
        "tables": [],
        "images": [],
        "page_numbers": set(),
    }

    original_elements = getattr(
        getattr(chunk, "metadata", None),
        "orig_elements",
        [],
    )

    for element_index, element in enumerate(original_elements):
        element_type = type(element).__name__
        page_number = page_number_for(element)

        if page_number is not None:
            content_data["page_numbers"].add(page_number)

        if element_type == "Table":
            table_html = getattr(
                getattr(element, "metadata", None),
                "text_as_html",
                element.text or "",
            )

            content_data["tables"].append({
                "element_index": element_index,
                "page_number": page_number,
                "html": table_html,
                "text": element.text or "",
            })

        elif element_type == "Image":
            image_base64 = getattr(
                getattr(element, "metadata", None),
                "image_base64",
                None,
            )

            if image_base64:
                content_data["images"].append({
                    "element_index": element_index,
                    "page_number": page_number,
                    "base64": image_base64,
                })

    content_data["page_numbers"] = sorted(content_data["page_numbers"])
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Generate a searchable text description covering text + tables + images.

    This is what actually gets embedded — not the raw text. The raw text,
    table HTML, and image base64 are preserved separately in metadata
    (see summarise_chunks) so app.py can show the real evidence back to the
    user even though the *searchable* representation is this AI summary.
    """
    try:
        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        message = HumanMessage(content=cast(Any, message_content))
        response = llm.invoke([message])
        return str(response.content)

    except Exception as e:
        print(f"AI summary failed, falling back to raw text: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def create_table_summary(table_html:str, table_text:str) -> str:
    return create_ai_enhanced_summary(
        text=f"Table content :\n {table_text}",
        tables=[table_html],
        images=[],
    )

def create_image_summary(image_base64: str) -> str:
    return create_ai_enhanced_summary(
        text="Describe this image or diagram for document retrieval. "
             "Include labels, entities, arrows, relationships, and numbers.",
        tables=[],
        images=[image_base64],
    )

def summarise_chunks(
    chunks,
    source_name: str,
    document_id: str,
) -> List[Document]:
    documents: List[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        content = separate_content_types(chunk)

        pages = content["page_numbers"]
        page_start = pages[0] if pages else None
        page_end = pages[-1] if pages else None

        # A. Text record
        if content["text"].strip():
            text_chunk_id = f"{document_id}:text:{chunk_index}"

            documents.append(
                Document(
                    page_content=content["text"],
                    metadata={
                        "chunk_id": text_chunk_id,
                        "document_id": document_id,
                        "source": source_name,
                        "modality": "text",
                        "page_start": page_start,
                        "page_end": page_end,
                        "related_assets": json.dumps([]),
                    },
                )
            )

        # B. Individual table records
        for table_index, table in enumerate(content["tables"]):
            table_chunk_id = f"{document_id}:table:{chunk_index}:{table_index}"

            table_summary = create_table_summary(
                table_html=table["html"],
                table_text=table["text"],
            )

            documents.append(
                Document(
                    page_content=table_summary,
                    metadata={
                        "chunk_id": table_chunk_id,
                        "document_id": document_id,
                        "source": source_name,
                        "modality": "table",
                        "page_number": table["page_number"],
                        # Safe enough for small tables. For huge tables, save
                        # HTML/CSV in assets too and store only a path here.
                        "table_html": table["html"],
                        "table_text": table["text"],
                    },
                )
            )

        # C. Individual image/diagram records
        for image_index, image in enumerate(content["images"]):
            image_chunk_id = f"{document_id}:image:{chunk_index}:{image_index}"

            image_path = save_image_asset(
                image_base64=image["base64"],
                document_id=document_id,
                page_number=image["page_number"],
                element_index=image["element_index"],
            )

            image_summary = create_image_summary(image["base64"])

            documents.append(
                Document(
                    page_content=image_summary,
                    metadata={
                        "chunk_id": image_chunk_id,
                        "document_id": document_id,
                        "source": source_name,
                        "modality": "image",
                        "page_number": image["page_number"],
                        "asset_path": image_path,
                    },
                )
            )

    return documents

def ingest_pdf(pdf_path: str, db: Chroma) -> int:
    """Run the full ingestion pipeline on a single PDF and add the resulting
    chunks into an already-connected Chroma `db` (does not create a new
    Chroma client, avoiding file-lock conflicts with app.py's live connection).

    Returns the number of chunks added.
    """


    source_name = os.path.basename(pdf_path)
    document_id = document_id_for(pdf_path)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)

    processed_chunks = summarise_chunks(
    chunks,
    source_name=source_name,
    document_id=document_id,
    )

    db.add_documents(processed_chunks)

    print(f"Added {len(processed_chunks)} chunks from {source_name} to the vector store")
    return len(processed_chunks)
    from image_ingest import ingest_image_records
    ingest_image_records(image_records, chroma_client)


if __name__ == "__main__":
    # Standalone CLI: bulk-ingest every PDF in ./docs into dbv2/chroma_db.
    # Useful for populating the store the first time, separate from the
    # single-file /upload endpoint in app.py.
    from langchain_huggingface import HuggingFaceEmbeddings

    docs_dir = "docs"
    persist_directory = "dbv2/chroma_db"

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"'{docs_dir}' does not exist — add your PDFs there first.")

    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{docs_dir}'.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    for filename in pdf_files:
        ingest_pdf(os.path.join(docs_dir, filename), db)

