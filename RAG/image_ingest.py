import copy
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from image_embeddings import SiglipEmbeddings

IMAGE_PERSIST_DIRECTORY = "dbv2/chroma_image_db"
IMAGE_COLLECTION_NAME = "image_vectors"

_image_embedding_model = SiglipEmbeddings()


def get_image_store() -> Chroma:
    return Chroma(
        collection_name=IMAGE_COLLECTION_NAME,
        persist_directory=IMAGE_PERSIST_DIRECTORY,
        embedding_function=_image_embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_image_records(processed_chunks: List[Document], image_store: Chroma) -> int:
    """
    processed_chunks: the full list returned by summarise_chunks() in ingest.py
    (mixed text/table/image Documents). This filters to modality == "image"
    and indexes only those, by pixel content.

    Returns the number of image documents indexed.
    """
    image_docs = [d for d in processed_chunks if d.metadata.get("modality") == "image"]
    if not image_docs:
        return 0

    # page_content must be the asset_path for SiglipEmbeddings.embed_documents()
    # to pick up — copy the doc so the original (summary as page_content, used
    # by the text store) is untouched.
    pixel_docs = []
    for doc in image_docs:
        asset_path = doc.metadata.get("asset_path")
        if not asset_path:
            continue
        pixel_doc = copy.deepcopy(doc)
        pixel_doc.page_content = asset_path
        pixel_docs.append(pixel_doc)

    if not pixel_docs:
        return 0

    image_store.add_documents(pixel_docs)
    return len(pixel_docs)