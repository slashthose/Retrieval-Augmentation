"""
image_retriever.py

Query-time image search + fusion with your existing text-store KNN results.
Matches your actual stack: langchain_chroma.Chroma, not a raw chromadb
client. Uses chunk_id (already in your Document metadata) as the fusion key.
"""

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

RRF_K = 60  # standard default; lower = top ranks matter more


def image_similarity_search(query: str, image_store: Chroma, top_k: int = 10) -> List[Document]:
    """Search the SigLIP image collection with a real text query — routed
    through SiglipEmbeddings.embed_query(), i.e. SigLIP's text tower."""
    return image_store.similarity_search(query, k=top_k)


def reciprocal_rank_fusion(ranked_id_lists: List[List[str]], k: int = RRF_K) -> List[str]:
    """
    ranked_id_lists: each item is a list of chunk_ids, best-first, from one
    retrieval source (e.g. text-store KNN, image-store KNN).
    Rank-based fusion — avoids comparing raw similarity scores across the
    two different embedding spaces (text-summary embedder vs SigLIP).
    """
    scores: dict[str, float] = {}
    for ranking in ranked_id_lists:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def merge_text_and_image_results(
    query: str,
    text_results: List[Document],   # from your existing db.similarity_search() call
    image_store: Chroma,
    top_k_images: int = 10,
) -> dict:
    """
    Drop-in for your MERGE step. Returns the fused chunk_id ranking plus a
    lookup so build_evidence_and_message() can pull asset_path back out for
    whichever image chunk_ids made the cut.
    """
    text_ids = [d.metadata["chunk_id"] for d in text_results]

    image_results = image_similarity_search(query, image_store, top_k=top_k_images)
    image_ids = [d.metadata["chunk_id"] for d in image_results]

    fused_ranking = reciprocal_rank_fusion([text_ids, image_ids])

    doc_lookup = {d.metadata["chunk_id"]: d for d in text_results + image_results}

    return {
        "fused_ranking": fused_ranking,
        "doc_lookup": doc_lookup,
    }