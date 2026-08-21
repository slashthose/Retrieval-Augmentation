"""
Multimodal Retriever
--------------------
Retrieves and reranks:
    1. Text
    2. Tables
    3. Images

Design goals:
- Keep factual questions text/table focused.
- Retrieve images when the user explicitly asks about visual content.
- Do not force an image into every response.
- Do not assume MiniLM and SigLIP scores are directly comparable.
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

# General minimum score.
# This is deliberately lower than the previous 0.10 because
# image/SigLIP relevance scores are on a different scale.
RELEVANCE_FLOOR = 0.05


# Number of candidates retrieved from each modality.
TEXT_K = 5
TABLE_K = 3
IMAGE_K = 3


# Final number of evidence items.
TOP_N = 5


# Intent boosts.
IMAGE_BOOST = 1.50
TABLE_BOOST = 1.25


# Minimum boosted image score required before an image can
# be deliberately included for a visual-intent question.
IMAGE_MIN_SCORE = 0.10


# Minimum boosted table score required before a table can
# be deliberately included for a table-intent question.
TABLE_MIN_SCORE = 0.10


# ============================================================
# INTENT KEYWORDS
# ============================================================

IMAGE_INTENT_KEYWORDS = [
    "image",
    "images",
    "diagram",
    "diagrams",
    "figure",
    "figures",
    "chart",
    "charts",
    "graph",
    "graphs",
    "picture",
    "pictures",
    "photo",
    "photos",
    "screenshot",
    "visual",
    "visuals",
    "illustration",
    "illustrations",
    "look at",
    "look like",
    "show me",
    "what does",
    "what is shown",
    "what's shown",
    "shown in",
    "shown on",
    "according to the figure",
    "according to the diagram",
    "according to the chart",
]


TABLE_INTENT_KEYWORDS = [
    "table",
    "tables",
    "compare",
    "comparison",
    "row",
    "rows",
    "column",
    "columns",
    "spreadsheet",
    "breakdown",
    "numbers for",
    "statistics",
    "stats",
    "highest value",
    "lowest value",
]


# ============================================================
# HELPER: SEARCH ONE MODALITY
# ============================================================

def _search_modality(
    store: Chroma,
    query: str,
    k: int,
    modality: str | None = None,
) -> List[Tuple[Document, float]]:
    """
    Search one vector store, optionally restricted by modality.

    Returns:
        List of (Document, relevance_score)
    """

    if modality:
        filter_dict = {"modality": modality}

        return store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_dict,
        )

    return store.similarity_search_with_relevance_scores(
        query,
        k=k,
    )


# ============================================================
# HELPER: DETECT USER INTENT
# ============================================================

def _has_image_intent(query: str) -> bool:
    """
    Return True when the user explicitly asks for visual content.
    """

    q = query.lower()

    return any(
        keyword in q
        for keyword in IMAGE_INTENT_KEYWORDS
    )


def _has_table_intent(query: str) -> bool:
    """
    Return True when the user explicitly asks about tables,
    comparisons, rows, columns, statistics, etc.
    """

    q = query.lower()

    return any(
        keyword in q
        for keyword in TABLE_INTENT_KEYWORDS
    )


# ============================================================
# HELPER: MODALITY BOOST
# ============================================================

def _detect_intent_boost(
    query: str,
    modality: str,
) -> float:
    """
    Apply a boost only when the user's query indicates
    that the modality is useful.

    Important:
    We do NOT boost images for ordinary factual questions.
    """

    q = query.lower()

    if modality == "image":
        if _has_image_intent(q):
            return IMAGE_BOOST

        return 1.0

    if modality == "table":
        if _has_table_intent(q):
            return TABLE_BOOST

        return 1.0

    return 1.0


# ============================================================
# HELPER: GET BEST RESULT FOR A MODALITY
# ============================================================

def _best_modality_candidate(
    candidates: List[Tuple[Document, float]],
    modality: str,
):
    """
    Return the highest-scoring candidate for a specific modality.
    """

    modality_candidates = [
        (doc, score)
        for doc, score in candidates
        if doc.metadata.get("modality") == modality
    ]

    if not modality_candidates:
        return None

    return max(
        modality_candidates,
        key=lambda pair: pair[1],
    )


# ============================================================
# MAIN MULTIMODAL RERANKER
# ============================================================

def rerank_multimodal_results(
    query: str,
    text_db: Chroma,
    image_db: Chroma,
    text_k: int = TEXT_K,
    table_k: int = TABLE_K,
    image_k: int = IMAGE_K,
    top_n: int = TOP_N,
):
    """
    Retrieve text, tables, and images and return a ranked
    multimodal evidence set.

    Important behavior:

    Normal factual query:
        Text/table results naturally dominate.
        Images are NOT forced into the result.

    Visual query:
        The best sufficiently relevant image gets one
        reserved evidence slot.

    Table query:
        The best sufficiently relevant table gets one
        reserved evidence slot.
    """

    candidates: List[Tuple[Document, float]] = []


    # ========================================================
    # 1. TEXT RETRIEVAL
    # ========================================================

    text_results = _search_modality(
        text_db,
        query,
        text_k,
        "text",
    )

    for doc, score in text_results:

        boosted_score = (
            score
            * _detect_intent_boost(query, "text")
        )

        candidates.append(
            (doc, boosted_score)
        )


    # ========================================================
    # 2. TABLE RETRIEVAL
    # ========================================================

    table_results = _search_modality(
        text_db,
        query,
        table_k,
        "table",
    )

    for doc, score in table_results:

        boosted_score = (
            score
            * _detect_intent_boost(query, "table")
        )

        candidates.append(
            (doc, boosted_score)
        )


    # ========================================================
    # 3. IMAGE RETRIEVAL
    # ========================================================

    image_results = _search_modality(
        image_db,
        query,
        image_k,
        "image",
    )

    for doc, score in image_results:

        boosted_score = (
            score
            * _detect_intent_boost(query, "image")
        )

        candidates.append(
            (doc, boosted_score)
        )


    # ========================================================
    # 4. REMOVE VERY WEAK RESULTS
    # ========================================================

    candidates = [
        (doc, score)
        for doc, score in candidates
        if score >= RELEVANCE_FLOOR
    ]


    # ========================================================
    # 5. NORMAL RANKING
    # ========================================================

    candidates.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )


    # ========================================================
    # 6. VISUAL-INTENT HANDLING
    # ========================================================
    #
    # IMPORTANT:
    #
    # We DO NOT force an image into every answer.
    #
    # Only if the user explicitly asks about a visual AND
    # a sufficiently relevant image exists do we reserve
    # one slot for it.
    #

    wants_image = _has_image_intent(query)

    if wants_image:

        best_image = _best_modality_candidate(
            candidates,
            "image",
        )

        if best_image is not None:

            image_doc, image_score = best_image

            if image_score >= IMAGE_MIN_SCORE:

                # Remove the selected image from the
                # normal ranking before reserving its slot.
                candidates = [
                    (doc, score)
                    for doc, score in candidates
                    if not (
                        doc.metadata.get("modality") == "image"
                        and doc.id == image_doc.id
                    )
                ]

                # Reserve exactly one slot.
                candidates = candidates[
                    :max(0, top_n - 1)
                ]

                candidates.append(
                    best_image
                )


    # ========================================================
    # 7. TABLE-INTENT HANDLING
    # ========================================================
    #
    # Same principle:
    # A table is deliberately included only when the user
    # explicitly asks about tables/comparisons AND the table
    # is sufficiently relevant.
    #

    wants_table = _has_table_intent(query)

    if wants_table:

        best_table = _best_modality_candidate(
            candidates,
            "table",
        )

        if best_table is not None:

            table_doc, table_score = best_table

            if table_score >= TABLE_MIN_SCORE:

                candidates = [
                    (doc, score)
                    for doc, score in candidates
                    if not (
                        doc.metadata.get("modality") == "table"
                        and doc.id == table_doc.id
                    )
                ]

                candidates = candidates[
                    :max(0, top_n - 1)
                ]

                candidates.append(
                    best_table
                )


    # ========================================================
    # 8. FINAL SORT
    # ========================================================

    candidates.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )


    # ========================================================
    # 9. RETURN TOP N
    # ========================================================

    return candidates[:top_n]