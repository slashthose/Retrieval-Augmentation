from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PERSIST_DIRECTORY = "dbv2/chroma_db"

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)


def retrieve(query: str, k: int = 3) -> list[tuple[Document, float]]:
    scored_chunks = db.similarity_search_with_score(query, k=k)
    return scored_chunks


# ---------------------------------------------------------------------------
# STANDALONE TEST — run this file directly to watch retrieval happen with
# nothing else involved. No LLM call, no prompt building, no API.
#
#     python retriever.py
#
# Change the query below to something you know is answered somewhere in
# your ingested PDFs, and read the output. You should be able to look at
# the printed chunk text and judge for yourself: "yes, that's relevant" or
# "no, that's wrong" — that judgment is literally what recall@k automates.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "What are transformers?"

    results = retrieve(test_query, k=3)

    print(f"Query: {test_query}\n")
    for i, (chunk, distance) in enumerate(results, start=1):
        print(f"--- Result {i} (distance={distance:.4f}) ---")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
        print()