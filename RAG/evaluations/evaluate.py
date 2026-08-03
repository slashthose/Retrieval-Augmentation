from langchain_chroma import Chroma

from config import *

from utils import *

from metrics import *

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model
)

queries = load_queries("labeled_queries.json")

results = []

for sample in queries:

    question = sample["query"]

    relevant = sample["relevant_chunks"]

    docs = vectorstore.similarity_search(
        question,
        k=TOP_K
    )

    retrieved = [
        doc.metadata["chunk_id"]
        for doc in docs
    ]

    recall = recall_at_k(
        retrieved,
        relevant
    )

    precision = precision_at_k(
        retrieved,
        relevant
    )

    hitrate = hit_rate_at_k(
        retrieved,
        relevant
    )

    rr = reciprocal_rank(
        retrieved,
        relevant
    )

    ndcg = ndcg_at_k(
        retrieved,
        relevant
    )

    results.append({

        "Question": question,

        "Recall@5": recall,

        "Precision@5": precision,

        "HitRate@5": hitrate,

        "MRR": rr,

        "nDCG": ndcg

    })

save_results(
    results,
    "results.csv"
)

print("Evaluation Complete")

print()

print("Average Recall:",

      sum(r["Recall@5"] for r in results)/len(results))

print("Average Precision:",

      sum(r["Precision@5"] for r in results)/len(results))

print("Average MRR:",

      sum(r["MRR"] for r in results)/len(results))