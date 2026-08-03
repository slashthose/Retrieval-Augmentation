from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "dbv2/chroma_db"

TOP_K = 5

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)