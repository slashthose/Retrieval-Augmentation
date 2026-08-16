import os
import shutil
import tempfile

from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

PERSIST_DIR = "storage"



Settings.llm = Ollama(model="llama3",
                       temperature=0,
                      request_timeout=120.0,
                      additional_kwargs={"keep_alive": "30m"})
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def build_or_load_index(data_dir="docs"):
    """
    Build the index once, then cache it to disk.
    Why cache? Re-embedding the same documents every time the app starts
    wastes time and API calls. This mirrors production RAG systems.
    """
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


def get_query_engine(index, top_k=3):
    return index.as_query_engine(similarity_top_k=3)

def index_from_uploaded_file(file_path):
    work_dir = tempfile.mkdtemp()
    shutil.copy(file_path, work_dir)
    documents = SimpleDirectoryReader(work_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def answer(question):
    return str(get_query_engine.query(question))
