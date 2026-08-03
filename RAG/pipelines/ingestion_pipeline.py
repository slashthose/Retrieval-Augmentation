import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f'Loading Documents from{docs_path}...')


    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path}does not exist.Please create it and add your company files.")

#Loading Files

    loader = DirectoryLoader(
        path=docs_path,
        glob='*.txt',  #only look for text files
        loader_cls= TextLoader
        )


    documents=loader.load()#list of lang chain docs


    if len(documents)==0:#error handling
        raise FileNotFoundError(f"No txt files found in {docs_path}.Please add your company files.")



    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source:  {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)}characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents):
    text_splitter= CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks,persist_directory="db/chroma_db"):
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    db=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print(f"\n Vector store created and persisted at {persist_directory}")
    return db


def main():
    print("Main Function")

    documents=load_documents(docs_path="docs")
    chunks=split_documents(documents)
    create_vector_store(chunks)



if __name__=="__main__":
    main()

