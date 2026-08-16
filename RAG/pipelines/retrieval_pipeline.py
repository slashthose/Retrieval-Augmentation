from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistant_directory ="db/chroma_db"

#Load embeddings and vectory store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db= Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}



)



query = "What did NVIDIA released RIVA?"

retriever = db.as_retriever(search_kwargs={"k":2})
#retrieves top 3 chunks


relevant_docs = retriever.invoke(query)
print(f"User Query: {query}")

for i, doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

    combined_input = f"""Based on the following documents, please answer this question:{query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you cant find the answer in the documents, say " I dont have enough information to answer based on the provided documents"""


llm = ChatOllama(model="llama3", temperature=0)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)

]

result = llm.invoke(messages)

print("\n---Generated Response---")

print("Content only:")
print(result.content)
