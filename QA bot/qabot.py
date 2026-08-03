from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

llm = OllamaLLM(model="llama3", temperature=0, keep_alive="30m")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Documents Loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

# Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Vector db
def vector_database(chunks):
    embedding_model = embed_model
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever_obj,
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("## 📄 RAG Chatbot (Langchain + Gradio)")

    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    query_input = gr.Textbox(label="Ask a question", lines=2)
    output_box = gr.Textbox(label="Answer")
    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn=retriever_qa,
        inputs=[file_input, query_input],
        outputs=[output_box]
    )

demo.launch(server_name="127.0.0.1", server_port=7860)
