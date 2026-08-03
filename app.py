import gradio as gr
from rag_engine import index_from_uploaded_file
import time

def process_and_answer(file, query, history_index_state):
    """
    We keep the built index in Gradio's session state so we don't
    re-embed the document on every single question — only when a
    NEW file is uploaded.
    """
    if history_index_state is None and file is not None:
        t0 = time.time()
        history_index_state = index_from_uploaded_file(file.name)
        print(f"Index build: {time.time()-t0:.2f}s")

    if history_index_state is None:
        return "Please upload a document first.", history_index_state

    t1 = time.time()
    query_engine = history_index_state.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    print(f"Query: {time.time()-t1:.2f}s")
    return str(response), history_index_state


with gr.Blocks(title="LlamaIndex RAG Chatbot") as demo:
    gr.Markdown("## 📄 RAG Chatbot (LlamaIndex + Gradio)")
    index_state = gr.State(None)  # persists the index across turns

    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    query_input = gr.Textbox(label="Ask a question", lines=2)
    output_box = gr.Textbox(label="Answer")
    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn=process_and_answer,
        inputs=[file_input, query_input, index_state],
        outputs=[output_box, index_state],
    )

demo.launch(server_name="127.0.0.1", server_port=7860)