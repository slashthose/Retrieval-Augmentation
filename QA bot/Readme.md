# RAG Lanchain & Gradio Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about its contents. Runs fully on-device using **Ollama** for the LLM and **HuggingFace** embeddings — no API keys, no cloud costs.

Includes two parallel implementations for comparison:
- **`qabot.py`** — built with **LangChain** (course-lab reference version)

---

## Features

- 📄 Upload a PDF and ask natural-language questions about it
- 🧠 Fully local inference — no OpenAI/API key required
- ⚡ Index caching so documents aren't re-embedded on every query
- 🔌 CPU-only friendly (no GPU required)
- 🖥️ Simple Gradio web UI

---

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Index
                                                            │
User Question → Embed Query → Retrieve Top-K Chunks → LLM ─┘
                                                            │
                                                       Answer
```


| Stage | LangChain version |
|---|---|
| Document loading | `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `HuggingFaceEmbeddings` (langchain) |
| Vector store | `Chroma` |
| LLM | `OllamaLLM` (langchain-ollama) |
| Query orchestration | `RetrievalQA` chain |
| UI | Gradio Blocks |

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running locally
- Pull the required models before first run:
  ```powershell
  ollama pull llama3
  ```

---

## Setup

1. **Clone / open the project folder**

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Start Ollama** (if not already running)
   ```powershell
   ollama serve
   ```

---

## Running the app

### LangChain version (course-lab reference)
```powershell
python qabot.py
```

Then open your browser to:
```
http://127.0.0.1:7860
```

> ⚠️ Note: Gradio prints `Running on local URL: http://0.0.0.0:7860` — this is **not** a browsable address. Always use `127.0.0.1` or `localhost` instead.

---

## Usage

1. Upload a PDF using the file uploader.
2. Type a question in the text box.
3. Click **Submit**.
4. The answer appears in the output box, grounded in the uploaded document.

The index is cached per-session (`gr.State`), so re-asking questions on the same document doesn't re-embed it — only a new upload triggers re-embedding.

---

## Project structure

```
QA bot/
|__app.py
├── qabot.py               # Standalone LangChain implementation
|__rag_engine.py           
├── requirements.txt
└── README.md
```

---

## Configuration

Key settings you may want to tune, in `rag_engine.py` / `qabot.py`:

| Setting | Location | Effect |
|---|---|---|
| `model="llama3"` | `OllamaLLM(...)` | Swap for a smaller model (e.g. `llama3.2:1b`) for faster CPU testing |
| `similarity_top_k` | `as_query_engine(similarity_top_k=3)` | Fewer chunks retrieved = faster, less context |
| `chunk_size`, `chunk_overlap` | `RecursiveCharacterTextSplitter` | Larger chunks = more context per chunk, fewer total chunks |
| `keep_alive="30m"` | LLM init | Keeps the Ollama model resident in memory between queries to avoid reload latency |

---

## Performance notes

Running fully on CPU means generation speed is the main bottleneck, not retrieval:

- Use a smaller Ollama model (`llama3.2:1b`, `phi3:mini`) while developing/testing.
- Set `keep_alive` so Ollama doesn't unload the model between queries.
- Lower `similarity_top_k` to reduce prompt size sent to the LLM.
- Load the embedding model and index **once** at startup — never rebuild inside the query function.

---

## Known limitations

- Single PDF at a time (no multi-document corpus support yet).
- No conversation memory — each question is answered independently, without chat history context.
- CPU-only inference is noticeably slower than GPU; expect longer response times for larger documents or bigger models.

---

## Roadmap ideas

- [ ] Multi-document upload and cross-document retrieval
- [ ] Conversational memory (multi-turn context)
- [ ] Source citation display (which chunk/page an answer came from)
- [ ] Streaming responses in the Gradio UI
- [ ] Deploy to Render/Vercel with a lightweight hosted LLM fallback

---

## Tech stack

`Python` · `LlamaIndex` · `LangChain` · `Gradio` · `Ollama` · `HuggingFace Embeddings (BAAI/bge-small-en-v1.5)` · `ChromaDB`
