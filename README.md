pip install -r requirements.txt# 🕉️ Sanskrit RAG System

**Retrieval-Augmented Generation for Sanskrit Documents — CPU-Only**

> Author: Prashant Darshanwar
> Date: 29 May 2026

---

## Overview

A fully CPU-based Question Answering pipeline over Sanskrit documents. The system ingests Sanskrit texts (PDF, TXT, DOCX), indexes them using dense vector embeddings, and answers questions in **English or Devanagari script** using a quantized local LLM — no GPU required.

---

## Project Structure

```
RAG_SANSKRIT_PRASHANT/
├── code/
│   ├── __pycache__/                        # Python bytecode cache (auto-generated)
│   ├── .env                                # Environment variables (API keys, paths)
│   ├── app.py                              # Streamlit web UI entry point
│   ├── ingest.py                           # Offline: load → clean → chunk → embed → FAISS
│   ├── rag_pipeline.py                     # Online: query → retrieve → generate → answer
│   ├── requirements.txt                    # Python dependencies
│   ├── test.py                             # Unit / integration tests
│   └── utils.py                            # Shared helpers (clean_text, strip_e5_prefix, etc.)
├── data/
│   ├── sanskrit_docs/
│   │   └── Rag-docs.txt                    # Sanskrit source corpus (5 stories)
│   └── vectorstore/
│       ├── index.faiss                     # FAISS binary index (auto-generated)
│       └── index.pkl                       # Docstore pickle (auto-generated)
├── model/
│   ├── ggml-model-q4_0.gguf               # Fallback / alternate GGUF model
│   └── qwen2.5-1.5b-instruct-q4_0.gguf   # Primary LLM (Qwen2.5 1.5B Q4_0)
├── report/
│   └── Sanskrit_RAG_Technical_Report.docx
├── venv/                                   # Python virtual environment (excluded from git)
└── README.md
```

---

## Requirements

- Python 3.10+
- ~2 GB RAM free
- No GPU needed

### Install dependencies

```bash
pip install langchain langchain-community langchain-huggingface langchain-text-splitters
pip install faiss-cpu sentence-transformers transformers torch
pip install llama-cpp-python pymupdf docx2txt
```

Or install from the provided file:

```bash
pip install -r requirements.txt
pip install langchain-huggingface langchain-text-splitters llama-cpp-python pymupdf docx2txt
```

---

## Download the LLM

Download the GGUF model from HuggingFace and place it in the `model/` folder:

```
Model : Qwen2.5-1.5B-Instruct (Q4_0 GGUF)
File  : qwen2.5-1.5b-instruct-q4_0.gguf
Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
```

```bash
mkdir -p model
# Download manually or use huggingface-hub:
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_0.gguf --local-dir model/
```

> The `model/` folder also contains `ggml-model-q4_0.gguf` as a fallback/alternate model. The pipeline uses `qwen2.5-1.5b-instruct-q4_0.gguf` by default (`MODEL_PATH` in `rag_pipeline.py`).

---

## Usage

### Step 1 — Add your Sanskrit documents

Copy your `.pdf`, `.txt`, or `.docx` files into `data/sanskrit_docs/`. The file `Rag-docs.txt` is already included.

### Step 2 — Build the vector index

```bash
python code/ingest.py
```

This will clean and chunk the documents, generate embeddings using `intfloat/multilingual-e5-base`, and save the FAISS index to `data/vectorstore/`.

**Alternative** — build directly from a single `.txt` file:

```bash
python code/rag_pipeline.py --build data/Rag-docs.txt
```

### Step 3 — Start the QA system

```bash
python code/rag_pipeline.py
```

You will see a prompt. Type your question in English or Sanskrit (Devanagari):

```
Ask a question:
> Who is the foolish servant?

Answer:
Shankhanaad is the foolish servant.
```

Type `exit` or `quit` to stop.

---

## Sample Questions

| Language | Question |
|----------|----------|
| English | Who is the foolish servant? |
| English | What did Shankhanaad do with the milk? |
| English | How did Kalidasa help the poet get one lakh rupees? |
| English | How did the old woman defeat Ghantakarna? |
| English | Why did the foreign scholar turn back? |
| English | What is the moral of the devotee story? |
| Sanskrit | शंखनादः कः आसीत्? |
| Sanskrit | वृद्धा किम् अकरोत्? |
| Sanskrit | कालीदासः पण्डितेन सह किम् अकरोत्? |

---

## Configuration

Key settings are at the top of each script:

**`ingest.py`**

| Variable | Default | Description |
|----------|---------|-------------|
| `FOLDER_PATH` | `../data/sanskrit_docs` | Source document folder |
| `VECTORSTORE_PATH` | `../data/vectorstore` | Where to save the FAISS index |
| `CHUNK_SIZE` | `400` | Max characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `MIN_CHUNK_LEN` | `40` | Drop chunks shorter than this |

**`rag_pipeline.py`**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../model/qwen2.5-1.5b-instruct-q4_0.gguf` | Path to LLM |
| `TOP_K` | `8` | Number of chunks to retrieve |
| `SCORE_THRESHOLD` | `1.15` | L2 distance cutoff (lower = stricter) |
| `MAX_CONTEXT_CHARS` | `2000` | Max context sent to LLM |
| `MAX_TOKENS` | `120` | Max tokens in generated answer |
| `TEMPERATURE` | `0.05` | Near-deterministic generation |

---

## How It Works

```
INGESTION (offline)
Documents → clean_text() → RecursiveCharacterTextSplitter
         → "passage: " prefix → multilingual-e5-base embeddings
         → FAISS vector store (saved to disk)

QUERY (online)
User input → clean_text() → detect_language()
           → "query: " prefix → FAISS similarity search (Top-8)
           → score filter (L2 < 1.15) → strip prefix → build context
           → Qwen2.5 ChatML prompt → llama-cpp generation → answer
```

The embedding model (`intfloat/multilingual-e5-base`) supports 100+ languages including Devanagari, enabling semantic retrieval across Sanskrit text without Sanskrit-specific fine-tuning.

---

## Troubleshooting

**All retrieved chunks show SKIP in the score log**
→ Raise `SCORE_THRESHOLD` slightly (e.g. `1.20` or `1.25`) in `rag_pipeline.py`.

**"No documents found" during ingest**
→ Check that `FOLDER_PATH` points to the correct directory and files have `.pdf`, `.txt`, or `.docx` extensions.

**LLM load error**
→ Verify the GGUF file path matches `MODEL_PATH` in `rag_pipeline.py` exactly.

**Slow generation**
→ Increase `n_threads` in `rag_pipeline.py` to match your CPU core count. Decrease `MAX_TOKENS` for shorter answers.

---

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Document loading | LangChain Community (PyMuPDF, TextLoader, Docx2txt) |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | `intfloat/multilingual-e5-base` via HuggingFace |
| Vector store | FAISS (faiss-cpu) |
| LLM | Qwen2.5-1.5B-Instruct Q4_0 GGUF |
| LLM backend | llama-cpp-python |

---

## .gitignore Recommendation

Add the following to `.gitignore` before pushing to GitHub to avoid committing large binary files:

```
venv/
__pycache__/
*.gguf
data/vectorstore/
.env
```

---

*Sanskrit RAG System — Prashant Darshanwar — 29 May 2026*