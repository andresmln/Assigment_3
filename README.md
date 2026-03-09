# 📄 RAG Document Intelligence System

A complete, locally-running **Retrieval-Augmented Generation (RAG)** system for document question-answering. Upload PDF and DOCX files, ask natural-language questions, and receive **grounded answers with source citations**.

All services run locally via Docker — **no external APIs, no cloud costs**.

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│  Flask API   │────▶│   MinIO      │
│   Frontend   │     │  (port 5000) │     │  (port 9000) │
│  (port 8501) │     │              │     │  Object Store │
└──────────────┘     │  Orchestrator│     └──────────────┘
                     │              │
                     │  ┌────────┐  │     ┌──────────────┐
                     │  │SQLite  │  │────▶│  ChromaDB    │
                     │  │Metadata│  │     │  (port 8000) │
                     │  └────────┘  │     │  Vector Store │
                     │              │     └──────────────┘
                     │              │
                     │              │     ┌──────────────┐
                     │              │────▶│  llama.cpp   │
                     │              │     │  (port 8080) │
                     └──────────────┘     │  LLM Server  │
                                          └──────────────┘
```

### Pipeline Stages

1. **Ingest** — Parse PDF/DOCX, detect scanned PDFs, store raw files in MinIO
2. **Chunk** — Fixed-size (512 words, 50 overlap) or recursive splitting
3. **Index** — Embed with `all-MiniLM-L6-v2`, store in ChromaDB
4. **Retrieve** — Cosine similarity search, return top-k chunks
5. **Generate** — Build grounded prompt, call llama.cpp, return cited answer

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Assigment 3"
```

### 2. Download the LLM model

Download the **TinyLlama-1.1B-Chat** GGUF model (~670 MB):

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download the model
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

> ⚠️ **Do NOT commit model files to Git.** They are 670 MB+ and are excluded via `.gitignore`.

### 3. Start all services

```bash
docker-compose up --build
```

This starts **5 services**:

| Service      | Port | URL                        |
|-------------|------|----------------------------|
| Flask API   | 5000 | http://localhost:5000       |
| MinIO       | 9000 | http://localhost:9000       |
| MinIO Console| 9001| http://localhost:9001       |
| ChromaDB    | 8000 | http://localhost:8000       |
| LLM Server  | 8080 | http://localhost:8080       |
| Frontend    | 8501 | http://localhost:8501       |

### 4. Verify everything is running

```bash
curl http://localhost:5000/health
```

## 📡 API Endpoints

### Health Check

```bash
curl http://localhost:5000/health
```

### Upload a Document

```bash
# Upload a PDF
curl -X POST -F "file=@document.pdf" http://localhost:5000/documents

# Upload a DOCX with recursive chunking
curl -X POST -F "file=@document.docx" -F "strategy=recursive" http://localhost:5000/documents
```

### List Documents

```bash
curl http://localhost:5000/documents
```

### Delete a Document

```bash
curl -X DELETE http://localhost:5000/documents/<document-id>
```

### Query (Ask a Question)

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 5}'
```

**Response format:**
```json
{
    "question": "What is machine learning?",
    "answer": "Based on the documents, machine learning is... [Source 1]",
    "sources": [
        {
            "doc_id": "abc-123",
            "filename": "ai_overview.pdf",
            "chunk_index": 3,
            "score": 0.8542,
            "text": "Machine learning is a subset of AI..."
        }
    ],
    "num_sources": 5
}
```

## 🧪 Evaluation

### Run Retrieval Evaluation

```bash
# First upload the evaluation documents, then:
python eval/run_eval.py
```

Results are saved to `results/eval_results.json` with:
- **Hit Rate @k** — Did the relevant doc appear in top-k?
- **MRR** — Mean Reciprocal Rank
- **Precision @k** — Fraction of relevant results in top-k

## 🛠️ Technical Details

| Component       | Technology                              |
|----------------|----------------------------------------|
| LLM            | TinyLlama-1.1B-Chat (Q4_K_M GGUF)     |
| LLM Server     | llama.cpp (llama-server)               |
| Embeddings     | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store   | ChromaDB                                |
| Object Store   | MinIO                                   |
| API Framework  | Flask                                   |
| Frontend       | Streamlit                               |
| PDF Parsing    | pdfplumber                              |
| DOCX Parsing   | python-docx                             |
| Metadata       | SQLite                                  |

## ⚠️ Known Limitations

1. **TinyLlama (1.1B)** is a small model — answer quality improves significantly with larger models (e.g., Mistral-7B)
2. **No OCR support** — scanned PDFs are detected and rejected with an error message
3. **CPU-only inference** — generation can take 15-60 seconds depending on hardware
4. **Fixed embedding model** — `all-MiniLM-L6-v2` works well for English but is limited for multilingual content
5. **No authentication** — the API has no auth layer (intended for local development)

## 📁 Project Structure

```
├── docker-compose.yml          # Orchestrates all 5 services
├── flask_app/
│   ├── Dockerfile              # Flask container
│   ├── requirements.txt        # Python dependencies
│   ├── app.py                  # Flask API (5 endpoints)
│   ├── config.py               # Environment configuration
│   ├── ingestion.py            # PDF/DOCX parsing + MinIO
│   ├── chunking.py             # Fixed-size + recursive chunking
│   ├── retrieval.py            # Embeddings + ChromaDB search
│   └── llm_client.py           # llama.cpp HTTP client
├── frontend/
│   ├── Dockerfile              # Streamlit container
│   └── app.py                  # Web UI
├── models/                     # GGUF model files (NOT committed)
├── eval/
│   ├── eval_dataset.json       # Annotated evaluation questions
│   └── run_eval.py             # Evaluation metrics script
├── results/                    # Evaluation output
├── report/                     # Technical report
├── .gitignore
└── README.md
```
