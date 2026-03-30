# RAG Document Intelligence System

A Question Answering system based on Retrieval-Augmented Generation (RAG) that allows users to upload documents (PDF/DOCX), index them, and ask questions about their content using a local LLM.

## Architecture

The system is composed of 5 services orchestrated with Docker Compose:

| Service | Technology | Port | Description |
|---------|------------|------|-------------|
| **MinIO** | minio/minio | 9000 / 9001 | Object storage for the original documents |
| **ChromaDB** | chromadb/chroma | 8000 | Vector database for chunk embeddings |
| **LLM** | llama.cpp | 8080 | Inference server running the Qwen2.5-3B model (Q4_K_M) |
| **Flask API** | Flask + Python 3.11 | 5000 | REST API that orchestrates the RAG pipeline |
| **Frontend** | Streamlit | 8501 | Web interface for uploading documents and asking questions |

## Project Structure

```
.
├── docker-compose.yml          # Service orchestration
├── flask_app/                  # Backend - RAG system REST API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                  # Endpoints: /documents, /query, /health
│   ├── config.py               # Configuration via environment variables
│   ├── ingestion.py            # PDF/DOCX parsing and MinIO storage
│   ├── chunking.py             # Chunking strategies (fixed_size, recursive)
│   ├── retrieval.py            # Embeddings (BGE-base-en-v1.5) + ChromaDB search
│   └── llm_client.py           # Client for the llama.cpp server (OpenAI-compatible API)
├── frontend/                   # Frontend - Streamlit web interface
│   ├── Dockerfile
│   └── app.py                  # UI with tabs: Upload, Documents, Ask Questions
├── models/                     # LLM models in GGUF format
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
├── eval/                       # Retrieval evaluation
│   ├── eval_dataset.json       # Dataset with 15 questions and ground truth
│   ├── run_eval.py             # Evaluation script (Hit Rate, MRR, Precision @k)
│   ├── create_eval_docs.py     # Evaluation document generation
│   └── sample_docs/            # Sample documents for evaluation
│       ├── artificial_intelligence_overview.pdf
│       ├── climate_change_report.pdf
│       └── python_programming_guide.docx
├── results/                    # Evaluation results
│   └── eval_results.json
├── data/                       # Assignment 2 data (Human Value Detection)
│   ├── arguments-*.tsv
│   ├── labels-*.tsv
│   └── value-categories.json
├── report/                     # Project report in LaTeX
│   └── main.tex
├── Assigment 2.ipynb           # Assignment 2 notebook
├── assignment2_complete.py     # Assignment 2 complete script
└── assignment 3.pdf            # Assignment 3 specification
```

## RAG Pipeline

1. **Ingestion**: The user uploads a PDF or DOCX file. Text is extracted (pdfplumber / python-docx) and the original file is stored in MinIO.
2. **Chunking**: The text is split into fragments using one of two strategies:
   - `fixed_size`: sliding window of 512 words with 50-word overlap.
   - `recursive`: hierarchical splitting by paragraphs, sentences, and words.
3. **Indexing**: Each chunk is converted into an embedding using `BAAI/bge-base-en-v1.5` (Sentence Transformers) and stored in ChromaDB with metadata.
4. **Retrieval**: Given a question, the query embedding is generated and the top-k most similar chunks are retrieved (cosine similarity) with deduplication.
5. **Generation**: The relevant chunks are sent as context to the LLM (Qwen2.5-3B via llama.cpp) to generate an answer with `[Source N]` citations.

## Usage

### Prerequisites

- Python >= 3.10
- Docker and Docker Compose
- The GGUF model file at `models/qwen2.5-3b-instruct-q4_k_m.gguf`

### Setup (local Python environment)

Create and activate a virtual environment, then install all dependencies from `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

> **Note (Debian/Ubuntu):** If `python3 -m venv` fails, install the venv package first:
> ```bash
> sudo apt install python3-venv
> ```

### Start the Services (Docker)

The RAG system runs as 5 Docker containers. Launch everything with:

```bash
docker compose up --build
```

### Access the Application

- **Frontend (Streamlit)**: http://localhost:8501
- **REST API (Flask)**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (user: `minioadmin`, password: `minioadmin`)
- **ChromaDB**: http://localhost:8000

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents` | Upload a PDF/DOCX document |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/<id>` | Delete a document and its chunks |
| `POST` | `/query` | Ask a question about the documents |
| `GET` | `/health` | Health check for all services |

### Run the Evaluation

With the Docker services running and the virtual environment activated:

```bash
python eval/run_eval.py
```

Computes Hit Rate @k, MRR @k, and Precision @k for k = {1, 3, 5} on a dataset of 15 questions with ground truth across 3 sample documents (AI, Climate Change, Python).

## Authors
Andrés Malón Insausti & Roberto Aldanondo


