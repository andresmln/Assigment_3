"""
Flask REST API — Central orchestration layer for the RAG system.
Exposes 5 endpoints: POST/GET/DELETE /documents, POST /query, GET /health.
"""
import os
import uuid
import json
import sqlite3
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from config import Config
from ingestion import (
    get_minio_client,
    ensure_bucket,
    upload_to_minio,
    delete_from_minio,
    parse_document,
)
from chunking import chunk_document
from retrieval import index_chunks, delete_chunks_by_doc_id, retrieve
from llm_client import generate_answer, check_llm_health

import requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# SQLite metadata store
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("METADATA_DB", "/app/metadata.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the documents metadata table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_type TEXT,
            upload_date TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            chunk_strategy TEXT DEFAULT 'fixed_size',
            file_size INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# POST /documents — Upload a PDF or DOCX file
# ---------------------------------------------------------------------------
@app.route("/documents", methods=["POST"])
def upload_document():
    """Upload a PDF or DOCX file. Returns a document ID."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' form field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    lower_name = filename.lower()

    if not (lower_name.endswith(".pdf") or lower_name.endswith(".docx")):
        return jsonify({"error": "Unsupported format. Only PDF and DOCX are accepted."}), 400

    # Read file bytes
    file_bytes = file.read()
    doc_id = str(uuid.uuid4())
    content_type = file.content_type or "application/octet-stream"

    # Determine chunking strategy
    strategy = request.form.get("strategy", "fixed_size")
    if strategy not in ("fixed_size", "recursive"):
        strategy = "fixed_size"

    # Step 1: Parse document text
    try:
        text = parse_document(file_bytes, filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # Step 2: Upload raw file to MinIO
    try:
        minio_client = get_minio_client()
        object_name = f"{doc_id}/{filename}"
        upload_to_minio(minio_client, file_bytes, object_name, content_type)
    except Exception as e:
        return jsonify({"error": f"Failed to store file in MinIO: {str(e)}"}), 500

    # Step 3: Chunk the text
    chunks = chunk_document(text, doc_id, filename, strategy=strategy)

    # Step 4: Index chunks in ChromaDB
    try:
        num_indexed = index_chunks(chunks)
    except Exception as e:
        # Cleanup MinIO on failure
        try:
            delete_from_minio(minio_client, object_name)
        except Exception:
            pass
        return jsonify({"error": f"Failed to index chunks: {str(e)}"}), 500

    # Step 5: Store metadata in SQLite
    conn = get_db()
    conn.execute(
        "INSERT INTO documents (id, filename, content_type, upload_date, chunk_count, chunk_strategy, file_size) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (doc_id, filename, content_type, datetime.now(timezone.utc).isoformat(),
         num_indexed, strategy, len(file_bytes)),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "id": doc_id,
        "filename": filename,
        "chunks": num_indexed,
        "strategy": strategy,
        "message": "Document uploaded, parsed, and indexed successfully.",
    }), 201


# ---------------------------------------------------------------------------
# GET /documents — List all uploaded documents
# ---------------------------------------------------------------------------
@app.route("/documents", methods=["GET"])
def list_documents():
    """List all uploaded documents (ID, filename, upload date)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, content_type, upload_date, chunk_count, chunk_strategy, file_size "
        "FROM documents ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()

    documents = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "upload_date": row["upload_date"],
            "chunk_count": row["chunk_count"],
            "chunk_strategy": row["chunk_strategy"],
            "file_size": row["file_size"],
        }
        for row in rows
    ]

    return jsonify({"documents": documents, "total": len(documents)}), 200


# ---------------------------------------------------------------------------
# DELETE /documents/<id> — Delete a document and its indexed chunks
# ---------------------------------------------------------------------------
@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document and its indexed chunks."""
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    if row is None:
        conn.close()
        return jsonify({"error": f"Document {doc_id} not found."}), 404

    filename = row["filename"]
    errors = []

    # Delete from MinIO
    try:
        minio_client = get_minio_client()
        object_name = f"{doc_id}/{filename}"
        delete_from_minio(minio_client, object_name)
    except Exception as e:
        errors.append(f"MinIO deletion error: {str(e)}")

    # Delete from ChromaDB
    try:
        chunks_deleted = delete_chunks_by_doc_id(doc_id)
    except Exception as e:
        errors.append(f"ChromaDB deletion error: {str(e)}")
        chunks_deleted = 0

    # Delete from metadata store
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

    result = {
        "id": doc_id,
        "filename": filename,
        "chunks_deleted": chunks_deleted,
        "message": "Document deleted successfully.",
    }
    if errors:
        result["warnings"] = errors

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# POST /query — Submit a question, get a RAG answer
# ---------------------------------------------------------------------------
@app.route("/query", methods=["POST"])
def query():
    """Submit a question. Returns the generated answer and source chunks."""
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    top_k = data.get("top_k", Config.TOP_K)

    # Step 1: Retrieve relevant chunks
    try:
        sources = retrieve(question, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # Step 2: Generate answer with LLM
    try:
        result = generate_answer(question, sources)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    return jsonify({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "num_sources": len(result["sources"]),
    }), 200


# ---------------------------------------------------------------------------
# GET /health — Health check for all services
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check confirming all services are reachable."""
    status = {"status": "healthy", "services": {}}

    # Check MinIO
    try:
        minio_client = get_minio_client()
        minio_client.list_buckets()
        status["services"]["minio"] = "ok"
    except Exception as e:
        status["services"]["minio"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Check ChromaDB
    try:
        resp = requests.get(f"{Config.chromadb_url()}/api/v2/heartbeat", timeout=5)
        if resp.status_code == 200:
            status["services"]["chromadb"] = "ok"
        else:
            status["services"]["chromadb"] = f"error: status {resp.status_code}"
            status["status"] = "degraded"
    except Exception as e:
        status["services"]["chromadb"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Check LLM
    if check_llm_health():
        status["services"]["llm"] = "ok"
    else:
        status["services"]["llm"] = "error: unreachable"
        status["status"] = "degraded"

    # Check metadata DB
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        status["services"]["metadata_db"] = "ok"
    except Exception as e:
        status["services"]["metadata_db"] = f"error: {str(e)}"
        status["status"] = "degraded"

    http_code = 200 if status["status"] == "healthy" else 503
    return jsonify(status), http_code


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    # Ensure MinIO bucket exists on startup
    try:
        client = get_minio_client()
        ensure_bucket(client)
        print("[STARTUP] MinIO bucket ready.")
    except Exception as e:
        print(f"[STARTUP] MinIO not yet available: {e}")

    print("[STARTUP] Flask RAG API starting on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
