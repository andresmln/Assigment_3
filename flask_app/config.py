"""
Configuration module — reads all settings from environment variables.
"""
import os


class Config:
    # MinIO
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # ChromaDB
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_chunks")

    # LLM (llama.cpp server)
    LLM_HOST = os.getenv("LLM_HOST", "localhost")
    LLM_PORT = int(os.getenv("LLM_PORT", "8080"))

    # Chunking defaults
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @classmethod
    def llm_url(cls):
        return f"http://{cls.LLM_HOST}:{cls.LLM_PORT}"

    @classmethod
    def chromadb_url(cls):
        return f"http://{cls.CHROMADB_HOST}:{cls.CHROMADB_PORT}"
