"""
Retrieval module — Bi-Encoder embeddings + ChromaDB vector search.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from config import Config

# Global model cache
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client() -> chromadb.HttpClient:
    """Create and return a ChromaDB HTTP client."""
    return chromadb.HttpClient(
        host=Config.CHROMADB_HOST,
        port=Config.CHROMADB_PORT,
    )


def get_or_create_collection(client: chromadb.HttpClient):
    """Get or create the RAG chunks collection."""
    return client.get_or_create_collection(
        name=Config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: List[Dict]) -> int:
    """
    Embed and index a list of chunks into ChromaDB.
    Returns the number of chunks indexed.
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [f"{c['doc_id']}_chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "strategy": c.get("strategy", "fixed_size"),
        }
        for c in chunks
    ]

    # ChromaDB upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    return len(chunks)


def delete_chunks_by_doc_id(doc_id: str):
    """Delete all chunks belonging to a document from ChromaDB."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Query to find all chunks for this doc_id
    results = collection.get(
        where={"doc_id": doc_id},
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])

    return len(results["ids"]) if results["ids"] else 0


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query.
    Returns a list of dicts with: text, doc_id, filename, chunk_index, score.
    """
    if top_k is None:
        top_k = Config.TOP_K

    model = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if collection is empty
    if collection.count() == 0:
        return []

    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances; for cosine, distance = 1 - similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            retrieved.append({
                "text": results["documents"][0][i],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "filename": results["metadatas"][0][i]["filename"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "score": round(similarity, 4),
            })

    return retrieved
