"""
LLM client — communicates with the llama.cpp server to generate answers.
Uses the OpenAI-compatible /v1/chat/completions endpoint.
"""
import requests
from typing import List, Dict
from config import Config


def build_messages(question: str, sources: List[Dict]) -> List[Dict]:
    """
    Build chat messages for the LLM with source citations.
    """
    system_msg = (
        "You are a helpful document assistant. Answer the question using ONLY the "
        "provided context below. Cite your sources using [Source N] notation.\n"
        "If the context does not contain enough information to answer the question, "
        'say "I cannot fully answer this based on the provided documents" and explain '
        "what information is missing."
    )

    if not sources:
        user_msg = (
            f"Question: {question}\n\n"
            "No relevant documents were found in the knowledge base. "
            "Please inform the user that they need to upload relevant documents first."
        )
    else:
        context_parts = []
        # Truncate each chunk to ~250 words to fit within 2048 token context
        max_words_per_chunk = 250
        for i, source in enumerate(sources, 1):
            text = source['text']
            words = text.split()
            if len(words) > max_words_per_chunk:
                text = " ".join(words[:max_words_per_chunk]) + "..."
            context_parts.append(
                f"[Source {i}] (File: {source['filename']}, "
                f"Chunk: {source['chunk_index']}):\n"
                f"\"{text}\""
            )
        context = "\n\n".join(context_parts)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_answer(question: str, sources: List[Dict]) -> Dict:
    """
    Generate an answer using the local LLM via OpenAI-compatible API.
    Returns a dict with 'answer' and 'sources'.
    """
    messages = build_messages(question, sources)

    try:
        response = requests.post(
            f"{Config.llm_url()}/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.3,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        answer_text = result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        answer_text = (
            "LLM service is not available. Please ensure the llama.cpp server "
            "is running and try again."
        )
    except requests.exceptions.Timeout:
        answer_text = "LLM request timed out. The model may be overloaded."
    except Exception as e:
        answer_text = f"Error generating answer: {str(e)}"

    return {
        "answer": answer_text,
        "sources": [
            {
                "doc_id": s["doc_id"],
                "filename": s["filename"],
                "chunk_index": s["chunk_index"],
                "score": s["score"],
                "text": s["text"][:500],
            }
            for s in sources
        ],
    }


def check_llm_health() -> bool:
    """Check if the LLM server is reachable."""
    try:
        response = requests.get(f"{Config.llm_url()}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
