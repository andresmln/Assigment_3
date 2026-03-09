"""
Chunking strategies — split document text into retrievable units.
"""
import re
from typing import List, Dict
from config import Config


def fixed_size_chunks(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[Dict]:
    """
    Fixed-size chunking with sliding window.
    Splits text into chunks of approximately `chunk_size` words with `overlap` word overlap.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP

    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "strategy": "fixed_size",
            "start_word": start,
            "end_word": min(end, len(words)),
        })
        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def recursive_split(
    text: str,
    doc_id: str,
    filename: str,
    max_chunk_size: int = None,
) -> List[Dict]:
    """
    Recursive splitting — tries to split by paragraph, then sentence, then word.
    Preserves author boundaries as much as possible.
    """
    if max_chunk_size is None:
        max_chunk_size = Config.CHUNK_SIZE

    # Define split hierarchy
    separators = ["\n\n", "\n", ". ", " "]

    def _split_recursive(text_block: str, sep_idx: int) -> List[str]:
        """Recursively split text using separators from coarsest to finest."""
        if len(text_block.split()) <= max_chunk_size:
            return [text_block] if text_block.strip() else []

        if sep_idx >= len(separators):
            # Last resort: just do word-level split
            words = text_block.split()
            result = []
            for i in range(0, len(words), max_chunk_size):
                result.append(" ".join(words[i:i + max_chunk_size]))
            return result

        sep = separators[sep_idx]
        parts = text_block.split(sep)

        result = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate.split()) <= max_chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If a single part is too large, recurse with the next separator
                if len(part.split()) > max_chunk_size:
                    result.extend(_split_recursive(part, sep_idx + 1))
                else:
                    current = part
                    continue
                current = ""

        if current:
            result.append(current)

        return result

    raw_chunks = _split_recursive(text, 0)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx,
                "text": chunk_text.strip(),
                "strategy": "recursive",
            })

    return chunks


def chunk_document(
    text: str,
    doc_id: str,
    filename: str,
    strategy: str = "fixed_size",
) -> List[Dict]:
    """
    Chunk a document using the specified strategy.
    Available strategies: 'fixed_size', 'recursive'
    """
    if strategy == "recursive":
        return recursive_split(text, doc_id, filename)
    else:
        return fixed_size_chunks(text, doc_id, filename)
