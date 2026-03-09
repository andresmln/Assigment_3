"""
Document ingestion — parse PDF/DOCX files and upload to MinIO.
"""
import io
import pdfplumber
from docx import Document as DocxDocument
from minio import Minio
from config import Config


def get_minio_client():
    """Create and return a MinIO client."""
    return Minio(
        Config.MINIO_ENDPOINT,
        access_key=Config.MINIO_ACCESS_KEY,
        secret_key=Config.MINIO_SECRET_KEY,
        secure=Config.MINIO_SECURE,
    )


def ensure_bucket(client: Minio):
    """Create the documents bucket if it doesn't exist."""
    if not client.bucket_exists(Config.MINIO_BUCKET):
        client.make_bucket(Config.MINIO_BUCKET)


def upload_to_minio(client: Minio, file_bytes: bytes, object_name: str, content_type: str):
    """Upload raw file bytes to MinIO."""
    ensure_bucket(client)
    data = io.BytesIO(file_bytes)
    client.put_object(
        Config.MINIO_BUCKET,
        object_name,
        data,
        length=len(file_bytes),
        content_type=content_type,
    )


def delete_from_minio(client: Minio, object_name: str):
    """Delete a file from MinIO."""
    client.remove_object(Config.MINIO_BUCKET, object_name)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    Raises ValueError if the PDF appears to be scanned (image-only).
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n\n".join(text_parts)

    # Scanned PDF detection: if extracted text is too short, it's likely image-only
    if len(full_text.strip()) < 50:
        raise ValueError(
            "Scanned or image-only PDF detected. No extractable text found. "
            "Please upload a text-based PDF or use OCR preprocessing."
        )

    return full_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    if len(full_text.strip()) < 10:
        raise ValueError("DOCX file appears to be empty or contains no extractable text.")

    return full_text


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Parse a document based on its file extension.
    Returns the extracted text.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Only PDF and DOCX are supported.")
