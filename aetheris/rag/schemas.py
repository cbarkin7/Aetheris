"""RAG domain schemas."""
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    source_path: str
    n_chunks: int
    collection_name: str
    document_id: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievalResult(BaseModel):
    content: str
    source: str
    page: int | None = None
    score: float
    document_id: str
    chunk_index: int
