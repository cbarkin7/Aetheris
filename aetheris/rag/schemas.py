"""RAG domain schemas."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    source_path: str
    n_chunks: int
    collection_name: str
    document_id: str
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class RetrievalResult(BaseModel):
    content: str
    source: str
    page: int | None = None
    score: float
    document_id: str
    chunk_index: int
