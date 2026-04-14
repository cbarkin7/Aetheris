"""FastAPI request/response Pydantic models."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    thread_id: str = Field(..., description="Session/thread identifier")
    user_id: str = Field(default="default", description="User identifier")
    stream: bool = Field(default=True)


class HITLResumeRequest(BaseModel):
    approved: bool
    user_id: str = Field(default="default")


class MessageSchema(BaseModel):
    role: Literal["human", "ai", "system", "tool"]
    content: str
    timestamp: datetime | None = None


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageSchema]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentSchema(BaseModel):
    document_id: str
    filename: str
    source: str


class IngestResultSchema(BaseModel):
    document_id: str
    filename: str
    n_chunks: int
    collection_name: str
    ingested_at: datetime


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryResponse(BaseModel):
    user_id: str
    preferences: dict[str, str]


class MemoryUpdateRequest(BaseModel):
    preferences: dict[str, str]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    chroma_ok: bool
    sqlite_ok: bool
    app_env: str


class LangSmithHealthResponse(BaseModel):
    langsmith_connected: bool
    project_name: str
    error: str | None = None
