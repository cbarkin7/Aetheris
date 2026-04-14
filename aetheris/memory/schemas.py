"""Memory domain schemas."""
from datetime import datetime
from pydantic import BaseModel, Field


class UserMemoryEntry(BaseModel):
    user_id: str
    key: str
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserMemory(BaseModel):
    user_id: str
    preferences: dict[str, str] = Field(default_factory=dict)
