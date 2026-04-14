"""User memory read/write endpoints."""
from fastapi import APIRouter

from aetheris.api.schemas import MemoryResponse, MemoryUpdateRequest
from aetheris.memory.long_term import load_user_memory, upsert_user_memory

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.get("/{user_id}", response_model=MemoryResponse)
def get_memory(user_id: str) -> MemoryResponse:
    prefs = load_user_memory(user_id)
    return MemoryResponse(user_id=user_id, preferences=prefs)


@router.put("/{user_id}", response_model=MemoryResponse)
def update_memory(user_id: str, body: MemoryUpdateRequest) -> MemoryResponse:
    upsert_user_memory(user_id, body.preferences)
    updated = load_user_memory(user_id)
    return MemoryResponse(user_id=user_id, preferences=updated)
