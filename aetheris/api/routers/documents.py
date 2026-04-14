"""Document ingestion and management endpoints."""
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from aetheris.api.schemas import DocumentSchema, IngestResultSchema
from aetheris.config import Settings
from aetheris.api.dependencies import get_app_settings
from aetheris.rag.ingest import SUPPORTED_EXTENSIONS, ingest_file
from aetheris.rag.retriever import delete_document, list_documents

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=IngestResultSchema, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    settings: Settings = Depends(get_app_settings),
) -> IngestResultSchema:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    uploads_dir = settings.uploads_path
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / (file.filename or "upload")

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(dest, collection_name="aetheris")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return IngestResultSchema(
        document_id=result.document_id,
        filename=Path(result.source_path).name,
        n_chunks=result.n_chunks,
        collection_name=result.collection_name,
        ingested_at=result.ingested_at,
    )


@router.get("", response_model=list[DocumentSchema])
def list_user_documents(
    user_id: str = "default",
) -> list[DocumentSchema]:
    docs = list_documents(collection_name="aetheris")
    return [DocumentSchema(**d) for d in docs]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: str) -> None:
    deleted = delete_document(document_id, collection_name="aetheris")
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
