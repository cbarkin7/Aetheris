"""Document ingestion and management endpoints."""
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from aetheris.api.schemas import DocumentConflictDetail, DocumentSchema, IngestResultSchema
from aetheris.config import Settings
from aetheris.api.dependencies import get_app_settings
from aetheris.rag.ingest import SUPPORTED_EXTENSIONS, document_id_for_path, ingest_file
from aetheris.rag.retriever import delete_document, get_document_info, list_documents

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=IngestResultSchema, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    force: bool = Form(default=False),
    settings: Settings = Depends(get_app_settings),
) -> IngestResultSchema:
    """
    Sube e ingesta un documento en la base de conocimiento RAG.

    - Si el documento ya existe y `force=False` devuelve 409 con información del conflicto.
    - Si el documento ya existe y `force=True` elimina los fragmentos obsoletos y reingesta.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    uploads_dir = settings.uploads_path
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / (file.filename or "upload")

    # Compute stable document_id before touching disk (based on destination path)
    doc_id = document_id_for_path(dest)
    existing = get_document_info(doc_id, collection_name="aetheris")

    if existing and not force:
        # Parse ingested_at safely
        ingested_at_dt: datetime | None = None
        if existing.get("ingested_at"):
            try:
                ingested_at_dt = datetime.fromisoformat(existing["ingested_at"])
            except ValueError:
                pass

        conflict = DocumentConflictDetail(
            document_id=doc_id,
            filename=existing["filename"],
            ingested_at=ingested_at_dt,
            n_chunks=existing["n_chunks"],
        )
        raise HTTPException(status_code=409, detail=conflict.model_dump(mode="json"))

    # Persist file to disk
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # If re-ingesting, remove stale chunks first
    if existing and force:
        deleted = delete_document(doc_id, collection_name="aetheris")
        import logging
        logging.getLogger(__name__).info(
            "[RAG] Reingestando '%s': eliminados %d fragmentos obsoletos", file.filename, deleted
        )

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
    result = []
    for d in docs:
        ingested_at_dt: datetime | None = None
        if d.get("ingested_at"):
            try:
                ingested_at_dt = datetime.fromisoformat(d["ingested_at"])
            except ValueError:
                pass
        result.append(DocumentSchema(
            document_id=d["document_id"],
            filename=d["filename"],
            source=d["source"],
            ingested_at=ingested_at_dt,
            n_chunks=d.get("n_chunks"),
        ))
    return result


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: str) -> None:
    deleted = delete_document(document_id, collection_name="aetheris")
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
