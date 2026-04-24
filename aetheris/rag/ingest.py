"""
RAG ingestion pipeline.
Loads documents from disk, chunks them, embeds and stores in Chroma.
"""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from aetheris.config import get_settings
from aetheris.rag.schemas import IngestResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _document_id(file_path: Path) -> str:
    """Stable document ID based on file path."""
    return hashlib.md5(str(file_path.resolve()).encode()).hexdigest()


def document_id_for_path(file_path: Path | str) -> str:
    """Public helper: returns the stable document ID for a given path."""
    return _document_id(Path(file_path))


def load_documents(source: Path | str) -> list[Document]:
    """Load a single file into LangChain Document objects."""
    path = Path(source)
    logger.info("[RAG][INGEST] → load_documents | inicio | fichero='%s'", path.name)

    if not path.exists():
        logger.error("[RAG][INGEST] → load_documents | fichero no encontrado | path='%s'", path)
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("[RAG][INGEST] → load_documents | extensión no soportada | ext='%s'", ext)
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
        loader_name = "PyMuPDFLoader"
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
        loader_name = "Docx2txtLoader"
    else:
        loader = TextLoader(str(path), encoding="utf-8")
        loader_name = "TextLoader"

    logger.debug("[RAG][INGEST] → load_documents | usando loader='%s'", loader_name)
    docs = loader.load()
    doc_id = _document_id(path)
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["document_id"] = doc_id
        doc.metadata["filename"] = path.name

    logger.info(
        "[RAG][INGEST] → load_documents | completado | fichero='%s' páginas=%d doc_id='%s'",
        path.name, len(docs), doc_id[:8],
    )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks."""
    logger.info(
        "[RAG][INGEST] → chunk_documents | inicio | docs=%d chunk_size=%d overlap=%d",
        len(docs), chunk_size, chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info("[RAG][INGEST] → chunk_documents | completado | chunks=%d", len(chunks))
    return chunks


def _get_chroma(collection_name: str, embeddings: Any) -> Chroma:
    settings = get_settings()
    persist_dir = str(settings.chroma_persist_path)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(
    chunks: list[Document],
    collection_name: str = "aetheris",
    embeddings: Any = None,
) -> Chroma:
    """Embed chunks and upsert into Chroma. Returns the Chroma instance."""
    logger.info(
        "[RAG][INGEST] → embed_and_store | inicio | chunks=%d colección='%s'",
        len(chunks), collection_name,
    )
    settings = get_settings()
    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        logger.debug("[RAG][INGEST] → embed_and_store | modelo='%s'", settings.embedding_model)

    vectorstore = _get_chroma(collection_name, embeddings)

    # Stamp ingestion timestamp on every chunk so it can be queried later
    now_iso = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata["ingested_at"] = now_iso

    # Build stable IDs per chunk to allow safe re-ingestion (idempotent)
    ids = [
        f"{chunk.metadata['document_id']}_{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]
    vectorstore.add_documents(chunks, ids=ids)
    logger.info(
        "[RAG][INGEST] → embed_and_store | completado | chunks=%d colección='%s'",
        len(chunks), collection_name,
    )
    return vectorstore


def ingest_file(
    file_path: Path | str,
    collection_name: str = "aetheris",
    embeddings: Any = None,
) -> IngestResult:
    """Full pipeline: load → chunk → embed → store. Returns IngestResult."""
    path = Path(file_path)
    logger.info("[RAG][INGEST] → ingest_file | inicio | fichero='%s' colección='%s'", path.name, collection_name)

    settings = get_settings()
    docs = load_documents(path)
    chunks = chunk_documents(
        docs,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    embed_and_store(chunks, collection_name=collection_name, embeddings=embeddings)

    doc_id = _document_id(path)
    logger.info(
        "[RAG][INGEST] → ingest_file | completado | fichero='%s' doc_id='%s' chunks=%d",
        path.name, doc_id[:8], len(chunks),
    )
    return IngestResult(
        source_path=str(path),
        n_chunks=len(chunks),
        collection_name=collection_name,
        document_id=doc_id,
    )
