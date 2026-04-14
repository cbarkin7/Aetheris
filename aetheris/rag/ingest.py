"""
RAG ingestion pipeline.
Loads documents from disk, chunks them, embeds and stores in Chroma.
"""
import hashlib
import logging
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


def load_documents(source: Path | str) -> list[Document]:
    """Load a single file into LangChain Document objects."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    doc_id = _document_id(path)
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["document_id"] = doc_id
        doc.metadata["filename"] = path.name

    logger.info("Loaded %d pages from '%s'", len(docs), path.name)
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    logger.info("Split into %d chunks", len(chunks))
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
    settings = get_settings()
    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    vectorstore = _get_chroma(collection_name, embeddings)

    # Build stable IDs per chunk to allow safe re-ingestion (idempotent)
    ids = [
        f"{chunk.metadata['document_id']}_{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]
    vectorstore.add_documents(chunks, ids=ids)
    logger.info("Stored %d chunks in collection '%s'", len(chunks), collection_name)
    return vectorstore


def ingest_file(
    file_path: Path | str,
    collection_name: str = "aetheris",
    embeddings: Any = None,
) -> IngestResult:
    """Full pipeline: load → chunk → embed → store. Returns IngestResult."""
    settings = get_settings()
    path = Path(file_path)

    docs = load_documents(path)
    chunks = chunk_documents(
        docs,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    embed_and_store(chunks, collection_name=collection_name, embeddings=embeddings)

    doc_id = _document_id(path)
    return IngestResult(
        source_path=str(path),
        n_chunks=len(chunks),
        collection_name=collection_name,
        document_id=doc_id,
    )
