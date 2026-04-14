"""Unit tests for RAG ingestion pipeline."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetheris.rag.ingest import chunk_documents, load_documents
from langchain_core.documents import Document


def test_load_documents_txt(sample_txt_file):
    docs = load_documents(sample_txt_file)
    assert len(docs) >= 1
    assert "AETHERIS" in docs[0].page_content
    assert docs[0].metadata["filename"] == "sample.txt"
    assert "document_id" in docs[0].metadata


def test_load_documents_md(sample_md_file):
    docs = load_documents(sample_md_file)
    assert len(docs) >= 1
    assert "AETHERIS" in docs[0].page_content


def test_load_documents_file_not_found():
    from aetheris.rag.ingest import load_documents
    with pytest.raises(FileNotFoundError):
        load_documents(Path("/nonexistent/file.txt"))


def test_load_documents_unsupported_extension(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents(f)


def test_chunk_documents_respects_size():
    doc = Document(page_content="word " * 500, metadata={"source": "test"})
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200  # some tolerance for splitter


def test_chunk_documents_preserves_metadata(sample_txt_file):
    docs = load_documents(sample_txt_file)
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
    for chunk in chunks:
        assert "document_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_chunk_documents_adds_chunk_index(sample_txt_file):
    docs = load_documents(sample_txt_file)
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=5)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
