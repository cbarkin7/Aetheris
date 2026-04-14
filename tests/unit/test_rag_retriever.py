"""Unit tests for RAG retriever (mocked Chroma)."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aetheris.rag.schemas import RetrievalResult


def test_retrieve_filters_below_threshold():
    mock_results = [
        (Document(page_content="relevant", metadata={"source": "a", "document_id": "d1", "chunk_index": 0}), 0.9),
        (Document(page_content="irrelevant", metadata={"source": "b", "document_id": "d2", "chunk_index": 0}), 0.1),
    ]

    with patch("aetheris.rag.retriever.get_vectorstore") as mock_vs:
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = mock_results
        mock_vs.return_value = mock_store

        from aetheris.rag.retriever import retrieve
        results = retrieve("test query")

    assert len(results) == 1
    assert results[0].content == "relevant"
    assert results[0].score == 0.9


def test_retrieve_returns_retrieval_result_objects():
    mock_results = [
        (Document(
            page_content="test content",
            metadata={"source": "doc.pdf", "document_id": "abc", "chunk_index": 2, "page": 3}
        ), 0.8),
    ]

    with patch("aetheris.rag.retriever.get_vectorstore") as mock_vs:
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = mock_results
        mock_vs.return_value = mock_store

        from aetheris.rag.retriever import retrieve
        results = retrieve("query")

    assert isinstance(results[0], RetrievalResult)
    assert results[0].source == "doc.pdf"
    assert results[0].page == 3
    assert results[0].chunk_index == 2


def test_retrieve_empty_when_all_below_threshold():
    mock_results = [
        (Document(page_content="low score", metadata={"source": "x", "document_id": "y", "chunk_index": 0}), 0.05),
    ]

    with patch("aetheris.rag.retriever.get_vectorstore") as mock_vs:
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = mock_results
        mock_vs.return_value = mock_store

        from aetheris.rag.retriever import retrieve
        results = retrieve("query")

    assert results == []
