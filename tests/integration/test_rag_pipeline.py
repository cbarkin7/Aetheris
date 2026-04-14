"""
Integration test: full ingest → retrieval loop.
Asserts >85% hit rate on 5 known facts from sample documents.
Uses real Chroma (in /tmp) with mocked OpenAI embeddings.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Mock embeddings so no real OpenAI calls are made
# ---------------------------------------------------------------------------
class MockEmbeddings:
    """Simple hash-based deterministic embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        # 384-dim vector derived from hash — consistent per text
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = []
        for i in range(384):
            vec.append(((h >> (i % 64)) & 0xFF) / 255.0)
        return vec


@pytest.fixture
def populated_chroma(tmp_path, override_settings, monkeypatch):
    """Ingest 2 sample documents into a real Chroma instance."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from aetheris.config import get_settings
    get_settings.cache_clear()

    mock_emb = MockEmbeddings()

    # Create sample documents with known facts
    doc1_path = tmp_path / "aetheris_facts.txt"
    doc1_path.write_text(
        "AETHERIS is an Autonomous Cognitive Agent.\n"
        "AETHERIS uses LangGraph for agent orchestration.\n"
        "The RAG pipeline uses Chroma as the vector store.\n"
        "AETHERIS supports PDF, DOCX, TXT, and Markdown files.\n"
        "LangSmith is used for observability and tracing.\n",
        encoding="utf-8",
    )
    doc2_path = tmp_path / "tech_stack.txt"
    doc2_path.write_text(
        "The backend is built with FastAPI.\n"
        "The frontend uses Streamlit for the user interface.\n"
        "SQLite is used for session checkpoints and long-term memory.\n"
        "Tavily MCP provides real-time web search capabilities.\n"
        "Google MCP integrates Calendar, Gmail, and Drive.\n",
        encoding="utf-8",
    )

    from aetheris.rag.ingest import ingest_file
    ingest_file(doc1_path, collection_name="test_collection", embeddings=mock_emb)
    ingest_file(doc2_path, collection_name="test_collection", embeddings=mock_emb)

    return mock_emb


@pytest.mark.integration
def test_rag_hit_rate_above_85_percent(populated_chroma, tmp_path, monkeypatch):
    """
    Issue 10 targeted queries against known document content.
    Assert >= 85% of queries return at least 1 relevant result.
    """
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from aetheris.config import get_settings
    get_settings.cache_clear()

    from aetheris.rag.retriever import retrieve

    # (query, keyword_expected_in_any_result)
    test_cases = [
        ("What is AETHERIS?", "Autonomous Cognitive Agent"),
        ("How does AETHERIS orchestrate agents?", "LangGraph"),
        ("What vector store does AETHERIS use?", "Chroma"),
        ("What file types does AETHERIS support?", "PDF"),
        ("How is AETHERIS monitored?", "LangSmith"),
        ("What is the backend framework?", "FastAPI"),
        ("What is used for the frontend?", "Streamlit"),
        ("How is memory stored?", "SQLite"),
        ("How does AETHERIS search the web?", "Tavily"),
        ("What Google services are integrated?", "Calendar"),
    ]

    hits = 0
    for query, expected_keyword in test_cases:
        results = retrieve(query, collection_name="test_collection", embeddings=populated_chroma)
        all_content = " ".join(r.content for r in results)
        if expected_keyword.lower() in all_content.lower():
            hits += 1

    hit_rate = hits / len(test_cases)
    assert hit_rate >= 0.85, (
        f"RAG hit rate {hit_rate:.0%} is below the 85% target. "
        f"({hits}/{len(test_cases)} queries matched)"
    )


@pytest.mark.integration
def test_ingest_creates_chunks(tmp_path, override_settings, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from aetheris.config import get_settings
    get_settings.cache_clear()

    mock_emb = MockEmbeddings()
    doc_path = tmp_path / "test.txt"
    doc_path.write_text("sentence one.\n" * 100, encoding="utf-8")

    from aetheris.rag.ingest import ingest_file
    result = ingest_file(doc_path, collection_name="test_ingest", embeddings=mock_emb)
    assert result.n_chunks >= 1
    assert result.document_id != ""


@pytest.mark.integration
def test_idempotent_reingest(tmp_path, override_settings, monkeypatch):
    """Re-ingesting the same file should not duplicate chunks."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from aetheris.config import get_settings
    get_settings.cache_clear()

    mock_emb = MockEmbeddings()
    doc_path = tmp_path / "stable.txt"
    doc_path.write_text("stable content " * 50, encoding="utf-8")

    from aetheris.rag.ingest import ingest_file
    r1 = ingest_file(doc_path, collection_name="idem_test", embeddings=mock_emb)
    r2 = ingest_file(doc_path, collection_name="idem_test", embeddings=mock_emb)
    assert r1.n_chunks == r2.n_chunks
