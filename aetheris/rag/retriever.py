"""
RAG retrieval layer.
Wraps Chroma with MMR search and score-threshold filtering.
"""
import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from aetheris.config import get_settings
from aetheris.rag.schemas import RetrievalResult

logger = logging.getLogger(__name__)


def _get_embeddings(settings=None) -> OpenAIEmbeddings:
    s = settings or get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        openai_api_key=s.openai_api_key,
    )


def get_vectorstore(
    collection_name: str = "aetheris",
    embeddings: Any = None,
) -> Chroma:
    settings = get_settings()
    emb = embeddings or _get_embeddings(settings)
    return Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=str(settings.chroma_persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever(
    collection_name: str = "aetheris",
    k: int | None = None,
    embeddings: Any = None,
) -> VectorStoreRetriever:
    """Return a Chroma retriever using MMR search."""
    settings = get_settings()
    effective_k = k or settings.rag_retrieval_k
    vectorstore = get_vectorstore(collection_name=collection_name, embeddings=embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": effective_k, "fetch_k": effective_k * 3},
    )


def retrieve(
    query: str,
    collection_name: str = "aetheris",
    k: int | None = None,
    embeddings: Any = None,
) -> list[RetrievalResult]:
    """
    Retrieve the top-k most relevant chunks for *query*.
    Filters out chunks with similarity score below settings.rag_score_threshold.
    """
    settings = get_settings()
    effective_k = k or settings.rag_retrieval_k
    vectorstore = get_vectorstore(collection_name=collection_name, embeddings=embeddings)

    results = vectorstore.similarity_search_with_relevance_scores(query, k=effective_k)

    filtered = [
        RetrievalResult(
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
            page=doc.metadata.get("page"),
            score=score,
            document_id=doc.metadata.get("document_id", ""),
            chunk_index=doc.metadata.get("chunk_index", 0),
        )
        for doc, score in results
        if score >= settings.rag_score_threshold
    ]

    logger.info(
        "Retrieved %d/%d chunks above threshold %.2f for query: '%s...'",
        len(filtered),
        len(results),
        settings.rag_score_threshold,
        query[:60],
    )
    return filtered


def list_documents(collection_name: str = "aetheris", embeddings: Any = None) -> list[dict]:
    """Return a deduplicated list of ingested documents from Chroma metadata."""
    vectorstore = get_vectorstore(collection_name=collection_name, embeddings=embeddings)
    collection = vectorstore._collection
    all_meta = collection.get(include=["metadatas"])["metadatas"]

    seen: dict[str, dict] = {}
    for meta in all_meta:
        doc_id = meta.get("document_id", "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "filename": meta.get("filename", ""),
                "source": meta.get("source", ""),
            }
    return list(seen.values())


def delete_document(document_id: str, collection_name: str = "aetheris", embeddings: Any = None) -> int:
    """Delete all chunks for a document_id. Returns number of chunks deleted."""
    vectorstore = get_vectorstore(collection_name=collection_name, embeddings=embeddings)
    collection = vectorstore._collection
    results = collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    logger.info("Deleted %d chunks for document_id='%s'", len(ids), document_id)
    return len(ids)
