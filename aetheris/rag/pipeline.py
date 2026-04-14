"""
End-to-end RAG chain (for standalone use outside the agent graph).
The agent graph uses retrieve() directly and feeds context to the llm_node.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are AETHERIS, an intelligent personal assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say so clearly.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')} | Page: {d.metadata.get('page', '-')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(retriever: VectorStoreRetriever, llm) -> Runnable:
    """
    Build a streaming-compatible RAG chain.

    Usage:
        chain = build_rag_chain(retriever, llm)
        response = chain.invoke({"question": "What is X?"})
    """
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
