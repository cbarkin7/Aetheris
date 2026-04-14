"""
Endpoints de chat con streaming SSE y reanudación de Human-in-the-Loop.
"""
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from aetheris.api.dependencies import get_compiled_graph
from aetheris.api.schemas import ChatHistoryResponse, ChatRequest, HITLResumeRequest, MessageSchema

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _build_config(thread_id: str, user_id: str, callbacks: list | None = None) -> dict:
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg


async def _stream_graph(graph, input_data: dict | None, config: dict) -> AsyncGenerator[str, None]:
    """Generador asíncrono que emite eventos SSE desde el grafo."""
    try:
        async for event in graph.astream_events(input_data, config=config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                # Only stream tokens from the final llm_node, not from internal
                # nodes like manager_node that also call the LLM.
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name not in ("llm_node",):
                    continue
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    data = json.dumps({"type": "token", "content": chunk.content})
                    yield f"data: {data}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "hitl_node":
                state = event["data"].get("output", {})
                pending = state.get("tool_calls_pending", [])
                if pending:
                    data = json.dumps({"type": "hitl_required", "actions": pending})
                    yield f"data: {data}\n\n"
                    return

            elif kind == "on_chain_end" and event.get("name") == "input_guardrail_node":
                state = event["data"].get("output", {})
                if state.get("guardrail_passed") is False:
                    violations = state.get("guardrail_violations", [])
                    data = json.dumps({
                        "type": "guardrail_blocked",
                        "violations": violations,
                    })
                    yield f"data: {data}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        logger.error("Error en stream: %s", exc, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


@router.post("")
async def chat(
    body: ChatRequest,
    graph=Depends(get_compiled_graph),
) -> StreamingResponse:
    """Iniciar o continuar una sesión de chat. Devuelve un stream SSE."""
    from aetheris.observability.tracing import get_langsmith_callbacks
    callbacks = get_langsmith_callbacks()

    config = _build_config(body.thread_id, body.user_id, callbacks)
    input_data = {
        "messages": [HumanMessage(content=body.message)],
        "thread_id": body.thread_id,
        "user_id": body.user_id,
        "rag_context": [],
        "tool_calls_pending": [],
        "hitl_approved": None,
        "user_memory": {},
        "guardrail_passed": None,
        "guardrail_violations": [],
        "llm_provider": "",
        "error": None,
        "intent": "unknown",
    }

    return StreamingResponse(
        _stream_graph(graph, input_data, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{thread_id}/resume")
async def resume_after_hitl(
    thread_id: str,
    body: HITLResumeRequest,
    graph=Depends(get_compiled_graph),
) -> StreamingResponse:
    """Reanudar un grafo interrumpido tras la aprobación/rechazo HITL."""
    from aetheris.observability.tracing import get_langsmith_callbacks
    callbacks = get_langsmith_callbacks()

    config = _build_config(thread_id, body.user_id, callbacks)
    await graph.aupdate_state(config, {"hitl_approved": body.approved})

    return StreamingResponse(
        _stream_graph(graph, None, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{thread_id}/history", response_model=ChatHistoryResponse)
async def get_history(
    thread_id: str,
    user_id: str = "default",
    graph=Depends(get_compiled_graph),
) -> ChatHistoryResponse:
    """Recuperar el historial de conversación desde el checkpoint de LangGraph."""
    config = _build_config(thread_id, user_id)
    try:
        state = await graph.aget_state(config)
        raw_messages = state.values.get("messages", []) if state else []
    except Exception:
        raw_messages = []

    messages = []
    for m in raw_messages:
        if isinstance(m, HumanMessage):
            role = "human"
        elif isinstance(m, AIMessage):
            role = "ai"
        else:
            role = "system"
        content = m.content if isinstance(m.content, str) else str(m.content)
        messages.append(MessageSchema(role=role, content=content))

    return ChatHistoryResponse(thread_id=thread_id, messages=messages)
