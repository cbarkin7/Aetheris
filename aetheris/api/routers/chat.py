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
    thread_id = config.get("configurable", {}).get("thread_id", "?")
    tokens_emitidos = 0
    nodo_actual = ""

    logger.debug("[API][CHAT] → _stream_graph | iniciando stream | thread='%s'", thread_id)
    try:
        async for event in graph.astream_events(input_data, config=config, version="v2"):
            kind = event.get("event", "")
            event_node = event.get("metadata", {}).get("langgraph_node", "")

            # Trazar cambios de nodo (sin repetir el mismo nodo)
            if event_node and event_node != nodo_actual:
                nodo_actual = event_node
                logger.info("[API][CHAT] → _stream_graph | nodo='%s' | thread='%s'", nodo_actual, thread_id)

            if kind == "on_chat_model_stream":
                # Solo emitir tokens del llm_node (no del manager_node ni otros)
                if event_node not in ("llm_node",):
                    continue
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    tokens_emitidos += 1
                    data = json.dumps({"type": "token", "content": chunk.content})
                    yield f"data: {data}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "hitl_node":
                state = event["data"].get("output", {})
                pending = state.get("tool_calls_pending", [])
                if pending:
                    acciones = [p["name"] for p in pending]
                    logger.info(
                        "[API][CHAT] → _stream_graph | HITL requerido | thread='%s' acciones=%s",
                        thread_id, acciones,
                    )
                    data = json.dumps({"type": "hitl_required", "actions": pending})
                    yield f"data: {data}\n\n"
                    return

            elif kind == "on_chain_end" and event.get("name") == "input_guardrail_node":
                state = event["data"].get("output", {})
                if state.get("guardrail_passed") is False:
                    violations = state.get("guardrail_violations", [])
                    logger.warning(
                        "[API][CHAT] → _stream_graph | guardrail bloqueó mensaje | thread='%s' violaciones=%s",
                        thread_id, violations,
                    )
                    data = json.dumps({
                        "type": "guardrail_blocked",
                        "violations": violations,
                    })
                    yield f"data: {data}\n\n"

        logger.info(
            "[API][CHAT] → _stream_graph | completado | thread='%s' tokens=%d",
            thread_id, tokens_emitidos,
        )
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        logger.error("[API][CHAT] → _stream_graph | ERROR | thread='%s' error=%s", thread_id, exc, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


@router.post("")
async def chat(
    body: ChatRequest,
    graph=Depends(get_compiled_graph),
) -> StreamingResponse:
    """Iniciar o continuar una sesión de chat. Devuelve un stream SSE."""
    logger.info(
        "[API][CHAT] → POST /chat | recibida | thread='%s' user='%s' msg_len=%d",
        body.thread_id, body.user_id, len(body.message),
    )

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
    logger.info(
        "[API][CHAT] → POST /chat/%s/resume | user='%s' approved=%s",
        thread_id, body.user_id, body.approved,
    )

    from aetheris.observability.tracing import get_langsmith_callbacks
    callbacks = get_langsmith_callbacks()

    config = _build_config(thread_id, body.user_id, callbacks)
    await graph.aupdate_state(config, {"hitl_approved": body.approved})
    logger.info("[API][CHAT] → aupdate_state | completado | thread='%s' hitl_approved=%s", thread_id, body.approved)

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
    logger.info("[API][CHAT] → GET /chat/%s/history | user='%s'", thread_id, user_id)
    config = _build_config(thread_id, user_id)
    try:
        state = await graph.aget_state(config)
        raw_messages = state.values.get("messages", []) if state else []
    except Exception as exc:
        logger.warning("[API][CHAT] → get_history | error al obtener estado | %s", exc)
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

    logger.info("[API][CHAT] → get_history | completado | thread='%s' mensajes=%d", thread_id, len(messages))
    return ChatHistoryResponse(thread_id=thread_id, messages=messages)
