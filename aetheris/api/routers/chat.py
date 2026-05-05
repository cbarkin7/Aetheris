"""
Endpoints de chat con streaming SSE y reanudación de Human-in-the-Loop.
"""
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from aetheris.api.dependencies import get_compiled_graph
from aetheris.api.schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ConversationListResponse,
    ConversationSummary,
    DeleteConversationResponse,
    HITLResumeRequest,
    MessageSchema,
)
from aetheris.memory.long_term import delete_conversation, list_conversations, save_conversation

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

            elif kind == "on_chain_end" and event.get("name") == "google_planner_node":
                output = event["data"].get("output", {})

                if output.get("data_collection_required"):
                    # PASO 0 (tarea completa) o datos faltantes: google_planner_node
                    # generó texto en lugar de tool_calls y llm_node hará pass-through.
                    # Emitir el contenido como tokens SSE para que el frontend lo muestre.
                    for msg in output.get("messages", []):
                        content = getattr(msg, "content", None) or ""
                        if isinstance(content, str) and content:
                            tokens_emitidos += 1
                            data = json.dumps({"type": "token", "content": content})
                            yield f"data: {data}\n\n"
                    # No hacer return: el grafo continúa normalmente hasta END.

            elif kind == "on_chain_end" and event.get("name") == "hitl_node":
                output = event["data"].get("output", {})
                pending = output.get("tool_calls_pending", [])

                if pending and any(p.get("requires_approval") for p in pending):
                    # Hay al menos una acción destructiva — mostrar modal HITL.
                    # El grafo está pausado en hitl_wait_node, por lo que hacemos
                    # return para detener el stream SSE hasta que el usuario decida.
                    acciones = [p["name"] for p in pending]
                    logger.info(
                        "[API][CHAT] → _stream_graph | HITL requerido | thread='%s' acciones=%s",
                        thread_id, acciones,
                    )
                    data = json.dumps({"type": "hitl_required", "actions": pending})
                    yield f"data: {data}\n\n"
                    return

                # Caso restante: lecturas auto-aprobadas (hitl_approved=True) o cola vacía.
                # El grafo continúa directamente a google_action_node sin interrupción.

            elif kind == "on_chain_end" and event.get("name") == "google_action_node":
                # Emitir feedback inmediato por cada acción ejecutada (antes del resumen LLM)
                output = event["data"].get("output", {})
                for result in output.get("action_results", []):
                    if result.get("ok"):
                        data = json.dumps({
                            "type": "action_result",
                            "name": result["name"],
                            "summary": result.get("summary", ""),
                        })
                    else:
                        data = json.dumps({
                            "type": "action_error",
                            "name": result["name"],
                            "error": result.get("error", "Error desconocido"),
                        })
                    yield f"data: {data}\n\n"

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
    # Generar thread_id si el cliente no lo envía (nueva conversación)
    thread_id = body.thread_id or str(uuid.uuid4())

    logger.info(
        "[API][CHAT] → POST /chat | recibida | thread='%s' user='%s' msg_len=%d",
        thread_id, body.user_id, len(body.message),
    )

    from aetheris.observability.tracing import get_langsmith_callbacks
    callbacks = get_langsmith_callbacks()

    # Persistir conversación en el historial (título = primeros 80 chars del mensaje)
    title = body.message[:80].strip()
    try:
        save_conversation(thread_id, body.user_id, title)
    except Exception as _save_err:
        logger.debug("[API][CHAT] → save_conversation no crítico | %s", _save_err)

    config = _build_config(thread_id, body.user_id, callbacks)
    input_data = {
        "messages": [HumanMessage(content=body.message)],
        "thread_id": thread_id,
        "user_id": body.user_id,
        "rag_context": [],
        "tool_calls_pending": [],
        "tool_calls_queue": [],
        "hitl_approved": None,
        "user_memory": {},
        "guardrail_passed": None,
        "guardrail_violations": [],
        "llm_provider": "",
        "error": None,
        "intent": "unknown",
    }

    async def _stream_with_id() -> AsyncGenerator[str, None]:
        # Primer evento: notificar el thread_id activo al frontend
        yield f"data: {json.dumps({'type': 'conversation_id', 'thread_id': thread_id})}\n\n"
        async for chunk in _stream_graph(graph, input_data, config):
            yield chunk

    return StreamingResponse(
        _stream_with_id(),
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

    # Validar que el hilo tiene un checkpoint con pasos pendientes antes de reanudar
    try:
        state = await graph.aget_state(config)
        if state is None or not state.next:
            raise HTTPException(status_code=404, detail="No hay ninguna acción pendiente para este hilo")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[API][CHAT] → resume | no se pudo verificar el estado | %s", exc)

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


@router.delete("/{thread_id}", response_model=DeleteConversationResponse)
async def delete_conversation_endpoint(
    thread_id: str,
) -> DeleteConversationResponse:
    """
    Elimina una conversación y todos sus datos asociados:
    - Registro en el historial de conversaciones (memory.db → tabla conversations)
    - Checkpoints LangGraph (checkpoints.db → tablas checkpoints, checkpoint_writes,
      checkpoint_blobs)

    Después de eliminar, el thread_id queda inválido: iniciar una nueva conversación
    desde el frontend.
    """
    logger.info("[API][CHAT] → DELETE /chat/%s | eliminando conversación", thread_id)
    try:
        result = delete_conversation(thread_id)
    except Exception as exc:
        logger.error("[API][CHAT] → DELETE /chat/%s | error | %s", thread_id, exc)
        raise HTTPException(status_code=500, detail=f"Error al eliminar la conversación: {exc}")

    logger.info(
        "[API][CHAT] → DELETE /chat/%s | completado | memory=%s checkpoints=%s",
        thread_id, result["memory_deleted"], result["checkpoints_deleted"],
    )
    return DeleteConversationResponse(
        thread_id=thread_id,
        memory_deleted=result["memory_deleted"],
        checkpoints_deleted=result["checkpoints_deleted"],
    )


@router.get("/threads/{user_id}", response_model=ConversationListResponse)
async def list_user_conversations(
    user_id: str,
    limit: int = 30,
) -> ConversationListResponse:
    """Devuelve las últimas conversaciones del usuario para mostrar en el historial lateral."""
    logger.info("[API][CHAT] → GET /chat/threads/%s | limit=%d", user_id, limit)
    try:
        raw = list_conversations(user_id, limit=limit)
    except Exception as exc:
        logger.warning("[API][CHAT] → list_user_conversations | error | %s", exc)
        raw = []
    convs = [
        ConversationSummary(
            thread_id=r["thread_id"],
            title=r["title"],
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at"),
        )
        for r in raw
    ]
    return ConversationListResponse(user_id=user_id, conversations=convs)
