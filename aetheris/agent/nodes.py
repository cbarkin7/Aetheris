"""
Funciones de nodo LangGraph para el agente AETHERIS.

Cada nodo recibe AgentState y devuelve un dict parcial de actualización del estado.
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aetheris.agent.prompts import (
    HITL_DESCRIPTION_PROMPT,
    MANAGER_PROMPT,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEB_TOOL_SELECTOR_PROMPT,
)
from aetheris.agent.state import AgentState
from aetheris.config import get_settings
from aetheris.guardrails.input_guard import InputGuard
from aetheris.guardrails.output_guard import OutputGuard
from aetheris.llm import get_llm
from aetheris.memory.long_term import (
    extract_memory_updates,
    load_user_memory,
    store_long_term_fact,
    upsert_user_memory,
)
from aetheris.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_VALID_INTENTS = frozenset({"rag", "web_search", "google_action", "plain_llm"})

_HITL_DESTRUCTIVE_ACTIONS = frozenset({
    # --- @cocal/google-calendar-mcp (nombres reales con guión) ---
    "create-event",
    "create-events",
    "update-event",
    "delete-event",
    "respond-to-event",
    # variantes con guión bajo por compatibilidad
    "create_event",
    "update_event",
    "delete_event",
    "create_calendar_event",
    # --- @gongrzhe/server-gmail-autoauth-mcp (nombres reales con guión) ---
    "send-email",
    "reply-to-email",
    "create-draft",
    # variantes con guión bajo
    "send_email",
    "send_gmail",
    "reply_to_email",
    "create_draft",
    # --- Drive (futuro) ---
    "delete-file",
    "move-file",
    "delete_file",
    "move_file",
})

# Singletons de guardrails (inicialización perezosa)
_input_guard: InputGuard | None = None
_output_guard: OutputGuard | None = None


def _get_input_guard() -> InputGuard:
    global _input_guard
    if _input_guard is None:
        s = get_settings()
        _input_guard = InputGuard(
            max_length=s.guardrails_max_input_length,
            redact_pii=s.guardrails_redact_pii,
            block_injections=s.guardrails_block_injections,
        )
    return _input_guard


def _get_output_guard() -> OutputGuard:
    global _output_guard
    if _output_guard is None:
        _output_guard = OutputGuard()
    return _output_guard


def _last_human(messages) -> HumanMessage | None:
    """Devuelve el último HumanMessage del historial, o None."""
    return next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)


def _messages_to_text(messages) -> str:
    """Serializa el historial de mensajes a texto legible por el LLM."""
    parts = []
    for m in messages:
        role = getattr(m, "type", type(m).__name__).replace("message", "").strip("_")
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _find_tool(tools: list, name: str):
    """Busca una herramienta por nombre exacto en la lista de herramientas MCP."""
    return next((t for t in tools if t.name == name), None)


def _find_tool_by_keyword(tools: list, keyword: str):
    """Busca la primera herramienta cuyo nombre contenga la palabra clave."""
    return next((t for t in tools if keyword in t.name.lower()), None)


# ---------------------------------------------------------------------------
# Nodo: input_guardrail_node
# ---------------------------------------------------------------------------

def input_guardrail_node(state: AgentState) -> dict:
    """
    Valida y sanea el mensaje del usuario.
    Si detecta inyección de prompt, bloquea y redirige a llm_node con rechazo.
    Si hay PII, redacta el mensaje y continúa normalmente.
    """
    user_id = state.get("user_id", "?")

    if not get_settings().guardrails_enabled:
        logger.debug("[GUARDRAIL-IN] → input_guardrail_node | saltado | guardrails desactivados")
        return {"guardrail_passed": True, "guardrail_violations": []}

    last_human = _last_human(state["messages"])
    if not last_human:
        logger.debug("[GUARDRAIL-IN] → input_guardrail_node | saltado | sin mensaje humano")
        return {"guardrail_passed": True, "guardrail_violations": []}

    text = last_human.content if isinstance(last_human.content, str) else ""
    logger.info(
        "[GUARDRAIL-IN] → input_guardrail_node | inicio | user='%s' msg_len=%d",
        user_id, len(text),
    )

    result = _get_input_guard().check(text)

    if not result.passed:
        logger.warning(
            "[GUARDRAIL-IN] → input_guardrail_node | BLOQUEADO | user='%s' violaciones=%s",
            user_id, result.violations,
        )
        return {"guardrail_passed": False, "guardrail_violations": result.violations}

    redacted = result.redactions and result.sanitized_text != text
    logger.info(
        "[GUARDRAIL-IN] → input_guardrail_node | completado | passed=True redacciones_pii=%s",
        len(result.redactions) if redacted else 0,
    )

    if redacted:
        return {
            "guardrail_passed": True,
            "guardrail_violations": [],
            "messages": [HumanMessage(content=result.sanitized_text)],
        }

    return {"guardrail_passed": True, "guardrail_violations": []}


# ---------------------------------------------------------------------------
# Nodo: load_memory_node
# ---------------------------------------------------------------------------

def load_memory_node(state: AgentState) -> dict:
    """Carga preferencias a largo plazo del usuario e inyecta contexto de mem0."""
    user_id = state.get("user_id", "default")
    logger.info("[MEMORIA-CARGA] → load_memory_node | inicio | user='%s'", user_id)

    memory = load_user_memory(user_id)

    last_human = _last_human(state["messages"])
    mem0_ok = False
    if last_human:
        query = last_human.content if isinstance(last_human.content, str) else ""
        try:
            from aetheris.memory.mem0_memory import search_memory
            mem0_results = search_memory(query, user_id, limit=3)
            mem0_snippets = [r.get("memory", "") for r in mem0_results if r.get("memory")]
            if mem0_snippets:
                memory["_mem0_context"] = " | ".join(mem0_snippets)
                mem0_ok = True
        except Exception as exc:
            logger.debug("[MEMORIA-CARGA] → mem0 no disponible | %s", exc)

    logger.info(
        "[MEMORIA-CARGA] → load_memory_node | completado | entradas=%d mem0=%s",
        len(memory), "ok" if mem0_ok else "no disponible",
    )
    return {"user_memory": memory}


# ---------------------------------------------------------------------------
# Nodo: manager_node
# ---------------------------------------------------------------------------

def manager_node(state: AgentState) -> dict:
    """
    Agente manager: analiza la conversación y planifica la secuencia óptima de herramientas.

    Devuelve el primer paso como `intent` y los pasos restantes como `execution_plan`.
    Puede encadenar hasta 2 herramientas (e.g. rag → web_search).
    """
    n_msgs = len(state["messages"])
    user_id = state.get("user_id", "?")
    logger.info(
        "[MANAGER] → manager_node | inicio | user='%s' mensajes=%d",
        user_id, n_msgs,
    )

    llm, provider = get_llm()
    messages_text = _messages_to_text(state["messages"])
    memory_str = json.dumps(state.get("user_memory", {}), ensure_ascii=False)
    prompt = MANAGER_PROMPT.format(user_memory=memory_str, messages=messages_text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Limpiar bloques de código markdown si el LLM los incluye
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        plan_data = json.loads(raw)
        steps = [s for s in plan_data.get("steps", ["plain_llm"]) if s in _VALID_INTENTS]
        if not steps:
            steps = ["plain_llm"]

    except Exception as exc:
        logger.warning("[MANAGER] → manager_node | error al parsear plan | %s → fallback plain_llm", exc)
        steps = ["plain_llm"]

    first_step = steps[0]
    remaining = steps[1:]

    logger.info(
        "[MANAGER] → manager_node | plan=%s proveedor=%s | intent='%s' pasos_restantes=%s",
        steps, provider, first_step, remaining,
    )
    return {"intent": first_step, "execution_plan": remaining, "llm_provider": provider}


# ---------------------------------------------------------------------------
# Nodo: plan_dispatch_node
# ---------------------------------------------------------------------------

def plan_dispatch_node(state: AgentState) -> dict:
    """
    Toma el siguiente paso del plan de ejecución y lo establece como `intent`.
    Si el plan está vacío, establece `plain_llm` para ir al llm_node.
    """
    plan = list(state.get("execution_plan", []))
    if plan:
        next_intent = plan.pop(0)
        logger.info(
            "[PLAN] → plan_dispatch_node | siguiente='%s' restante=%s",
            next_intent, plan,
        )
        return {"intent": next_intent, "execution_plan": plan}

    logger.info("[PLAN] → plan_dispatch_node | plan vacío → plain_llm")
    return {"intent": "plain_llm", "execution_plan": []}


# ---------------------------------------------------------------------------
# Nodo: rag_node
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Recupera fragmentos relevantes de documentos para el último mensaje."""
    last_human = _last_human(state["messages"])
    if not last_human:
        logger.warning("[RAG] → rag_node | saltado | sin mensaje humano")
        return {"rag_context": []}

    query = last_human.content if isinstance(last_human.content, str) else ""
    logger.info("[RAG] → rag_node | inicio | query='%.60s'", query)

    results = retrieve(query)
    context = [{"content": r.content, "source": r.source, "score": r.score} for r in results]

    if context:
        scores = [c["score"] for c in context]
        logger.info(
            "[RAG] → rag_node | completado | fragmentos=%d score_max=%.3f score_min=%.3f",
            len(context), max(scores), min(scores),
        )
    else:
        logger.info("[RAG] → rag_node | completado | fragmentos=0 (sin resultados sobre el umbral)")

    return {"rag_context": context}


# ---------------------------------------------------------------------------
# Nodo: web_search_node
# ---------------------------------------------------------------------------

async def web_search_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """
    Ejecuta una operación web con Tavily (async).

    Usa WEB_TOOL_SELECTOR_PROMPT para que el LLM elija la herramienta Tavily
    más adecuada (search / research / extract / crawl / map) y construya sus
    argumentos a partir del mensaje del usuario.
    Fallback a tavily_search si el selector falla.
    """
    logger.info(
        "[WEB-SEARCH] → web_search_node | inicio | herramientas_mcp=%d",
        len(mcp_tools) if mcp_tools else 0,
    )

    if not mcp_tools:
        logger.warning("[WEB-SEARCH] → web_search_node | sin herramientas MCP → fallback plain_llm")
        return {"intent": "plain_llm"}

    # Filtrar únicamente las herramientas Tavily disponibles
    tavily_tools = [t for t in mcp_tools if t.name.startswith("tavily_")]
    if not tavily_tools:
        logger.warning("[WEB-SEARCH] → web_search_node | sin herramientas Tavily → fallback plain_llm")
        return {"intent": "plain_llm"}

    last_human = _last_human(state["messages"])
    query = last_human.content if last_human else ""

    # ── Selección de herramienta via LLM ────────────────────────────────────
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description[:120]}" for t in tavily_tools
    )
    selector_prompt = WEB_TOOL_SELECTOR_PROMPT.format(
        tool_descriptions=tool_descriptions,
        query=query,
    )

    # Mapa de argumentos por defecto para cada tool Tavily
    # (usados como fallback si el selector LLM falla o devuelve args vacíos)
    _DEFAULT_ARGS: dict[str, dict] = {
        "tavily_search":   {"query": query},
        "tavily_research": {"input": query},   # ← la API exige "input", no "query"
        "tavily_extract":  {"urls": []},
        "tavily_crawl":    {"url": ""},
        "tavily_map":      {"url": ""},
    }

    selected_tool_name = "tavily_search"
    tool_args: dict = {"query": query}

    try:
        llm, _ = get_llm()
        raw = llm.invoke([HumanMessage(content=selector_prompt)]).content.strip()
        # Limpiar bloque markdown si el LLM lo incluye
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        selection = json.loads(raw)
        selected_tool_name = selection.get("tool", "tavily_search")
        # Usar args del selector; si vienen vacíos, usar los defaults por tool
        tool_args = selection.get("args") or _DEFAULT_ARGS.get(selected_tool_name, {"query": query})
        logger.info(
            "[WEB-SEARCH] → web_search_node | selector | tool='%s' args=%s",
            selected_tool_name, str(tool_args)[:120],
        )
    except Exception as exc:
        logger.warning(
            "[WEB-SEARCH] → web_search_node | selector fallido (%s) → tavily_search", exc
        )

    # Resolver la herramienta seleccionada; si no existe, usar la primera disponible
    tool = _find_tool(tavily_tools, selected_tool_name) or tavily_tools[0]
    logger.info("[WEB-SEARCH] → web_search_node | ejecutando | tool='%s'", tool.name)

    try:
        # Las herramientas MCP de langchain-mcp-adapters son async-only.
        result = await tool.ainvoke(tool_args)
        result_str = str(result)
        logger.info(
            "[WEB-SEARCH] → web_search_node | completado | tool='%s' resultado_len=%d",
            tool.name, len(result_str),
        )
        search_message = AIMessage(content=f"[Resultados web — {tool.name}]\n{result_str}")
        return {"messages": [search_message]}
    except Exception as exc:
        logger.error("[WEB-SEARCH] → web_search_node | fallido | tool='%s' error=%s", tool.name, exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Nodo: hitl_node
# ---------------------------------------------------------------------------

def hitl_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """
    Prepara tool_calls_pending con descripción legible de cada acción destructiva.
    El grafo pausa con interrupt_after DESPUÉS de este nodo: on_chain_end llega
    al SSE y el frontend muestra el modal de confirmación con las descripciones.
    """
    user_id = state.get("user_id", "?")
    logger.info(
        "[HITL] → hitl_node | inicio | user='%s' herramientas_mcp=%d",
        user_id, len(mcp_tools) if mcp_tools else 0,
    )

    last_human = _last_human(state["messages"])
    if not last_human or not mcp_tools:
        logger.warning("[HITL] → hitl_node | sin mensaje o herramientas → informando al usuario")
        return {
            "hitl_approved": False,
            "intent": "plain_llm",
            "messages": [AIMessage(
                content="⚠️ Las herramientas de Google Workspace no están disponibles en este momento. "
                        "Verifica que los servidores MCP estén configurados correctamente."
            )],
        }

    google_tools = [
        t for t in mcp_tools
        if any(kw in t.name.lower() for kw in ("google", "calendar", "gmail", "drive", "event", "email"))
    ]
    if not google_tools:
        logger.info("[HITL] → hitl_node | sin herramientas Google → informando al usuario")
        return {
            "hitl_approved": False,
            "intent": "plain_llm",
            "messages": [AIMessage(
                content="⚠️ No se encontraron herramientas de Google Workspace activas. "
                        "Comprueba las credenciales OAuth en la configuración."
            )],
        }

    # Invocar LLM con las herramientas para detectar qué acciones quiere ejecutar
    logger.info("[HITL] → hitl_node | herramientas_google=%d | detectando acciones", len(google_tools))
    llm_with_tools, _ = get_llm(tools=google_tools)
    response = llm_with_tools.invoke(list(state["messages"]))
    tool_calls = getattr(response, "tool_calls", []) or []

    # Para cada acción destructiva, generar descripción legible con HITL_DESCRIPTION_PROMPT
    llm, _ = get_llm()
    pending = []
    for tc in tool_calls:
        if tc["name"] not in _HITL_DESTRUCTIVE_ACTIONS:
            continue
        desc_prompt = HITL_DESCRIPTION_PROMPT.format(
            tool_name=tc["name"],
            tool_args=json.dumps(tc["args"], ensure_ascii=False),
        )
        try:
            description = llm.invoke([HumanMessage(content=desc_prompt)]).content.strip()
        except Exception as exc:
            logger.debug("[HITL] → hitl_node | descripción fallback | %s", exc)
            description = f"{tc['name']}: {json.dumps(tc['args'], ensure_ascii=False)}"

        # Conservar el id real del tool_call para que google_action_node
        # pueda crear ToolMessages con tool_call_id correcto (requisito OpenAI).
        pending.append({
            "id": tc.get("id", tc["name"]),
            "name": tc["name"],
            "args": tc["args"],
            "description": description,
        })

    logger.info(
        "[HITL] → hitl_node | completado | acciones_pendientes=%d | %s",
        len(pending), [p["name"] for p in pending],
    )

    if pending:
        # Hay acciones destructivas: añadir el mensaje del LLM (con tool_calls)
        # al estado para que google_action_node pueda completar el intercambio
        # con los ToolMessages correspondientes.
        return {"tool_calls_pending": pending, "messages": [response]}

    # Sin acciones destructivas: NO añadir el AIMessage con tool_calls al historial.
    # Si se añadiera sin los ToolMessages de respuesta, OpenAI devolvería 400
    # ("tool_calls must be followed by tool messages").
    # El grafo continúa hacia llm_node que genera la respuesta final.
    return {"tool_calls_pending": []}


# ---------------------------------------------------------------------------
# Nodo: google_action_node
# ---------------------------------------------------------------------------

async def google_action_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """
    Ejecuta las llamadas aprobadas a herramientas de Google Workspace (async).
    Las herramientas MCP de langchain-mcp-adapters son async-only: se usa ainvoke().
    Registra cada resultado en action_results para que el SSE emita feedback
    inmediato por acción (éxito o fallo) antes de que llm_node genere el resumen.
    """
    pending = state.get("tool_calls_pending", [])
    logger.info(
        "[GOOGLE] → google_action_node | inicio | acciones_aprobadas=%d",
        len(pending),
    )

    if not pending or not mcp_tools:
        logger.warning("[GOOGLE] → google_action_node | sin acciones o herramientas → sin efecto")
        return {"action_results": []}

    result_messages: list = []
    action_results: list[dict] = []

    for call in pending:
        tool = _find_tool(mcp_tools, call["name"])
        if not tool:
            logger.warning("[GOOGLE] → google_action_node | herramienta '%s' no encontrada", call["name"])
            action_results.append({
                "ok": False,
                "name": call["name"],
                "error": f"Herramienta '{call['name']}' no encontrada en el servidor MCP",
            })
            tc_id_missing = call.get("id", call["name"])
            result_messages.append(
                ToolMessage(
                    content=f"Error: herramienta '{call['name']}' no disponible",
                    tool_call_id=tc_id_missing,
                )
            )
            continue

        # Usar el id real del tool_call (requerido por OpenAI para emparejar con la AIMessage)
        tc_id = call.get("id", call["name"])
        try:
            logger.info("[GOOGLE] → google_action_node | ejecutando | tool='%s'", call["name"])
            result = await tool.ainvoke(call["args"])
            summary = str(result)[:300]
            result_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))
            action_results.append({"ok": True, "name": call["name"], "summary": summary})
            logger.info("[GOOGLE] → google_action_node | tool='%s' → completado", call["name"])
        except Exception as exc:
            error_msg = str(exc)
            logger.error("[GOOGLE] → google_action_node | tool='%s' → fallido | %s", call["name"], error_msg)
            result_messages.append(ToolMessage(content=f"Error: {error_msg}", tool_call_id=tc_id))
            action_results.append({"ok": False, "name": call["name"], "error": error_msg})

    ok_count = sum(1 for r in action_results if r["ok"])
    logger.info(
        "[GOOGLE] → google_action_node | completado | ok=%d/%d",
        ok_count, len(pending),
    )
    return {"messages": result_messages, "tool_calls_pending": [], "action_results": action_results}


# ---------------------------------------------------------------------------
# Nodo: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> dict:
    """
    Genera la respuesta final del asistente incorporando contexto RAG,
    resultados de herramientas y memoria del usuario.
    """
    user_id = state.get("user_id", "?")
    rag_context = state.get("rag_context", [])
    guardrail_ok = state.get("guardrail_passed")

    logger.info(
        "[LLM] → llm_node | inicio | user='%s' rag_ctx=%d guardrail=%s hitl_aprobado=%s",
        user_id, len(rag_context), guardrail_ok, state.get("hitl_approved"),
    )

    # Rechazar si el guardrail bloqueó la entrada
    if guardrail_ok is False:
        violations = state.get("guardrail_violations", [])
        logger.warning("[LLM] → llm_node | RECHAZADO por guardrail | violaciones=%s", violations)
        rejection = (
            "Lo siento, no puedo procesar esa solicitud. Se han detectado "
            "patrones de seguridad que impiden continuar. Por favor, reformula tu mensaje."
        )
        return {"messages": [AIMessage(content=rejection)]}

    llm, provider = get_llm()
    memory_str = json.dumps(state.get("user_memory", {}), ensure_ascii=False, indent=2)
    system_content = SYSTEM_PROMPT.format(user_memory=memory_str)

    if rag_context:
        context_str = "\n\n".join(
            f"[{i+1}] (fuente: {c['source']}, puntuación: {c['score']:.2f})\n{c['content']}"
            for i, c in enumerate(rag_context)
        )
        system_content += "\n\n" + RAG_SYSTEM_PROMPT.format(rag_context=context_str)
        logger.debug("[LLM] → llm_node | contexto RAG inyectado | fragmentos=%d", len(rag_context))

    if state.get("hitl_approved") is False and state.get("tool_calls_pending"):
        system_content += "\n\nEl usuario ha rechazado la acción de Google solicitada. Acéptalo con amabilidad."
        logger.info("[LLM] → llm_node | acción HITL rechazada por el usuario")

    messages = [SystemMessage(content=system_content)] + list(state["messages"])
    logger.debug("[LLM] → llm_node | invocando LLM | proveedor=%s mensajes_totales=%d", provider, len(messages))

    response = llm.invoke(messages)
    response_len = len(response.content) if isinstance(response.content, str) else 0

    logger.info(
        "[LLM] → llm_node | completado | proveedor=%s respuesta_len=%d",
        provider, response_len,
    )
    return {"messages": [response], "llm_provider": provider}


# ---------------------------------------------------------------------------
# Nodo: output_guardrail_node
# ---------------------------------------------------------------------------

def output_guardrail_node(state: AgentState) -> dict:
    """Sanea la última respuesta del asistente antes de entregarla."""
    if not get_settings().guardrails_enabled:
        logger.debug("[GUARDRAIL-OUT] → output_guardrail_node | saltado | guardrails desactivados")
        return {}

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {}

    text = last_msg.content if isinstance(last_msg.content, str) else ""
    logger.info("[GUARDRAIL-OUT] → output_guardrail_node | inicio | msg_len=%d", len(text))

    result = _get_output_guard().check(text)

    if result.sanitized_text != text:
        logger.info(
            "[GUARDRAIL-OUT] → output_guardrail_node | REDACTADO | redacciones=%s violaciones=%s",
            result.redactions, result.violations,
        )
        return {"messages": [AIMessage(content=result.sanitized_text)]}

    logger.info("[GUARDRAIL-OUT] → output_guardrail_node | completado | sin redacciones")
    return {}


# ---------------------------------------------------------------------------
# Nodo: save_memory_node
# ---------------------------------------------------------------------------

def save_memory_node(state: AgentState) -> dict:
    """
    Extrae y persiste hechos memorables de la conversación en tres capas:
      1. SQLite KV   → preferencias explícitas del usuario
      2. Chroma      → hechos semánticos a largo plazo
      3. mem0        → memoria conversacional contextual
    """
    user_id = state.get("user_id", "default")
    thread_id = state.get("thread_id", "")
    logger.info(
        "[MEMORIA-GUARDA] → save_memory_node | inicio | user='%s' thread='%s'",
        user_id, thread_id,
    )

    llm, _ = get_llm()

    # Capa 1 + 2: extraer preferencias y almacenar en SQLite + Chroma
    updates = extract_memory_updates(list(state["messages"]), llm)
    if updates:
        upsert_user_memory(user_id, updates)
        for key, value in updates.items():
            try:
                store_long_term_fact(user_id, f"{key}: {value}", source="memory_extraction")
            except Exception as exc:
                logger.debug("[MEMORIA-GUARDA] → store_long_term_fact | error (no crítico) | %s", exc)
        logger.info(
            "[MEMORIA-GUARDA] → save_memory_node | SQLite+Chroma | actualizaciones=%d claves=%s",
            len(updates), list(updates.keys()),
        )
    else:
        logger.debug("[MEMORIA-GUARDA] → save_memory_node | sin actualizaciones de memoria")

    # Capa 3: registrar en mem0
    mem0_ok = False
    try:
        from aetheris.memory.mem0_memory import add_conversation_memory
        recent_msgs = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in state["messages"][-4:]
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        if recent_msgs:
            add_conversation_memory(recent_msgs, user_id, session_id=thread_id)
            mem0_ok = True
    except Exception as exc:
        logger.debug("[MEMORIA-GUARDA] → mem0 no disponible | %s", exc)

    logger.info(
        "[MEMORIA-GUARDA] → save_memory_node | completado | user='%s' mem0=%s",
        user_id, "ok" if mem0_ok else "no disponible",
    )
    return {}
