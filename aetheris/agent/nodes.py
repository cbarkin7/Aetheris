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
    "create_calendar_event",
    "send_email",
    "delete_file",
    "move_file",
    "update_event",
    "delete_event",
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
    if not get_settings().guardrails_enabled:
        return {"guardrail_passed": True, "guardrail_violations": []}

    last_human = _last_human(state["messages"])
    if not last_human:
        return {"guardrail_passed": True, "guardrail_violations": []}

    text = last_human.content if isinstance(last_human.content, str) else ""
    result = _get_input_guard().check(text)

    if not result.passed:
        logger.warning("Guardrail de entrada bloqueó mensaje: %s", result.violations)
        return {"guardrail_passed": False, "guardrail_violations": result.violations}

    if result.redactions and result.sanitized_text != text:
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
    memory = load_user_memory(user_id)

    last_human = _last_human(state["messages"])
    if last_human:
        query = last_human.content if isinstance(last_human.content, str) else ""
        try:
            from aetheris.memory.mem0_memory import search_memory
            mem0_results = search_memory(query, user_id, limit=3)
            mem0_snippets = [r.get("memory", "") for r in mem0_results if r.get("memory")]
            if mem0_snippets:
                memory["_mem0_context"] = " | ".join(mem0_snippets)
        except Exception as exc:
            logger.debug("mem0 no disponible: %s", exc)

    logger.debug("Memoria cargada para user='%s': %d entradas", user_id, len(memory))
    return {"user_memory": memory}


# ---------------------------------------------------------------------------
# Nodo: manager_node  (sustituye a router_node)
# ---------------------------------------------------------------------------

def manager_node(state: AgentState) -> dict:
    """
    Agente manager: analiza la conversación y planifica la secuencia óptima de herramientas.

    Devuelve el primer paso como `intent` y los pasos restantes como `execution_plan`.
    Puede encadenar hasta 2 herramientas (e.g. rag → web_search).
    """
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
        logger.warning("Manager: error al parsear plan (%s) — usando plain_llm", exc)
        steps = ["plain_llm"]

    first_step = steps[0]
    remaining = steps[1:]

    logger.info(
        "Manager: plan=%s, proveedor=%s",
        steps, provider,
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
        logger.debug("Plan dispatch: siguiente paso='%s', restante=%s", next_intent, plan)
        return {"intent": next_intent, "execution_plan": plan}
    return {"intent": "plain_llm", "execution_plan": []}


# ---------------------------------------------------------------------------
# Nodo: rag_node
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Recupera fragmentos relevantes de documentos para el último mensaje."""
    last_human = _last_human(state["messages"])
    if not last_human:
        return {"rag_context": []}

    query = last_human.content if isinstance(last_human.content, str) else ""
    results = retrieve(query)
    context = [{"content": r.content, "source": r.source, "score": r.score} for r in results]
    logger.info("RAG: recuperados %d fragmentos para query='%.60s'", len(context), query)
    return {"rag_context": context}


# ---------------------------------------------------------------------------
# Nodo: web_search_node
# ---------------------------------------------------------------------------

def web_search_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """Ejecuta una búsqueda web de Tavily mediante herramientas MCP."""
    if not mcp_tools:
        logger.warning("web_search_node: sin herramientas MCP — fallback a plain_llm")
        return {"intent": "plain_llm"}

    tavily_tool = _find_tool_by_keyword(mcp_tools, "search")
    if not tavily_tool:
        logger.warning("web_search_node: herramienta de búsqueda no encontrada")
        return {"intent": "plain_llm"}

    last_human = _last_human(state["messages"])
    query = last_human.content if last_human else ""

    try:
        result = tavily_tool.invoke({"query": query})
        search_message = AIMessage(content=f"[Resultados de búsqueda web]\n{result}")
        return {"messages": [search_message]}
    except Exception as exc:
        logger.error("web_search_node: búsqueda Tavily fallida: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Nodo: hitl_node
# ---------------------------------------------------------------------------

def hitl_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """
    Inspecciona la acción de Google y prepara tool_calls_pending.
    Se ejecuta DESPUÉS de la interrupción (interrupt_before=["hitl_node"]).
    """
    last_human = _last_human(state["messages"])
    if not last_human or not mcp_tools:
        return {"hitl_approved": False}

    google_tools = [
        t for t in mcp_tools
        if any(kw in t.name.lower() for kw in ("google", "calendar", "gmail", "drive", "event", "email"))
    ]
    if not google_tools:
        return {"hitl_approved": False, "intent": "plain_llm"}

    llm_with_tools, _ = get_llm(tools=google_tools)
    response = llm_with_tools.invoke(list(state["messages"]))

    tool_calls = getattr(response, "tool_calls", []) or []
    pending = [
        {"name": tc["name"], "args": tc["args"]}
        for tc in tool_calls
        if tc["name"] in _HITL_DESTRUCTIVE_ACTIONS
    ]
    return {"tool_calls_pending": pending, "messages": [response]}


# ---------------------------------------------------------------------------
# Nodo: google_action_node
# ---------------------------------------------------------------------------

def google_action_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """Ejecuta las llamadas aprobadas a herramientas de Google Workspace."""
    pending = state.get("tool_calls_pending", [])
    if not pending or not mcp_tools:
        return {}

    tool_map = {t.name: t for t in mcp_tools}
    result_messages: list = []

    for call in pending:
        tool = _find_tool(mcp_tools, call["name"])
        if not tool:
            logger.warning("google_action_node: herramienta '%s' no encontrada", call["name"])
            continue
        try:
            result = tool.invoke(call["args"])
            result_messages.append(ToolMessage(content=str(result), tool_call_id=call["name"]))
            logger.info("google_action_node: ejecutada '%s'", call["name"])
        except Exception as exc:
            logger.error("google_action_node: '%s' falló: %s", call["name"], exc)
            result_messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=call["name"]))

    return {"messages": result_messages, "tool_calls_pending": []}


# ---------------------------------------------------------------------------
# Nodo: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> dict:
    """
    Genera la respuesta final del asistente incorporando contexto RAG,
    resultados de herramientas y memoria del usuario.
    """
    # Rechazar si el guardrail bloqueó la entrada
    if state.get("guardrail_passed") is False:
        violations = state.get("guardrail_violations", [])
        logger.warning("llm_node: respuesta bloqueada por guardrails: %s", violations)
        rejection = (
            "Lo siento, no puedo procesar esa solicitud. Se han detectado "
            "patrones de seguridad que impiden continuar. Por favor, reformula tu mensaje."
        )
        return {"messages": [AIMessage(content=rejection)]}

    llm, provider = get_llm()
    memory_str = json.dumps(state.get("user_memory", {}), ensure_ascii=False, indent=2)
    system_content = SYSTEM_PROMPT.format(user_memory=memory_str)

    rag_context = state.get("rag_context", [])
    if rag_context:
        context_str = "\n\n".join(
            f"[{i+1}] (fuente: {c['source']}, puntuación: {c['score']:.2f})\n{c['content']}"
            for i, c in enumerate(rag_context)
        )
        system_content += "\n\n" + RAG_SYSTEM_PROMPT.format(rag_context=context_str)

    if state.get("hitl_approved") is False and state.get("tool_calls_pending"):
        system_content += "\n\nEl usuario ha rechazado la acción de Google solicitada. Acéptalo con amabilidad."

    messages = [SystemMessage(content=system_content)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response], "llm_provider": provider}


# ---------------------------------------------------------------------------
# Nodo: output_guardrail_node
# ---------------------------------------------------------------------------

def output_guardrail_node(state: AgentState) -> dict:
    """Sanea la última respuesta del asistente antes de entregarla."""
    if not get_settings().guardrails_enabled:
        return {}

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {}

    text = last_msg.content if isinstance(last_msg.content, str) else ""
    result = _get_output_guard().check(text)

    if result.sanitized_text != text:
        logger.info(
            "Guardrail de salida: redacciones=%s violaciones=%s",
            result.redactions, result.violations,
        )
        return {"messages": [AIMessage(content=result.sanitized_text)]}

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
    llm, _ = get_llm()
    user_id = state.get("user_id", "default")
    thread_id = state.get("thread_id", "")

    # Capa 1 + 2: extraer preferencias y almacenar en SQLite + Chroma
    updates = extract_memory_updates(list(state["messages"]), llm)
    if updates:
        upsert_user_memory(user_id, updates)
        for key, value in updates.items():
            try:
                store_long_term_fact(user_id, f"{key}: {value}", source="memory_extraction")
            except Exception as exc:
                logger.debug("Error almacenando hecho a largo plazo: %s", exc)
        logger.info("Guardadas %d actualizaciones de memoria para user='%s'", len(updates), user_id)

    # Capa 3: registrar en mem0
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
    except Exception as exc:
        logger.debug("mem0 no disponible para guardar: %s", exc)

    return {}
