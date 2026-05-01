"""
Funciones de nodo LangGraph para el agente AETHERIS.

Cada nodo recibe AgentState y devuelve un dict parcial de actualización del estado.
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Helper: sanear historial de mensajes antes de enviarlo al LLM
# ---------------------------------------------------------------------------

def _sanitize_tool_calls(messages: list) -> list:
    """
    Garantiza que todo AIMessage con tool_calls tenga su ToolMessage de respuesta.

    OpenAI y Bedrock exigen que cada tool_call_id del asistente sea respondido
    por un ToolMessage antes de la siguiente interacción. Cuando el usuario
    rechaza una acción HITL, el AIMessage con tool_calls queda huérfano.
    Este helper inyecta ToolMessages sintéticos para los IDs sin responder.
    """
    sanitized: list = []
    n = len(messages)

    for i, msg in enumerate(messages):
        sanitized.append(msg)

        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
            continue

        # IDs ya respondidos por ToolMessages inmediatamente siguientes
        responded: set[str] = set()
        for j in range(i + 1, n):
            nxt = messages[j]
            if isinstance(nxt, ToolMessage):
                responded.add(nxt.tool_call_id)
            else:
                break

        # Inyectar ToolMessage sintético para cada ID sin responder
        for tc in msg.tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
            if tc_id and tc_id not in responded:
                sanitized.append(
                    ToolMessage(
                        content="Acción cancelada por el usuario.",
                        tool_call_id=tc_id,
                        name=tc_name,
                    )
                )
                logger.debug(
                    "[LLM] _sanitize_tool_calls | ToolMessage sintético inyectado | tc_id='%s' name='%s'",
                    tc_id, tc_name,
                )

    return sanitized


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
    # --- @piotr-agier/google-drive-mcp (nombres exactos camelCase) ---
    # Escritura / modificación de ficheros
    "uploadFile",
    "deleteItem",
    "moveItem",
    "renameItem",
    "copyFile",
    "createTextFile",
    "updateTextFile",
    # Google Docs — operaciones destructivas
    "createGoogleDoc",
    "insertText",
    "deleteRange",
    "applyTextStyle",
    "insertTable",
    "addComment",
    # Google Sheets — operaciones destructivas
    "createGoogleSheet",
    "updateGoogleSheet",
    "appendSpreadsheetRows",
    "formatGoogleSheetCells",
    "addDataValidation",
    # Google Slides — operaciones destructivas
    "createGoogleSlides",
    "formatGoogleSlidesText",
    "setGoogleSlidesBackground",
    "deleteGoogleSlide",
    # Calendar duplicado desde Drive MCP (variante camelCase)
    "createCalendarEvent",
    "updateCalendarEvent",
    "deleteCalendarEvent",
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


def _restore_pii(args: dict, pii_map: dict[str, str]) -> dict:
    """
    Sustituye placeholders PII por los valores originales en los argumentos de una tool.

    Itera recursivamente sobre dicts y listas para cubrir estructuras anidadas
    (p. ej. {"to": ["[EMAIL_REDACTADO]"], "body": "...firma [EMAIL_REDACTADO]..."}).

    Args:
        args:    Argumentos de la tool call (dict, posiblemente anidado).
        pii_map: {placeholder: valor_original} generado por input_guardrail_node.

    Returns:
        Nuevo dict con los placeholders reemplazados por los valores reales.
    """
    if not pii_map:
        return args

    def _restore_value(val):
        if isinstance(val, str):
            for placeholder, original in pii_map.items():
                val = val.replace(placeholder, original)
            return val
        if isinstance(val, list):
            return [_restore_value(v) for v in val]
        if isinstance(val, dict):
            return {k: _restore_value(v) for k, v in val.items()}
        return val

    restored = _restore_value(args)
    if restored != args:
        logger.info(
            "[GOOGLE] _restore_pii | PII restaurado en args | placeholders=%s",
            list(pii_map.keys()),
        )
    return restored


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

    redacted = bool(result.redactions and result.sanitized_text != text)
    logger.info(
        "[GUARDRAIL-IN] → input_guardrail_node | completado | passed=True redacciones_pii=%d placeholders=%s",
        len(result.redactions) if redacted else 0,
        list(result.redactions.keys()) if redacted else [],
    )

    if redacted:
        # Reemplazar el mensaje original usando el mismo id — add_messages de LangGraph
        # actualiza en lugar de añadir cuando el id coincide.
        # pii_map {placeholder→valor_original} se persiste en state para que
        # google_action_node pueda restaurar los datos reales antes de invocar
        # herramientas de Google (Gmail, Calendar, Drive).
        return {
            "guardrail_passed": True,
            "guardrail_violations": list(result.redactions.keys()),
            "pii_map": result.redactions,
            "messages": [HumanMessage(content=result.sanitized_text, id=last_human.id)],
        }

    # Sin redacciones: NO devolver pii_map — el valor del checkpoint del turno
    # anterior debe conservarse para que google_action_node pueda restaurar PII
    # en turnos de seguimiento ("Vuelve a intentarlo", "añade X", etc.) que no
    # repiten los datos sensibles pero sí necesitan ejecutar la tool con ellos.
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
    # Excluir _mem0_context del user_memory del manager: es un resumen de conversación
    # voluminoso (pipe-separated) útil para llm_node pero que contamina el contexto
    # de enrutamiento con historial irrelevante para decidir el plan de herramientas.
    routing_memory = {k: v for k, v in state.get("user_memory", {}).items() if k != "_mem0_context"}
    memory_str = json.dumps(routing_memory, ensure_ascii=False)
    prompt = MANAGER_PROMPT.format(user_memory=memory_str, messages=messages_text)

    # FIXME: Eliminar en prod — variables inyectadas en MANAGER_PROMPT
    logger.debug(
        "[FIXME-PROMPT] manager_node | MANAGER_PROMPT vars\n"
        "  routing_memory → %s\n"
        "  messages       →\n%s",
        memory_str[:400],
        messages_text[:600],
    )

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

    # ── Corrección determinista ───────────────────────────────────────────────
    # El LLM tiende a elegir plain_llm cuando no detecta señales explícitas.
    # La override se aplica en cascada con prioridad decreciente:
    #
    # 1. Si el intent previo del checkpoint era google_action → mantenerlo.
    #    Cubre "Vuelve a intentarlo", "inténtalo de nuevo", "añade X y reintenta"
    #    donde el usuario continúa una acción de Google sin repetir los datos.
    #
    # 2. Si Chroma tiene fragmentos → forzar rag.
    #    Cubre preguntas factuales donde el LLM no sabe qué hay en los documentos.
    if steps == ["plain_llm"]:
        prev_intent = state.get("intent", "")
        if prev_intent == "google_action":
            steps = ["google_action"]
            logger.info(
                "[MANAGER] → manager_node | plain_llm→google_action (override) | "
                "contexto de acción Google activo en checkpoint"
            )
        else:
            try:
                from aetheris.rag.retriever import get_vectorstore
                _chroma_count = get_vectorstore()._collection.count()
                if _chroma_count > 0:
                    steps = ["rag"]
                    logger.info(
                        "[MANAGER] → manager_node | plain_llm→rag (override) | %d fragmentos en Chroma",
                        _chroma_count,
                    )
            except Exception:
                pass  # Si Chroma no está disponible, mantener plain_llm
    # ─────────────────────────────────────────────────────────────────────────

    # Fix 3: "plain_llm" como segundo paso es inválido.
    # llm_node siempre se ejecuta al final; incluirlo en el plan hace que
    # plan_dispatch_node sobreescriba intent="plain_llm" y llm_node pierda
    # la protección "not found" del RAG cuando no hay resultados.
    if len(steps) == 2 and steps[1] == "plain_llm":
        steps = [steps[0]]
        logger.info(
            "[MANAGER] → manager_node | plan saneado | 'plain_llm' eliminado como segundo paso"
        )

    first_step = steps[0]
    remaining = steps[1:]

    logger.info(
        "[MANAGER] → manager_node | plan=%s proveedor=%s | intent='%s' pasos_restantes=%s",
        steps, provider, first_step, remaining,
    )

    # Fix 1: Limpiar contextos del turno anterior para evitar contaminación
    # entre preguntas (state bleed). web_context y rag_context son propios de
    # cada turno y no deben persistir en el checkpointer entre turnos.
    return {
        "intent": first_step,
        "execution_plan": remaining,
        "llm_provider": provider,
        "web_context": None,  # Fix 1: limpiar entre turnos
        "rag_context": [],    # Fix 1: limpiar entre turnos
    }


# ---------------------------------------------------------------------------
# Nodo: plan_dispatch_node
# ---------------------------------------------------------------------------

def plan_dispatch_node(state: AgentState) -> dict:
    """
    Toma el siguiente paso del plan de ejecución y lo establece como `intent`.
    Si el plan está vacío, conserva el intent actual (Fix 2: no sobreescribir con plain_llm).
    """
    plan = list(state.get("execution_plan", []))
    if plan:
        next_intent = plan.pop(0)
        logger.info(
            "[PLAN] → plan_dispatch_node | siguiente='%s' restante=%s",
            next_intent, plan,
        )
        return {"intent": next_intent, "execution_plan": plan}

    # Fix 2: No sobreescribir intent cuando el plan ya está vacío.
    # Forzar intent="plain_llm" haría que llm_node pierda el contexto de la
    # operación actual (p. ej. "rag") y omita la protección "not found".
    logger.info("[PLAN] → plan_dispatch_node | plan vacío → llm_node (intent actual conservado)")
    return {"execution_plan": []}


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

    # FIXME: Eliminar en prod — variables inyectadas en WEB_TOOL_SELECTOR_PROMPT
    logger.debug(
        "[FIXME-PROMPT] web_search_node | WEB_TOOL_SELECTOR_PROMPT vars\n"
        "  query            → %.200s\n"
        "  tool_descriptions→\n%s",
        query,
        tool_descriptions,
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

    # Timeouts por herramienta: tavily_research es más lento que el resto
    _TOOL_TIMEOUTS = {
        "tavily_research": 45.0,
        "tavily_crawl":    30.0,
        "tavily_extract":  20.0,
        "tavily_search":   15.0,
        "tavily_map":      15.0,
    }
    timeout_s = _TOOL_TIMEOUTS.get(tool.name, 20.0)

    try:
        import asyncio
        # Las herramientas MCP de langchain-mcp-adapters son async-only.
        result = await asyncio.wait_for(tool.ainvoke(tool_args), timeout=timeout_s)
        result_str = str(result)
        logger.info(
            "[WEB-SEARCH] → web_search_node | completado | tool='%s' resultado_len=%d",
            tool.name, len(result_str),
        )
        # Guardamos en web_context (no en messages) para que llm_node lo inyecte
        # en el system prompt. Si lo guardásemos como AIMessage, el LLM lo
        # interpretaría como una respuesta suya anterior y generaría respuestas
        # incoherentes del tipo "voy a buscar más información".
        web_context = f"[Búsqueda web — {tool.name}]\n{result_str}"
        # intent="web_search" evita que route_after_tool aplique el fallback RAG→web
        # de nuevo cuando ya venimos de web_search_node.
        return {"web_context": web_context, "intent": "web_search"}
    except asyncio.TimeoutError:
        logger.warning(
            "[WEB-SEARCH] → web_search_node | TIMEOUT (%.0fs) en tool='%s' → fallback tavily_search",
            timeout_s, tool.name,
        )
        # Fallback a tavily_search si la herramienta principal tardó demasiado
        fallback = _find_tool(tavily_tools, "tavily_search") or tavily_tools[0]
        if fallback.name != tool.name:
            try:
                result = await asyncio.wait_for(
                    fallback.ainvoke({"query": query}), timeout=15.0
                )
                result_str = str(result)
                web_context = f"[Búsqueda web — {fallback.name} (fallback)]\n{result_str}"
                logger.info("[WEB-SEARCH] → web_search_node | fallback completado | len=%d", len(result_str))
                return {"web_context": web_context, "intent": "web_search"}
            except Exception as fb_exc:
                logger.error("[WEB-SEARCH] → web_search_node | fallback también falló | %s", fb_exc)
        return {"web_context": None, "error": f"Timeout en {tool.name}", "intent": "web_search"}
    except Exception as exc:
        logger.error("[WEB-SEARCH] → web_search_node | fallido | tool='%s' error=%s", tool.name, exc)
        return {"web_context": None, "error": str(exc), "intent": "web_search"}


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

    # Invocar LLM con las herramientas para detectar qué acciones quiere ejecutar.
    # _sanitize_tool_calls es obligatorio: en turnos posteriores el historial puede
    # contener AIMessages con tool_calls huérfanos (sin ToolMessage de respuesta)
    # de intentos anteriores que el usuario amplió con más información.
    # Sin esto, OpenAI devuelve 400 "tool_calls must be followed by tool messages".
    logger.info("[HITL] → hitl_node | herramientas_google=%d | detectando acciones", len(google_tools))
    llm_with_tools, _ = get_llm(tools=google_tools)
    sanitized_messages = _sanitize_tool_calls(list(state["messages"]))
    response = llm_with_tools.invoke(sanitized_messages)
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

    # Sin acciones destructivas (lecturas: list_events, list_calendars, read_email, etc.)
    # → auto-aprobar y ejecutar directamente en google_action_node sin interrupción HITL.
    # hitl_approved=True hace que route_after_hitl_node salte hitl_wait_node.
    if tool_calls:
        non_destructive = [
            {
                "id": tc.get("id", tc["name"]),
                "name": tc["name"],
                "args": tc["args"],
                "description": f"Lectura: {tc['name']}",
            }
            for tc in tool_calls
        ]
        logger.info(
            "[HITL] → hitl_node | acciones de lectura auto-aprobadas=%d | %s",
            len(non_destructive), [p["name"] for p in non_destructive],
        )
        # El AIMessage con tool_calls DEBE añadirse al historial para que
        # google_action_node pueda completar el intercambio con ToolMessages.
        return {
            "tool_calls_pending": non_destructive,
            "hitl_approved": True,
            "messages": [response],
        }

    # Sin ninguna tool call generada
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
            # Restaurar PII antes de ejecutar: los args contienen placeholders
            # ([EMAIL_REDACTADO], [TELEFONO_REDACTADO], etc.) porque el LLM trabajó
            # con el texto saneado. pii_map devuelve los valores originales necesarios
            # para que Gmail/Calendar/Drive reciban datos válidos.
            pii_map = state.get("pii_map", {})
            exec_args = _restore_pii(call["args"], pii_map)
            logger.info("[GOOGLE] → google_action_node | ejecutando | tool='%s'", call["name"])
            result = await tool.ainvoke(exec_args)
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
    return {
        "messages": result_messages,
        "tool_calls_pending": [],
        "hitl_approved": None,   # reset para que el siguiente turno empiece limpio
        "action_results": action_results,
    }


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

    intent = state.get("intent", "plain_llm")

    if rag_context:
        context_str = "\n\n".join(
            f"[{i+1}] (fuente: {c['source']}, puntuación: {c['score']:.2f})\n{c['content']}"
            for i, c in enumerate(rag_context)
        )
        system_content += "\n\n" + RAG_SYSTEM_PROMPT.format(rag_context=context_str)
        logger.debug("[LLM] → llm_node | contexto RAG inyectado | fragmentos=%d", len(rag_context))
    elif intent == "rag":
        # RAG se consultó pero no encontró nada: prohibir respuesta desde conocimiento general
        system_content += (
            "\n\nEl sistema ha buscado en los documentos del usuario y NO ha encontrado "
            "información relevante para esta pregunta. "
            "Debes responder ÚNICAMENTE: "
            "'No he encontrado información sobre ese tema en tus documentos. "
            "Si quieres, puedo buscar información actualizada en internet — "
            "indícamelo y lo haré.'"
        )
        logger.info("[LLM] → llm_node | RAG sin resultados → respuesta de no encontrado")

    web_context = state.get("web_context")
    if web_context:
        system_content += (
            "\n\n## Resultados de búsqueda web (búsqueda ya ejecutada)\n"
            "IMPORTANTE: La búsqueda web se ha realizado AUTOMÁTICAMENTE antes de esta respuesta. "
            "Los resultados están disponibles a continuación. "
            "NO digas que vas a buscar, ni 'un momento', ni 'procederé a buscar' — "
            "eso ya ocurrió. Integra directamente la información en tu respuesta "
            "combinándola con el contexto de documentos si lo hubiera. "
            "Cita las fuentes cuando sea relevante.\n\n"
            + web_context
        )
        logger.debug("[LLM] → llm_node | contexto web inyectado | len=%d", len(web_context))

    if state.get("hitl_approved") is False and state.get("tool_calls_pending"):
        system_content += "\n\nEl usuario ha rechazado la acción de Google solicitada. Acéptalo con amabilidad."
        logger.info("[LLM] → llm_node | acción HITL rechazada por el usuario")

    # FIXME: Eliminar en prod — variables inyectadas en SYSTEM_PROMPT / RAG_SYSTEM_PROMPT
    logger.debug(
        "[FIXME-PROMPT] llm_node | variables de prompt\n"
        "  intent      → %s\n"
        "  user_memory → %.300s\n"
        "  rag_context → %d fragmentos | %s\n"
        "  web_context → %s",
        intent,
        memory_str,
        len(rag_context),
        [(c.get("source", "?"), round(c.get("score", 0), 3)) for c in rag_context],
        (web_context[:200] + "…") if web_context else "None",
    )

    raw_messages = _sanitize_tool_calls(list(state["messages"]))
    messages = [SystemMessage(content=system_content)] + raw_messages
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
        return {"guardrail_passed": True, "guardrail_violations": []}

    messages = state.get("messages", [])
    if not messages:
        return {"guardrail_passed": True, "guardrail_violations": []}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {"guardrail_passed": True, "guardrail_violations": []}

    text = last_msg.content if isinstance(last_msg.content, str) else ""
    logger.info("[GUARDRAIL-OUT] → output_guardrail_node | inicio | msg_len=%d", len(text))

    result = _get_output_guard().check(text)

    if result.sanitized_text != text:
        logger.info(
            "[GUARDRAIL-OUT] → output_guardrail_node | REDACTADO | redacciones=%s violaciones=%s",
            result.redactions, result.violations,
        )
        return {
            "messages": [AIMessage(content=result.sanitized_text)],
            "guardrail_passed": True,
            "guardrail_violations": result.violations,
        }

    logger.info("[GUARDRAIL-OUT] → output_guardrail_node | completado | passed=True redacciones=0")
    return {"guardrail_passed": True, "guardrail_violations": []}


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
