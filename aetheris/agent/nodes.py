"""
Funciones de nodo LangGraph para el agente AETHERIS.

Cada nodo recibe AgentState y devuelve un dict parcial de actualización del estado.
"""
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

from aetheris.agent.prompts import (
    GOOGLE_PLANNER_PROMPT,
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
    "delete-email",
    "batch-delete-emails",
    "empty-trash",
    # variantes con guión bajo
    "send_email",
    "send_gmail",
    "reply_to_email",
    "create_draft",
    "delete_email",
    "batch_delete_emails",
    "empty_trash",
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


def _apply_sanitized_input(messages: list, sanitized_input: str | None) -> list:
    """
    Reemplaza el contenido del último HumanMessage con la versión saneada (PII redactada)
    ÚNICAMENTE para las llamadas al LLM — los mensajes originales en state.messages
    NO se modifican y se persisten con los datos reales en el checkpoint.

    Uso: llamar justo antes de invocar el LLM; NO usar para display ni historial.
    """
    if not sanitized_input:
        return messages
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if isinstance(result[i], HumanMessage):
            original = result[i]
            result[i] = HumanMessage(
                content=sanitized_input,
                id=getattr(original, "id", None),
            )
            break
    return result


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
        # Almacenar el texto saneado en sanitized_user_input (no en messages).
        # Los nodos que llaman al LLM (manager, planner, llm_node…) aplican
        # _apply_sanitized_input() justo antes de invocar, de modo que el LLM
        # solo ve los placeholders mientras el historial persistido en BD
        # conserva los datos reales del usuario.
        # pii_map {placeholder→valor_original} permite restaurar los datos
        # en google_action_node antes de invocar las tools de Google.
        logger.info(
            "[GUARDRAIL-IN] → input_guardrail_node | PII redactada | "
            "placeholders=%s (mensaje ORIGINAL conservado en historial)",
            list(result.redactions.keys()),
        )
        return {
            "guardrail_passed": True,
            "guardrail_violations": list(result.redactions.keys()),
            "pii_map": result.redactions,
            "sanitized_user_input": result.sanitized_text,
        }

    # Sin redacciones: limpiar sanitized_user_input del turno anterior para que
    # los nodos LLM usen el mensaje original directamente (no un texto obsoleto).
    # NO limpiar pii_map — podría necesitarse en turnos de seguimiento
    # ("Vuelve a intentarlo") donde el usuario no repite los datos sensibles.
    return {"guardrail_passed": True, "guardrail_violations": [], "sanitized_user_input": None}


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
    # Aplicar PII sanitation solo para el LLM — el historial original no se toca
    _llm_messages = _apply_sanitized_input(
        list(state["messages"]), state.get("sanitized_user_input")
    )
    messages_text = _messages_to_text(_llm_messages)
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
    # 1. Si el último mensaje contiene palabras clave inequívocas de Google Workspace
    #    → forzar google_action. Evita que la rag-override capture operaciones de Drive
    #    que el LLM malclasificó como plain_llm (ej. "mueve el documento a la carpeta X").
    #
    # 2. Si el intent previo del checkpoint era google_action → mantenerlo.
    #    Cubre "Vuelve a intentarlo", "inténtalo de nuevo", "añade X y reintenta"
    #    donde el usuario continúa una acción de Google sin repetir los datos.
    #
    # 3. Si Chroma tiene fragmentos → forzar rag.
    #    Cubre preguntas factuales donde el LLM no sabe qué hay en los documentos.
    #    SOLO aplica si las reglas 1 y 2 no han disparado.
    _GOOGLE_ACTION_KEYWORDS = {
        # Drive
        "drive", "carpeta", "archivo", "fichero", "hoja de cálculo", "spreadsheet",
        "mueve", "mover", "copia", "copiar", "sube", "subir", "descarga", "descargar",
        "renombra", "renombrar", "elimina archivos", "borra archivos",
        "crea carpeta", "crea un doc", "crea un documento", "crea una hoja",
        "createfolder", "createfile", "uploadfile",
        # Calendar
        "calendar", "evento", "cita", "reunión", "reunion",
        # Gmail
        "gmail", "correo", "email", "bandeja", "borrador",
    }
    if steps == ["plain_llm"]:
        last_human_msg = _last_human(state["messages"])
        last_text = (last_human_msg.content if last_human_msg and isinstance(last_human_msg.content, str) else "").lower()

        if any(kw in last_text for kw in _GOOGLE_ACTION_KEYWORDS):
            steps = ["google_action"]
            logger.info(
                "[MANAGER] → manager_node | plain_llm→google_action (keyword-override) | "
                "mensaje contiene palabras clave de Google Workspace"
            )
        else:
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
    # google_action_iterations se resetea aquí para que cada nuevo turno
    # empiece con el contador a 0 (el bucle es por turno, no global).
    return {
        "intent": first_step,
        "execution_plan": remaining,
        "llm_provider": provider,
        "web_context": None,            # Fix 1: limpiar entre turnos
        "rag_context": [],              # Fix 1: limpiar entre turnos
        "google_action_iterations": 0,  # Reset bucle google por turno
        "tool_calls_queue": [],         # Reset cola HITL entre turnos
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
    # Para el selector LLM y los args de Tavily usar la versión saneada si existe
    query = state.get("sanitized_user_input") or query

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
# Helpers de categorización de herramientas Google por dominio
# ---------------------------------------------------------------------------

def _build_current_date_str() -> str:
    """Construye el string de fecha/hora actual con rangos de semana para el prompt."""
    _now = datetime.now(tz=ZoneInfo("Europe/Madrid"))
    _wd = _now.weekday()  # 0=lunes … 6=domingo
    _mon_this = (_now - timedelta(days=_wd)).replace(hour=0, minute=0, second=0, microsecond=0)
    _sun_this = _mon_this + timedelta(days=6)
    _mon_next = _mon_this + timedelta(days=7)
    _sun_next = _mon_next + timedelta(days=6)
    _fri_next = _mon_next + timedelta(days=4)
    return (
        f"{_now.strftime('%A, %d de %B de %Y')} — {_now.strftime('%H:%M')} (Europe/Madrid)\n"
        f"  · Semana actual    : lunes {_mon_this.strftime('%d/%m/%Y')} – domingo {_sun_this.strftime('%d/%m/%Y')}\n"
        f"  · Próxima semana   : lunes {_mon_next.strftime('%d de %B')} – domingo {_sun_next.strftime('%d de %B de %Y')}\n"
        f"  · Próxima sem. lab.: lunes {_mon_next.strftime('%d/%m/%Y')} – viernes {_fri_next.strftime('%d/%m/%Y')}"
    )


def _categorize_google_tools(tools: list) -> dict[str, list]:
    """
    Separa las herramientas Google en tres dominios: calendar, gmail, drive.

    La separación es estricta: ninguna herramienta pertenece a dos dominios.
    Esto evita que el LLM confunda herramientas entre servicios (p.ej. llamar
    a search-events de Calendar para buscar archivos en Drive).
    """
    calendar_tools: list = []
    gmail_tools: list = []
    drive_tools: list = []
    for t in tools:
        name = t.name.lower()
        if any(kw in name for kw in ("event", "calendar")):
            calendar_tools.append(t)
        elif any(kw in name for kw in ("email", "gmail", "mail", "draft")):
            gmail_tools.append(t)
        else:
            drive_tools.append(t)
    return {"calendar": calendar_tools, "gmail": gmail_tools, "drive": drive_tools}


def _detect_relevant_domains(messages: list) -> set[str]:
    """
    Detecta qué dominios Google (calendar/gmail/drive) son relevantes para
    la petición actual.

    Estrategia de dos fases para evitar que contexto de acciones anteriores
    (ToolMessages de otros dominios) contamine la petición actual:

    FASE 1 — Último mensaje humano (fuente de verdad principal).
      Si el usuario menciona explícitamente palabras de un dominio, se usa
      SOLO ese dominio, sin importar el historial de herramientas reciente.

    FASE 2 — ToolMessages recientes (solo si la Fase 1 no detecta nada).
      Útil para continuaciones multi-paso donde el usuario dice cosas como
      "ahora elimínalo" sin especificar qué servicio, y la acción anterior
      (un search, por ejemplo) indica el dominio activo.
    """
    last_human = _last_human(messages)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    ).lower()

    # ── FASE 1: dominio desde el último mensaje del usuario ──────────────────
    domains: set[str] = set()

    if any(kw in user_text for kw in (
        "evento", "cita", "reunión", "reunion", "calendar", "event", "agenda",
    )):
        domains.add("calendar")

    if any(kw in user_text for kw in (
        "email", "correo", "gmail", "mail", "draft",
        "envía", "envia", "enviar", "bandeja", "asunto",
        "borrador de correo", "borrador de email", "borrador del correo",
    )):
        domains.add("gmail")

    if any(kw in user_text for kw in (
        "archivo", "fichero", "carpeta", "folder", "drive",
        "documento", "doc", "hoja", "sheet", "spreadsheet", "slide",
        "sube", "subir", "descarga", "descargar", "mueve", "mover",
        "copia", "copiar", "renombra", "renombrar",
    )):
        domains.add("drive")

    # Si el mensaje actual indica claramente uno o más dominios → usarlos solo.
    # Los ToolMessages de acciones anteriores NO deben contaminar esta petición.
    if domains:
        return domains

    # ── FASE 2: fallback — continuaciones ("ahora elimínalo", "hazlo", etc.) ──
    # Solo se activa cuando el mensaje del usuario no menciona ningún dominio.
    # Se miran los ToolMessages recientes para inferir el dominio activo.
    for m in reversed(messages[-8:]):
        if not (isinstance(m, ToolMessage) and isinstance(m.content, str)):
            continue
        tool_text = m.content.lower()[:300]

        if any(kw in tool_text for kw in ("event", "calendar", "vcalendar")):
            domains.add("calendar")
        if any(kw in tool_text for kw in ("email", "gmail", "message-id", "subject")):
            domains.add("gmail")
        if any(kw in tool_text for kw in (
            "fileid", "file_id", "mimetype", "drive", "folder",
            "google-apps", "spreadsheet", "document",
        )):
            domains.add("drive")

        if domains:
            break  # con un ToolMessage relevante es suficiente

    if not domains:
        domains = {"calendar", "gmail", "drive"}

    return domains


def _fix_folder_creation_tools(tool_calls: list, messages: list) -> list:
    """
    Corrección determinista: reemplaza createGoogleDoc / createGoogleSheet por
    createFolder cuando el LLM los usa para crear directorios.

    La heurística se activa cuando:
    1. El último mensaje humano contiene palabras clave de carpeta/directorio.
    2. El tool_call no lleva contenido sustancial (no es realmente un documento).

    Esto evita fallos silenciosos donde el agente "crea" un documento vacío
    en lugar de la carpeta solicitada, sin depender de que el LLM siga el prompt.
    """
    last_human = _last_human(messages)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    ).lower()

    folder_keywords = ("carpeta", "folder", "directorio", "directory")
    if not any(kw in user_text for kw in folder_keywords):
        return tool_calls  # El usuario no pide carpeta → no tocar

    fixed = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}

        if name in ("createGoogleDoc", "createGoogleSheet"):
            content = str(args.get("content", args.get("body", ""))).strip()
            # Sin contenido sustancial → el LLM quería una carpeta, no un documento
            if len(content) < 20:
                folder_name = args.get("title", args.get("name", "Nueva carpeta"))
                parent_id = args.get("folderId", args.get("parentFolderId"))
                new_args: dict = {"name": folder_name}
                if parent_id:
                    new_args["parentFolderId"] = parent_id
                logger.info(
                    "[PLANNER] _fix_folder_creation_tools | %s → createFolder | "
                    "name=%r parent=%r",
                    name, folder_name, parent_id,
                )
                fixed.append({**tc, "name": "createFolder", "args": new_args})
                continue

        fixed.append(tc)

    return fixed


_LIST_TOOLS_WITH_QUERY = frozenset({
    "listGoogleDocs", "listGoogleSheets", "listGoogleSlides",
    "listFolder", "search",
})


def _fix_list_tools(tool_calls: list) -> list:
    """
    La API de Google Drive no admite orderBy cuando la query contiene fullText.
    Si el LLM genera una llamada a listGoogleDocs/Sheets/Slides/search con
    un término fullText Y un parámetro orderBy, eliminar orderBy para evitar
    el error "Sorting is not supported for queries with fullText terms".
    """
    fixed = []
    for tc in tool_calls:
        if tc.get("name", "") not in _LIST_TOOLS_WITH_QUERY:
            fixed.append(tc)
            continue

        args = dict(tc.get("args", {}) or {})
        query = str(args.get("query", "")).lower()
        has_fulltext = "fulltext" in query or "contains" in query

        if has_fulltext and "orderBy" in args:
            logger.info(
                "[PLANNER] _fix_list_tools | '%s' | fullText + orderBy → eliminando orderBy",
                tc["name"],
            )
            args.pop("orderBy")
            fixed.append({**tc, "args": args})
        else:
            fixed.append(tc)

    return fixed


def _looks_like_drive_id(value: str) -> bool:
    """
    Heurística: un Drive file ID real tiene ≥ 25 caracteres alfanuméricos
    (más guiones y guiones bajos), sin espacios ni puntos.
    Un nombre de fichero ("prueba_v1.txt", "Mi Informe Final") no cumple esto.
    """
    return (
        len(value) >= 25
        and " " not in value
        and "." not in value
        and all(c.isalnum() or c in "-_" for c in value)
    )


def _fix_delete_tools(tool_calls: list, messages: list) -> list:
    """
    Corrección determinista pre-HITL para operaciones de borrado en Drive.

    Regla invariante: SIEMPRE hay que buscar el archivo antes de borrarlo.
    Si el LLM genera un borrado con un nombre de fichero en lugar de un Drive ID,
    se convierte en una búsqueda primero. El planner volverá a llamarse con el
    resultado de la búsqueda y generará el deleteItem con el ID real.

    Casos corregidos:
    1. deleteItem(fileId=<nombre>) → search(query="name='<nombre>'")
       Solo se deja pasar deleteItem si fileId ya es un Drive ID real (≥ 25 chars).
    2. deleteGoogleSlide sin slideObjectId → mismo tratamiento:
       · Con Drive ID en presentationId → deleteItem(fileId=...)  [ya es el ID real]
       · Con nombre de fichero → search(query="name='<nombre>'")
    """
    fixed = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}

        # ── Caso 1: deleteItem con nombre de fichero en fileId ────────────────
        if name == "deleteItem":
            file_id = str(args.get("fileId", "")).strip()
            if file_id and not _looks_like_drive_id(file_id):
                # fileId parece un nombre → buscar primero
                logger.info(
                    "[PLANNER] _fix_delete_tools | deleteItem con nombre '%s' → "
                    "search para obtener fileId", file_id,
                )
                fixed.append({
                    **tc,
                    "name": "search",
                    "args": {"query": f"name='{file_id}'"},
                })
                continue

        # ── Caso 2: deleteGoogleSlide sin slideObjectId ───────────────────────
        elif name == "deleteGoogleSlide":
            slide_obj_id = str(args.get("slideObjectId", "")).strip()
            presentation_id = str(args.get("presentationId", "")).strip()

            if not slide_obj_id:
                # Sin slideObjectId → el LLM confunde "borrar archivo" con
                # "borrar diapositiva". Redirigir igual que deleteItem.
                if presentation_id and _looks_like_drive_id(presentation_id):
                    logger.info(
                        "[PLANNER] _fix_delete_tools | deleteGoogleSlide sin slideObjectId → "
                        "deleteItem | fileId=%r", presentation_id,
                    )
                    fixed.append({**tc, "name": "deleteItem", "args": {"fileId": presentation_id}})
                elif presentation_id:
                    logger.info(
                        "[PLANNER] _fix_delete_tools | deleteGoogleSlide con nombre '%s' → "
                        "search para obtener fileId", presentation_id,
                    )
                    fixed.append({
                        **tc,
                        "name": "search",
                        "args": {"query": f"name='{presentation_id}'"},
                    })
                else:
                    fixed.append(tc)
                continue

        fixed.append(tc)

    return fixed


def _format_action_description(name: str, args: dict) -> str:
    """Genera descripción legible de una acción Google sin necesidad de LLM."""
    short_args = {
        k: (str(v)[:60] + "…" if len(str(v)) > 60 else v)
        for k, v in (args or {}).items()
    }
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in list(short_args.items())[:4])
    return f"{name}({args_str})"


# ---------------------------------------------------------------------------
# Nodo: google_planner_node
# ---------------------------------------------------------------------------

def google_planner_node(state: AgentState, mcp_tools: list | None = None) -> dict:
    """
    Nodo experto en planificar acciones de Google Workspace.

    Responsabilidades:
    1. Filtrar herramientas MCP al dominio relevante (Calendar / Gmail / Drive)
       para evitar confusión entre servicios.
    2. Llamar al LLM con las herramientas filtradas y el historial completo.
    3. Si el LLM devuelve tool_calls → construir tool_calls_pending.
    4. Si el LLM devuelve texto (sin tool_calls) → la tarea está completa o
       faltan datos: activar data_collection_required para que llm_node
       haga pass-through y el mensaje ya añadido llegue al usuario.

    Este nodo NO gestiona la aprobación HITL: eso es responsabilidad de hitl_node.
    """
    user_id = state.get("user_id", "?")
    logger.info(
        "[PLANNER] → google_planner_node | inicio | user='%s' herramientas_mcp=%d",
        user_id, len(mcp_tools) if mcp_tools else 0,
    )

    if not mcp_tools:
        logger.warning("[PLANNER] → google_planner_node | sin herramientas MCP")
        return {
            "tool_calls_pending": [],
            "data_collection_required": True,
            "messages": [AIMessage(
                content="⚠️ Las herramientas de Google Workspace no están disponibles. "
                        "Verifica que los servidores MCP estén configurados correctamente."
            )],
        }

    # Filtrar únicamente herramientas Google
    google_tools = [
        t for t in mcp_tools
        if any(kw in t.name.lower() for kw in ("google", "calendar", "gmail", "drive", "event", "email"))
    ]
    if not google_tools:
        logger.warning("[PLANNER] → google_planner_node | sin herramientas Google activas")
        return {
            "tool_calls_pending": [],
            "data_collection_required": True,
            "messages": [AIMessage(
                content="⚠️ No se encontraron herramientas de Google Workspace activas. "
                        "Comprueba las credenciales OAuth en la configuración."
            )],
        }

    # Categorizar y seleccionar solo los dominios necesarios para esta petición.
    # El LLM recibe ÚNICAMENTE las herramientas del dominio relevante, lo que
    # impide físicamente que llame a search-events (Calendar) para Drive, etc.
    tool_by_domain = _categorize_google_tools(google_tools)
    # messages_orig se usa para detección de dominios y para _detect_relevant_domains
    # (acepta mensajes con datos reales, es operación local).
    # messages_for_llm aplica PII sanitation para lo que se envía al modelo.
    messages_orig = list(state["messages"])
    messages_for_llm = _apply_sanitized_input(messages_orig, state.get("sanitized_user_input"))
    relevant_domains = _detect_relevant_domains(messages_orig)

    # ── PASO 0 DETERMINÍSTICO ────────────────────────────────────────────────
    # Comprueba si ya se ejecutó con éxito una acción terminal en esta sesión
    # (desde el último HumanMessage). Si es así, no hay que planificar más:
    # generar resumen directamente sin llamar al LLM de nuevo.
    #
    # Herramientas "terminales": destruyen, crean o envían algo definitivamente.
    # Las de búsqueda/lectura NO son terminales (son pasos intermedios).
    _TERMINAL_TOOL_NAMES = frozenset({
        # Drive — ficheros y carpetas
        "deleteItem", "moveItem", "renameItem", "copyFile",
        "uploadFile", "createFolder", "createTextFile", "updateTextFile",
        # Google Docs
        "createGoogleDoc", "insertText", "deleteRange",
        "applyTextStyle", "insertTable", "addComment",
        # Google Sheets
        "createGoogleSheet", "updateGoogleSheet", "appendSpreadsheetRows",
        "formatGoogleSheetCells", "addDataValidation",
        # Google Slides
        "createGoogleSlides", "deleteGoogleSlide",
        "formatGoogleSlidesText", "setGoogleSlidesBackground",
        # Calendar
        "create-event", "create-events", "update-event", "delete-event",
        "respond-to-event", "createCalendarEvent", "updateCalendarEvent",
        "deleteCalendarEvent",
        # Gmail
        "send-email", "reply-to-email", "create-draft",
        "delete-email", "batch-delete-emails", "empty-trash",
        "send_email", "reply_to_email", "create_draft",
        "delete_email", "batch_delete_emails", "empty_trash",
    })

    # Buscar ToolMessages exitosos/fallidos desde el último HumanMessage
    _last_human_idx = max(
        (i for i, m in enumerate(messages_orig) if isinstance(m, HumanMessage)),
        default=0,
    )
    _recent = messages_orig[_last_human_idx + 1:]
    _successful_terminals = [
        m for m in _recent
        if isinstance(m, ToolMessage)
        and getattr(m, "name", "") in _TERMINAL_TOOL_NAMES
        and not str(getattr(m, "content", "")).startswith("Error:")
    ]
    _failed_recent = [
        m for m in _recent
        if isinstance(m, ToolMessage)
        and str(getattr(m, "content", "")).startswith("Error:")
    ]

    # ── PASO 0.A — Tarea completada con éxito ────────────────────────────────
    if _successful_terminals and not _failed_recent:
        # Todas las acciones terminales ejecutadas con éxito y sin fallos pendientes.
        # Construir resumen directamente sin llamar al LLM del planificador.
        _tool_names = [getattr(m, "name", "acción") for m in _successful_terminals]
        _summaries = [str(getattr(m, "content", ""))[:200] for m in _successful_terminals]
        _summary_text = "\n".join(
            f"✅ **{name}** completado:\n> {summary}"
            for name, summary in zip(_tool_names, _summaries)
        )
        logger.info(
            "[PLANNER] → google_planner_node | PASO 0.A | "
            "tarea completa | terminales=%s",
            _tool_names,
        )
        return {
            "messages": [AIMessage(content=_summary_text)],
            "tool_calls_pending": [],
            "data_collection_required": True,
            "hitl_approved": None,
        }

    # ── FIN PASO 0 DETERMINÍSTICO ────────────────────────────────────────────
    selected_tools = []
    for domain in relevant_domains:
        selected_tools.extend(tool_by_domain.get(domain, []))
    if not selected_tools:
        selected_tools = google_tools  # fallback: todos los Google tools

    logger.info(
        "[PLANNER] → google_planner_node | dominios=%s herramientas_seleccionadas=%d/%d",
        sorted(relevant_domains), len(selected_tools), len(google_tools),
    )

    # Llamar al LLM con herramientas filtradas.
    # _sanitize_tool_calls elimina AIMessages con tool_calls huérfanos (sin ToolMessage
    # de respuesta) que quedan cuando el usuario rechaza una acción HITL o cuando el
    # agente es interrumpido. Sin esto, OpenAI devuelve 400.
    llm_with_tools, _ = get_llm(tools=selected_tools)
    sanitized_messages = _sanitize_tool_calls(messages_for_llm)
    system_msg = SystemMessage(
        content=GOOGLE_PLANNER_PROMPT.format(current_date=_build_current_date_str())
    )
    response = llm_with_tools.invoke([system_msg] + sanitized_messages)
    tool_calls = getattr(response, "tool_calls", []) or []

    if not tool_calls:
        # El LLM respondió con texto: bien porque la tarea está completa (PASO 0)
        # o porque faltan datos. En ambos casos, activar data_collection_required
        # para que llm_node haga pass-through y el mensaje llegue al usuario.
        logger.info(
            "[PLANNER] → google_planner_node | sin tool_calls → "
            "data_collection_required (tarea completa o datos faltantes)"
        )
        return {
            "messages": [response],
            "tool_calls_pending": [],
            "data_collection_required": True,
            "hitl_approved": None,
        }

    # Correcciones deterministas post-LLM (en orden de aplicación):
    # 1. createGoogleDoc/Sheet → createFolder cuando el usuario pide una carpeta.
    tool_calls = _fix_folder_creation_tools(tool_calls, messages_orig)
    # 2. deleteGoogleSlide sin slideObjectId / deleteItem con nombre → search primero.
    tool_calls = _fix_delete_tools(tool_calls, messages_orig)
    # 3. listGoogleDocs/Sheets/search con fullText + orderBy → eliminar orderBy.
    tool_calls = _fix_list_tools(tool_calls)

    # Construir tool_calls_pending con requires_approval para cada acción
    pending = []
    for tc in tool_calls:
        requires_approval = tc["name"] in _HITL_DESTRUCTIVE_ACTIONS
        pending.append({
            "id": tc.get("id", tc["name"]),
            "name": tc["name"],
            "args": tc["args"],
            "requires_approval": requires_approval,
        })

    logger.info(
        "[PLANNER] → google_planner_node | acciones=%d | %s",
        len(pending), [p["name"] for p in pending],
    )

    # El AIMessage con tool_calls DEBE añadirse al historial ahora, para que
    # google_action_node pueda crear ToolMessages con los tool_call_id correctos
    # (requisito de la API de OpenAI: cada tool_call debe tener su ToolMessage).
    return {
        "messages": [response],
        "tool_calls_pending": pending,
        "data_collection_required": False,
        "hitl_approved": None,  # Reset para que hitl_node evalúe de nuevo
    }


# ---------------------------------------------------------------------------
# Nodo: hitl_node
# ---------------------------------------------------------------------------

def hitl_node(state: AgentState) -> dict:
    """
    Gestiona la aprobación HITL de las acciones planificadas por google_planner_node,
    procesando UNA acción por iteración para que el usuario pueda aprobar o rechazar
    cada acción individualmente.

    Flujo de entrada:
    - Desde google_planner_node (primera vez):
        hitl_approved=None, tool_calls_pending=[A,B,C], tool_calls_queue=[]
        → combina pending+queue, saca A, pone B,C en queue.
    - Desde google_action_node (después de ejecutar):
        hitl_approved=None, tool_calls_pending=[], tool_calls_queue=[B,C]
        → saca B, pone C en queue.
    - Desde hitl_wait_node tras rechazo (hitl_approved=False):
        tool_calls_pending=[A], tool_calls_queue=[B,C]
        → inyecta ToolMessage de rechazo para A, saca B, pone C en queue.

    Decisión de flujo por acción:
    - requires_approval=False → auto-aprobar (hitl_approved=True), sin interrupción.
    - requires_approval=True  → esperar aprobación humana (hitl_approved=None →
      route_after_hitl_node → hitl_wait_node).
    """
    queue = list(state.get("tool_calls_queue", []))
    pending = list(state.get("tool_calls_pending", []))
    hitl_approved = state.get("hitl_approved")
    user_id = state.get("user_id", "?")

    result_messages: list = []

    if hitl_approved is False:
        # La acción anterior fue rechazada por el usuario.
        # Inyectar ToolMessage sintético para que OpenAI no rechace el historial
        # (el AIMessage ya tiene el tool_call_id de esta acción; necesita respuesta).
        if pending:
            rejected = pending[0]
            tc_id = rejected.get("id", rejected.get("name", ""))
            result_messages.append(ToolMessage(
                content="Acción rechazada por el usuario.",
                tool_call_id=tc_id,
                name=rejected.get("name", ""),
            ))
            logger.info(
                "[HITL] → hitl_node | rechazo | '%s' → ToolMessage sintético | cola=%d",
                rejected.get("name", ""), len(queue),
            )
        # Acción rechazada descartada: continuar con lo que queda en la cola
        all_remaining = queue
    else:
        # Entrada normal: combinar cola existente + pending nuevo (del planificador).
        # google_action_node ya limpió tool_calls_pending=[], así que en ese caso
        # pending=[] y solo avanza la cola. Desde el planificador, queue=[] y
        # pending=[A,B,C], lo que pobla la cola completa.
        all_remaining = queue + pending

    if not all_remaining:
        logger.info("[HITL] → hitl_node | sin más acciones en cola → fin del bucle")
        ret: dict = {"tool_calls_pending": [], "tool_calls_queue": [], "hitl_approved": None}
        if result_messages:
            ret["messages"] = result_messages
        return ret

    # Tomar la primera acción de la cola y poner el resto en tool_calls_queue
    current = dict(all_remaining[0])
    remaining_queue = all_remaining[1:]

    if "description" not in current:
        current["description"] = _format_action_description(
            current["name"], current.get("args", {})
        )

    needs_hitl = current.get("requires_approval", False)
    logger.info(
        "[HITL] → hitl_node | acción='%s' | requires_approval=%s | cola_restante=%d | user='%s'",
        current["name"], needs_hitl, len(remaining_queue), user_id,
    )

    ret = {
        "tool_calls_pending": [current],
        "tool_calls_queue": remaining_queue,
        # Auto-aprobar lecturas; para destructivas dejar None → hitl_wait_node
        "hitl_approved": True if not needs_hitl else None,
    }
    if result_messages:
        ret["messages"] = result_messages
    return ret


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
            # Detectar fallo de auth en herramientas Gmail (token Bearer expirado ~1h).
            # Si el error contiene "401", "auth", "token" o "unauthorized", intentar
            # refrescar el token y reintentar UNA vez con un cliente Gmail nuevo.
            _is_gmail = any(kw in call["name"].lower() for kw in ("gmail", "email", "mail"))
            _is_auth_error = any(kw in error_msg.lower() for kw in (
                "401", "unauthorized", "unauthenticated", "auth", "token", "invalid_grant",
            ))
            if _is_gmail and _is_auth_error:
                logger.warning(
                    "[GOOGLE] → google_action_node | tool='%s' | fallo de auth Gmail — "
                    "refrescando token y reintentando",
                    call["name"],
                )
                try:
                    from aetheris.mcp_tools.google_auth import get_google_access_token
                    from aetheris.config import get_settings
                    from langchain_mcp_adapters.client import MultiServerMCPClient
                    _settings = get_settings()
                    # Invalidar cache forzando llamada directa
                    import aetheris.mcp_tools.google_auth as _gauth
                    _gauth._cached_until = 0  # fuerza refresco en la siguiente llamada
                    fresh_token = get_google_access_token()
                    _fresh_client = MultiServerMCPClient({
                        "gmail": {
                            "transport": "http",
                            "url": _settings.gmail_mcp_url,
                            "headers": {"Authorization": f"Bearer {fresh_token}"},
                        }
                    })
                    _fresh_tools = await _fresh_client.get_tools()
                    _fresh_tool = next((t for t in _fresh_tools if t.name == call["name"]), None)
                    if _fresh_tool:
                        result = await _fresh_tool.ainvoke(exec_args)
                        summary = str(result)[:300]
                        result_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))
                        action_results.append({"ok": True, "name": call["name"], "summary": summary})
                        logger.info(
                            "[GOOGLE] → google_action_node | tool='%s' → completado tras refresco de token",
                            call["name"],
                        )
                        continue  # pasar al siguiente tool_call
                except Exception as retry_exc:
                    logger.error(
                        "[GOOGLE] → google_action_node | reintento Gmail fallido | %s", retry_exc
                    )
                    error_msg = f"Error de autenticación Gmail (reintento fallido): {retry_exc}"

            logger.error("[GOOGLE] → google_action_node | tool='%s' → fallido | %s", call["name"], error_msg)
            result_messages.append(ToolMessage(content=f"Error: {error_msg}", tool_call_id=tc_id))
            action_results.append({"ok": False, "name": call["name"], "error": error_msg})

    ok_count = sum(1 for r in action_results if r["ok"])
    iterations = state.get("google_action_iterations", 0) + 1
    logger.info(
        "[GOOGLE] → google_action_node | completado | ok=%d/%d | iteracion=%d",
        ok_count, len(pending), iterations,
    )
    return {
        "messages": result_messages,
        "tool_calls_pending": [],
        "hitl_approved": None,   # reset para que el siguiente hitl_node empiece limpio
        "action_results": action_results,
        "google_action_iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Nodo: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> dict:
    """
    Genera la respuesta final del asistente incorporando contexto RAG,
    resultados de herramientas y memoria del usuario.

    Pass-through cuando data_collection_required=True: hitl_node ya generó
    la pregunta de datos faltantes y la añadió al historial. El llm_node
    solo resetea el flag para que el siguiente turno funcione con normalidad.
    """
    if state.get("data_collection_required"):
        logger.info(
            "[LLM] → llm_node | data_collection_required=True → pass-through "
            "(hitl_node ya preguntó los datos faltantes)"
        )
        return {"data_collection_required": False}

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

    # Aplicar PII sanitation solo para el LLM — el historial original no se toca.
    # _sanitize_tool_calls también limpia AIMessages huérfanos del historial.
    _llm_hist = _apply_sanitized_input(list(state["messages"]), state.get("sanitized_user_input"))
    raw_messages = _sanitize_tool_calls(_llm_hist)
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
