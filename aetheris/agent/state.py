"""
Definición del estado del agente LangGraph de AETHERIS.
"""
from typing import Annotated, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Historial de conversación — add_messages gestiona la concatenación
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Identificadores de sesión
    thread_id: str
    user_id: str

    # Intención del paso actual (establecida por manager_node / plan_dispatch_node)
    intent: Literal["rag", "web_search", "google_action", "plain_llm", "unknown"]

    # Plan de ejecución multi-paso: pasos restantes después del actual
    execution_plan: list[str]

    # Fragmentos RAG recuperados: lista de {content, source, score}
    rag_context: list[dict]

    # Resultado de la búsqueda web (texto plano listo para inyectar en el prompt)
    web_context: str | None

    # Llamadas a herramientas MCP pendientes de aprobación HITL
    # Cada elemento: {name, args, description}
    tool_calls_pending: list[dict]

    # None = aún no preguntado | True = aprobado | False = rechazado
    hitl_approved: bool | None

    # Resultados de las acciones ejecutadas en google_action_node
    # Cada elemento: {ok: bool, name: str, summary?: str, error?: str}
    action_results: list[dict]

    # Preferencias del usuario cargadas desde memoria a largo plazo
    user_memory: dict

    # Resultado del guardrail de entrada
    guardrail_passed: bool | None

    # Violaciones detectadas por guardrails
    guardrail_violations: list[str]

    # Mapa reversible de redacciones PII: {placeholder: valor_original}
    # Generado por input_guardrail_node, consumido por google_action_node
    # para restaurar emails, teléfonos, etc. antes de invocar las tools de Google.
    pii_map: dict[str, str]

    # Proveedor LLM activo (para trazabilidad del fallback)
    llm_provider: str

    # Error irrecuperable, si ocurre
    error: str | None
