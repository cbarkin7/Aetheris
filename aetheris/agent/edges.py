"""
Funciones de enrutamiento condicional para el grafo LangGraph de AETHERIS.
Cada función recibe el AgentState actual y devuelve el nombre del siguiente nodo.
"""
from aetheris.agent.state import AgentState

_TOOL_NODE_MAP = {
    "rag": "rag_node",
    "web_search": "web_search_node",
    "google_action": "hitl_node",
    "plain_llm": "llm_node",
    "unknown": "llm_node",
}


def route_after_input_guardrail(state: AgentState) -> str:
    """Si el guardrail bloqueó la entrada, ir a llm_node (rechazo). Si no, cargar memoria."""
    if state.get("guardrail_passed") is False:
        return "llm_node"
    return "load_memory_node"


def route_by_intent(state: AgentState) -> str:
    """Enruta desde manager_node o plan_dispatch_node al nodo de herramienta adecuado."""
    intent = state.get("intent", "plain_llm")
    return _TOOL_NODE_MAP.get(intent, "llm_node")


def route_after_tool(state: AgentState) -> str:
    """
    Después de ejecutar una herramienta (rag, web_search), comprueba si
    quedan pasos en execution_plan. Si sí, va a plan_dispatch_node para
    continuar el plan; si no, va directamente a llm_node.
    """
    if state.get("execution_plan"):
        return "plan_dispatch_node"
    return "llm_node"


def route_after_hitl(state: AgentState) -> str:
    """Enruta desde hitl_node según la aprobación del usuario."""
    if state.get("hitl_approved") is True:
        return "google_action_node"
    return "llm_node"


def route_after_llm(state: AgentState) -> str:
    """Después de llm_node, siempre ir al guardrail de salida."""
    return "output_guardrail_node"
