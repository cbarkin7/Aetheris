"""
Funciones de enrutamiento condicional para el grafo LangGraph de AETHERIS.
Cada función recibe el AgentState actual y devuelve el nombre del siguiente nodo.
"""
from aetheris.agent.state import AgentState

_TOOL_NODE_MAP = {
    "rag": "rag_node",
    "web_search": "web_search_node",
    "google_action": "google_planner_node",
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
    Después de ejecutar una herramienta (rag_node o web_search_node):

    1. Si quedan pasos en execution_plan → plan_dispatch_node (continúa el plan).
    2. En cualquier otro caso → llm_node.

    No hay fallback automático RAG→web: la búsqueda web solo se activa cuando
    el usuario la pide explícitamente y el manager la incluye en el plan.
    """
    if state.get("execution_plan"):
        return "plan_dispatch_node"
    return "llm_node"


def route_after_planner(state: AgentState) -> str:
    """
    Enruta desde google_planner_node:
    - data_collection_required=True (LLM pidió datos o informó de tarea completa)
      → llm_node (pass-through: el AIMessage ya está en el historial).
    - tool_calls_pending no vacío → hitl_node (gestión de aprobación).
    - Sin acciones → llm_node.
    """
    if state.get("data_collection_required"):
        return "llm_node"
    if state.get("tool_calls_pending"):
        return "hitl_node"
    return "llm_node"


def route_after_hitl_node(state: AgentState) -> str:
    """
    Enruta desde hitl_node:
    - Acciones de lectura auto-aprobadas (hitl_approved=True)
      → google_action_node directamente (sin interrupción).
    - Acciones destructivas pendientes (hitl_approved=None)
      → hitl_wait_node (interrupt_before: pausa hasta que el usuario apruebe/rechace).
    - Sin acciones pendientes → llm_node.
    """
    if state.get("tool_calls_pending"):
        if state.get("hitl_approved") is True:
            return "google_action_node"
        return "hitl_wait_node"
    return "llm_node"


def route_after_hitl(state: AgentState) -> str:
    """
    Enruta desde hitl_wait_node según la aprobación del usuario.

    - Aprobado → google_action_node (ejecutar la acción actual).
    - Rechazado + cola no vacía → hitl_node (siguiente acción en la cola).
    - Rechazado + cola vacía → llm_node (generar resumen de lo ejecutado/rechazado).
    """
    if state.get("hitl_approved") is True:
        return "google_action_node"
    # Rechazado: continuar con la siguiente acción en cola, si la hay
    if state.get("tool_calls_queue"):
        return "hitl_node"
    return "llm_node"


def route_after_google(state: AgentState) -> str:
    """
    Después de google_action_node:

    1. Si quedan acciones en tool_calls_queue → hitl_node (siguiente acción uno a uno).
    2. Cola vacía → google_planner_node para encadenar acciones dependientes
       (buscar carpeta → crear carpeta → crear doc) o detectar tarea completa (PASO 0).
    3. Si se supera MAX_GOOGLE_ITERATIONS → llm_node (corta bucles infinitos).

    google_planner_node detecta el fin del bucle: cuando todas las acciones
    solicitadas aparecen en ToolMessages de éxito, genera un texto de resumen
    (sin tool_calls) → data_collection_required=True → route_after_planner
    → llm_node (pass-through) → END.
    """
    MAX_GOOGLE_ITERATIONS = 6
    iterations = state.get("google_action_iterations", 0)
    if iterations >= MAX_GOOGLE_ITERATIONS:
        import logging
        logging.getLogger(__name__).warning(
            "[EDGES] → route_after_google | MAX_GOOGLE_ITERATIONS (%d) alcanzado → llm_node",
            MAX_GOOGLE_ITERATIONS,
        )
        return "llm_node"
    # Acciones pendientes en la cola → procesarlas de una en una
    if state.get("tool_calls_queue"):
        return "hitl_node"
    # Cola vacía → replanning
    return "google_planner_node"
