"""
Definición y compilación del StateGraph LangGraph para AETHERIS.

Flujo principal:
  START → input_guardrail → [bloqueado→llm | OK→load_memory→manager]
    manager → route_by_intent → {rag | web_search | google_planner | llm}
    rag/web_search → route_after_tool → {plan_dispatch→route_by_intent | llm}
    google_planner → route_after_planner → {hitl | llm}
    hitl → route_after_hitl_node → {hitl_wait | google_action | llm}
    hitl_wait → route_after_hitl → {google_action | hitl (rechazado+cola) | llm}
    google_action → route_after_google → {hitl (cola) | google_planner | llm}
    llm → output_guardrail → save_memory → END
"""
import functools
import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aetheris.agent.edges import (
    route_after_google,
    route_after_hitl,
    route_after_hitl_node,
    route_after_input_guardrail,
    route_after_planner,
    route_after_tool,
    route_by_intent,
)
from aetheris.agent.nodes import (
    google_action_node,
    google_planner_node,
    hitl_node,
    input_guardrail_node,
    llm_node,
    load_memory_node,
    manager_node,
    output_guardrail_node,
    plan_dispatch_node,
    rag_node,
    save_memory_node,
    web_search_node,
)
from aetheris.agent.state import AgentState
from aetheris.memory.checkpointer import create_async_checkpointer

logger = logging.getLogger(__name__)

_compiled_graph: CompiledStateGraph | None = None

# Destinos posibles desde route_by_intent (usado en dos aristas condicionales)
_INTENT_TARGETS = {
    "rag_node": "rag_node",
    "web_search_node": "web_search_node",
    "google_planner_node": "google_planner_node",
    "llm_node": "llm_node",
}


def build_graph(mcp_tools: list | None = None, checkpointer=None) -> CompiledStateGraph:
    """
    Construye y compila el grafo del agente AETHERIS.

    Args:
        mcp_tools: Herramientas MCP cargadas en el lifespan de FastAPI.
        checkpointer: Checkpointer LangGraph (SqliteSaver por defecto).

    Returns:
        StateGraph compilado listo para su invocación.
    """
    if checkpointer is None:
        import asyncio
        checkpointer = asyncio.get_event_loop().run_until_complete(create_async_checkpointer())

    tools = mcp_tools or []
    _web_search = functools.partial(web_search_node, mcp_tools=tools)
    _google_planner = functools.partial(google_planner_node, mcp_tools=tools)
    _google_action = functools.partial(google_action_node, mcp_tools=tools)

    builder = StateGraph(AgentState)

    # Registrar nodos
    builder.add_node("input_guardrail_node", input_guardrail_node)
    builder.add_node("load_memory_node", load_memory_node)
    builder.add_node("manager_node", manager_node)
    builder.add_node("plan_dispatch_node", plan_dispatch_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("web_search_node", _web_search)
    # google_planner_node: LLM con herramientas filtradas por dominio → tool_calls_pending
    builder.add_node("google_planner_node", _google_planner)
    # hitl_node: solo gestiona aprobación HITL (sin LLM, sin planificación)
    builder.add_node("hitl_node", hitl_node)
    # hitl_wait_node: punto de interrupción vacío para acciones destructivas.
    # El grafo pausa ANTES de ejecutarlo (interrupt_before).
    builder.add_node("hitl_wait_node", lambda state: {})
    builder.add_node("google_action_node", _google_action)
    builder.add_node("llm_node", llm_node)
    builder.add_node("output_guardrail_node", output_guardrail_node)
    builder.add_node("save_memory_node", save_memory_node)

    # Arista de entrada
    builder.add_edge(START, "input_guardrail_node")

    # Guardrail de entrada → memory o rechazo
    builder.add_conditional_edges(
        "input_guardrail_node",
        route_after_input_guardrail,
        {"load_memory_node": "load_memory_node", "llm_node": "llm_node"},
    )

    # Memoria → manager
    builder.add_edge("load_memory_node", "manager_node")

    # Manager → primera herramienta (o llm directo)
    builder.add_conditional_edges("manager_node", route_by_intent, _INTENT_TARGETS)

    # Después de herramienta RAG/web → continuar plan o ir a llm
    builder.add_conditional_edges(
        "rag_node",
        route_after_tool,
        {"plan_dispatch_node": "plan_dispatch_node", "llm_node": "llm_node"},
    )
    builder.add_conditional_edges(
        "web_search_node",
        route_after_tool,
        {"plan_dispatch_node": "plan_dispatch_node", "llm_node": "llm_node"},
    )

    # plan_dispatch → siguiente herramienta (ciclo controlado por execution_plan)
    builder.add_conditional_edges("plan_dispatch_node", route_by_intent, _INTENT_TARGETS)

    # google_planner_node: planifica acciones con herramientas filtradas por dominio.
    # Si hay tool_calls → hitl_node (gestión de aprobación).
    # Si LLM respondió con texto (tarea completa o faltan datos) → llm_node (pass-through).
    builder.add_conditional_edges(
        "google_planner_node",
        route_after_planner,
        {"hitl_node": "hitl_node", "llm_node": "llm_node"},
    )

    # hitl_node: decide si las acciones requieren aprobación humana o son auto-aprobadas.
    builder.add_conditional_edges(
        "hitl_node",
        route_after_hitl_node,
        {
            "hitl_wait_node": "hitl_wait_node",          # acciones destructivas → interrupt
            "google_action_node": "google_action_node",  # lecturas → auto-ejecutar
            "llm_node": "llm_node",                      # sin acciones → respuesta directa
        },
    )

    # hitl_wait_node: punto de interrupción real (interrupt_before).
    # Al reanudar, route_after_hitl evalúa hitl_approved y enruta.
    # Si el usuario rechaza y hay más acciones en cola → hitl_node (siguiente acción).
    builder.add_conditional_edges(
        "hitl_wait_node",
        route_after_hitl,
        {
            "google_action_node": "google_action_node",
            "hitl_node": "hitl_node",  # rechazado + cola no vacía → siguiente acción
            "llm_node": "llm_node",
        },
    )

    # google_action_node:
    # - Cola no vacía → hitl_node (siguiente acción uno a uno).
    # - Cola vacía → google_planner_node para encadenar pasos dependientes o PASO 0.
    builder.add_conditional_edges(
        "google_action_node",
        route_after_google,
        {
            "hitl_node": "hitl_node",              # más acciones en cola → siguiente
            "google_planner_node": "google_planner_node",
            "llm_node": "llm_node",
        },
    )

    # LLM → guardrail salida → guardar memoria → FIN
    builder.add_edge("llm_node", "output_guardrail_node")
    builder.add_edge("output_guardrail_node", "save_memory_node")
    builder.add_edge("save_memory_node", END)

    # interrupt_before=["hitl_wait_node"]: el grafo pausa SOLO cuando hay acciones
    # destructivas pendientes. Las lecturas se auto-ejecutan sin interrupciones.
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_wait_node"],
    )

    logger.info("Grafo AETHERIS compilado (herramientas MCP: %d)", len(tools))

    # Guardar el diagrama visual (PNG via Mermaid).
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        graph_img = Path("data/graph.png")
        graph_img.parent.mkdir(parents=True, exist_ok=True)
        graph_img.write_bytes(png_bytes)
        logger.info("Diagrama del grafo guardado en: %s", graph_img.resolve())
    except Exception:  # noqa: BLE001
        logger.debug("No se pudo generar el diagrama del grafo.")

    return graph


def get_graph(mcp_tools: list | None = None) -> CompiledStateGraph:
    """Devuelve el grafo compilado singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(mcp_tools=mcp_tools)
    return _compiled_graph
