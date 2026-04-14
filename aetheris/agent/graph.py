"""
Definición y compilación del StateGraph LangGraph para AETHERIS.

Flujo principal:
  START → input_guardrail → [bloqueado→llm | OK→load_memory→manager]
    manager → route_by_intent → {rag | web_search | hitl | llm}
    rag/web_search → route_after_tool → {plan_dispatch→route_by_intent | llm}
    hitl → route_after_hitl → {google_action | llm}
    llm → output_guardrail → save_memory → END
"""
import functools
import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aetheris.agent.edges import (
    route_after_hitl,
    route_after_input_guardrail,
    route_after_tool,
    route_by_intent,
)
from aetheris.agent.nodes import (
    google_action_node,
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
    "hitl_node": "hitl_node",
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
    _hitl = functools.partial(hitl_node, mcp_tools=tools)
    _google_action = functools.partial(google_action_node, mcp_tools=tools)

    builder = StateGraph(AgentState)

    # Registrar nodos
    builder.add_node("input_guardrail_node", input_guardrail_node)
    builder.add_node("load_memory_node", load_memory_node)
    builder.add_node("manager_node", manager_node)
    builder.add_node("plan_dispatch_node", plan_dispatch_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("web_search_node", _web_search)
    builder.add_node("hitl_node", _hitl)
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

    # Después de herramienta → continuar plan o ir a llm
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

    # HITL
    builder.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {"google_action_node": "google_action_node", "llm_node": "llm_node"},
    )
    builder.add_edge("google_action_node", "llm_node")

    # LLM → guardrail salida → guardar memoria → FIN
    builder.add_edge("llm_node", "output_guardrail_node")
    builder.add_edge("output_guardrail_node", "save_memory_node")
    builder.add_edge("save_memory_node", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_node"],
    )

    logger.info("Grafo AETHERIS compilado (herramientas MCP: %d)", len(tools))
    return graph


def get_graph(mcp_tools: list | None = None) -> CompiledStateGraph:
    """Devuelve el grafo compilado singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(mcp_tools=mcp_tools)
    return _compiled_graph
