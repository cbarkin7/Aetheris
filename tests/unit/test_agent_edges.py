"""Tests unitarios para las funciones de enrutamiento condicional — funciones puras."""
import pytest

from aetheris.agent.edges import (
    route_after_google,
    route_after_hitl,
    route_after_hitl_node,
    route_after_input_guardrail,
    route_after_planner,
    route_after_tool,
    route_by_intent,
)


# ---------------------------------------------------------------------------
# route_after_input_guardrail
# ---------------------------------------------------------------------------

def test_route_after_input_guardrail_blocked(base_agent_state):
    base_agent_state["guardrail_passed"] = False
    assert route_after_input_guardrail(base_agent_state) == "llm_node"


def test_route_after_input_guardrail_ok(base_agent_state):
    base_agent_state["guardrail_passed"] = True
    assert route_after_input_guardrail(base_agent_state) == "load_memory_node"


def test_route_after_input_guardrail_none(base_agent_state):
    base_agent_state["guardrail_passed"] = None
    assert route_after_input_guardrail(base_agent_state) == "load_memory_node"


# ---------------------------------------------------------------------------
# route_by_intent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent,expected", [
    ("rag", "rag_node"),
    ("web_search", "web_search_node"),
    ("google_action", "google_planner_node"),
    ("plain_llm", "llm_node"),
    ("unknown", "llm_node"),
    ("garbage", "llm_node"),
])
def test_route_by_intent(intent, expected, base_agent_state):
    base_agent_state["intent"] = intent
    assert route_by_intent(base_agent_state) == expected


# ---------------------------------------------------------------------------
# route_after_tool
# ---------------------------------------------------------------------------

def test_route_after_tool_with_remaining_plan(base_agent_state):
    """Si hay pasos en el plan, ir a plan_dispatch_node."""
    base_agent_state["execution_plan"] = ["web_search"]
    assert route_after_tool(base_agent_state) == "plan_dispatch_node"


def test_route_after_tool_empty_plan(base_agent_state):
    """Si el plan está vacío, ir a llm_node."""
    base_agent_state["execution_plan"] = []
    assert route_after_tool(base_agent_state) == "llm_node"


def test_route_after_tool_multiple_steps(base_agent_state):
    """Siempre va a plan_dispatch si hay cualquier paso restante."""
    base_agent_state["execution_plan"] = ["rag", "web_search"]
    assert route_after_tool(base_agent_state) == "plan_dispatch_node"


# ---------------------------------------------------------------------------
# route_after_hitl
# ---------------------------------------------------------------------------

def test_route_after_hitl_approved(base_agent_state):
    base_agent_state["hitl_approved"] = True
    assert route_after_hitl(base_agent_state) == "google_action_node"


def test_route_after_hitl_rejected_queue_empty(base_agent_state):
    """Rechazado sin más acciones en cola → llm_node (generar resumen)."""
    base_agent_state["hitl_approved"] = False
    base_agent_state["tool_calls_queue"] = []
    assert route_after_hitl(base_agent_state) == "llm_node"


def test_route_after_hitl_rejected_queue_not_empty(base_agent_state):
    """Rechazado con más acciones en cola → hitl_node (siguiente acción)."""
    base_agent_state["hitl_approved"] = False
    base_agent_state["tool_calls_queue"] = [{"name": "send_email", "args": {}}]
    assert route_after_hitl(base_agent_state) == "hitl_node"


def test_route_after_hitl_none(base_agent_state):
    base_agent_state["hitl_approved"] = None
    assert route_after_hitl(base_agent_state) == "llm_node"


# ---------------------------------------------------------------------------
# route_after_planner
# ---------------------------------------------------------------------------

def test_route_after_planner_data_collection(base_agent_state):
    """data_collection_required → llm_node (pass-through)."""
    base_agent_state["data_collection_required"] = True
    base_agent_state["tool_calls_pending"] = []
    assert route_after_planner(base_agent_state) == "llm_node"


def test_route_after_planner_with_pending(base_agent_state):
    """Acciones pendientes → hitl_node."""
    base_agent_state["data_collection_required"] = False
    base_agent_state["tool_calls_pending"] = [{"name": "create-event", "args": {}}]
    assert route_after_planner(base_agent_state) == "hitl_node"


def test_route_after_planner_no_pending(base_agent_state):
    """Sin acciones → llm_node."""
    base_agent_state["data_collection_required"] = False
    base_agent_state["tool_calls_pending"] = []
    assert route_after_planner(base_agent_state) == "llm_node"


# ---------------------------------------------------------------------------
# route_after_hitl_node
# ---------------------------------------------------------------------------

def test_route_after_hitl_node_auto_approve(base_agent_state):
    """Lecturas auto-aprobadas → google_action_node."""
    base_agent_state["tool_calls_pending"] = [{"name": "listFolder", "args": {}}]
    base_agent_state["hitl_approved"] = True
    assert route_after_hitl_node(base_agent_state) == "google_action_node"


def test_route_after_hitl_node_destructive(base_agent_state):
    """Acciones destructivas → hitl_wait_node."""
    base_agent_state["tool_calls_pending"] = [{"name": "create-event", "args": {}}]
    base_agent_state["hitl_approved"] = None
    assert route_after_hitl_node(base_agent_state) == "hitl_wait_node"


def test_route_after_hitl_node_no_pending(base_agent_state):
    """Sin acciones pendientes → llm_node."""
    base_agent_state["tool_calls_pending"] = []
    assert route_after_hitl_node(base_agent_state) == "llm_node"


# ---------------------------------------------------------------------------
# route_after_google
# ---------------------------------------------------------------------------

def test_route_after_google_queue_not_empty(base_agent_state):
    """Cola no vacía → hitl_node (siguiente acción)."""
    base_agent_state["google_action_iterations"] = 1
    base_agent_state["tool_calls_queue"] = [{"name": "send_email", "args": {}}]
    assert route_after_google(base_agent_state) == "hitl_node"


def test_route_after_google_queue_empty(base_agent_state):
    """Cola vacía → google_planner_node (replanning o PASO 0)."""
    base_agent_state["google_action_iterations"] = 1
    base_agent_state["tool_calls_queue"] = []
    assert route_after_google(base_agent_state) == "google_planner_node"


def test_route_after_google_max_iterations(base_agent_state):
    """MAX_GOOGLE_ITERATIONS alcanzado → llm_node."""
    base_agent_state["google_action_iterations"] = 6
    base_agent_state["tool_calls_queue"] = [{"name": "create-event", "args": {}}]
    assert route_after_google(base_agent_state) == "llm_node"
