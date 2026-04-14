"""Tests unitarios para las funciones de enrutamiento condicional — funciones puras."""
import pytest

from aetheris.agent.edges import (
    route_after_hitl,
    route_after_input_guardrail,
    route_after_tool,
    route_after_llm,
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
    ("google_action", "hitl_node"),
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


def test_route_after_hitl_rejected(base_agent_state):
    base_agent_state["hitl_approved"] = False
    assert route_after_hitl(base_agent_state) == "llm_node"


def test_route_after_hitl_none(base_agent_state):
    base_agent_state["hitl_approved"] = None
    assert route_after_hitl(base_agent_state) == "llm_node"


# ---------------------------------------------------------------------------
# route_after_llm
# ---------------------------------------------------------------------------

def test_route_after_llm_goes_to_output_guardrail(base_agent_state):
    assert route_after_llm(base_agent_state) == "output_guardrail_node"
