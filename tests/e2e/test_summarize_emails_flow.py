"""E2E: Flujo de resumen de correos — Gmail → LLM → resumen estructurado."""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _plan(steps):
    return json.dumps({"reasoning": "test", "steps": steps})


@pytest.mark.e2e
def test_email_summary_routes_to_google_action(base_agent_state):
    base_agent_state["messages"] = [HumanMessage(content="Resume mis últimos 5 correos de Gmail")]
    fake_llm = GenericFakeChatModel(messages=iter([_plan(["google_action"])]))
    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "test")):
        from aetheris.agent.nodes import manager_node
        result = manager_node(base_agent_state)
    assert result["intent"] == "google_action"


@pytest.mark.e2e
def test_gmail_tool_result_fed_to_llm(base_agent_state):
    email_content = "Email 1: Reunión a las 14h. Email 2: Factura adjunta. Email 3: Proyecto aprobado."
    base_agent_state["messages"] = [
        HumanMessage(content="Resume mi bandeja"),
        ToolMessage(content=email_content, tool_call_id="gmail_list"),
    ]
    fake_llm = GenericFakeChatModel(messages=iter(["Resumen: Tienes una reunión, una factura y un proyecto aprobado."]))
    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "test")):
        from aetheris.agent.nodes import llm_node
        result = llm_node(base_agent_state)
    assert len(result["messages"]) == 1 and len(result["messages"][0].content) > 0


@pytest.mark.e2e
def test_google_action_node_executes_approved_tool(base_agent_state):
    base_agent_state["tool_calls_pending"] = [{"name": "list_emails", "args": {"max_results": 5}}]
    mock_tool = MagicMock()
    mock_tool.name = "list_emails"
    mock_tool.invoke.return_value = "Email 1: Hola. Email 2: Mundo."
    with patch("aetheris.agent.nodes.get_llm"):
        from aetheris.agent.nodes import google_action_node
        result = google_action_node(base_agent_state, mcp_tools=[mock_tool])
    assert len(result.get("messages", [])) == 1
    assert "Hola" in result["messages"][0].content
    assert result["tool_calls_pending"] == []


@pytest.mark.e2e
def test_provider_recorded_in_llm_response(base_agent_state):
    base_agent_state["messages"] = [HumanMessage(content="¿Qué tienes en bandeja?")]
    fake_llm = GenericFakeChatModel(messages=iter(["No hay mensajes nuevos."]))
    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "bedrock")):
        from aetheris.agent.nodes import llm_node
        result = llm_node(base_agent_state)
    assert result.get("llm_provider") == "bedrock"
