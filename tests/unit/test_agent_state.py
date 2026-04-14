"""Tests unitarios para la estructura AgentState y el reducer add_messages."""
from langchain_core.messages import AIMessage, HumanMessage

from aetheris.agent.state import AgentState


def test_agent_state_has_all_required_keys():
    keys = AgentState.__annotations__
    for key in (
        "messages", "thread_id", "user_id", "intent", "rag_context",
        "tool_calls_pending", "hitl_approved", "user_memory", "error",
        "guardrail_passed", "guardrail_violations", "llm_provider",
        "execution_plan",
    ):
        assert key in keys, f"Campo '{key}' ausente en AgentState"


def test_add_messages_reducer_appends():
    from langgraph.graph.message import add_messages
    msgs1 = [HumanMessage(content="hola")]
    msgs2 = [AIMessage(content="mundo")]
    result = add_messages(msgs1, msgs2)
    assert len(result) == 2
    assert result[0].content == "hola"
    assert result[1].content == "mundo"


def test_state_construction_with_all_fields():
    state: AgentState = {  # type: ignore
        "messages": [],
        "thread_id": "t1",
        "user_id": "u1",
        "intent": "plain_llm",
        "rag_context": [],
        "tool_calls_pending": [],
        "hitl_approved": None,
        "user_memory": {},
        "guardrail_passed": None,
        "guardrail_violations": [],
        "llm_provider": "",
        "execution_plan": [],
        "error": None,
    }
    assert state["intent"] == "plain_llm"
    assert state["execution_plan"] == []


def test_execution_plan_accepts_list_of_intents():
    state: AgentState = {  # type: ignore
        "messages": [HumanMessage(content="test")],
        "thread_id": "t1", "user_id": "u1",
        "intent": "rag",
        "rag_context": [], "tool_calls_pending": [],
        "hitl_approved": None, "user_memory": {},
        "guardrail_passed": True, "guardrail_violations": [],
        "llm_provider": "openai",
        "execution_plan": ["web_search"],
        "error": None,
    }
    assert state["execution_plan"] == ["web_search"]
    assert state["llm_provider"] == "openai"
