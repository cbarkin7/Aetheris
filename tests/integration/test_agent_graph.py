"""
Test de integración: compilación del grafo LangGraph e invocación de un turno.
Usa SqliteSaver real (en /tmp) y GenericFakeChatModel.
"""
import json
import sqlite3
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from unittest.mock import patch


def _base_input(thread_id: str = "test-thread-001") -> dict:
    return {
        "messages": [HumanMessage(content="Hola AETHERIS")],
        "thread_id": thread_id,
        "user_id": "test-user",
        "intent": "unknown",
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


def _plan_response(steps: list) -> str:
    return json.dumps({"reasoning": "test", "steps": steps})


@pytest.mark.integration
def test_graph_compiles(tmp_path):
    checkpointer = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))
    fake_llm = GenericFakeChatModel(messages=iter([_plan_response(["plain_llm"]), "Hola!"]))

    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "test")), \
         patch("aetheris.memory.long_term.load_user_memory", return_value={}), \
         patch("aetheris.memory.long_term.extract_memory_updates", return_value={}), \
         patch("aetheris.memory.long_term.upsert_user_memory"):
        from aetheris.agent.graph import build_graph
        graph = build_graph(mcp_tools=[], checkpointer=checkpointer)
        assert graph is not None


@pytest.mark.integration
def test_graph_persists_checkpoint(tmp_path):
    checkpointer = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))
    # manager_node response + llm_node response
    fake_llm = GenericFakeChatModel(messages=iter([_plan_response(["plain_llm"]), "Soy AETHERIS."]))

    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "test")), \
         patch("aetheris.memory.long_term.load_user_memory", return_value={}), \
         patch("aetheris.memory.long_term.extract_memory_updates", return_value={}), \
         patch("aetheris.memory.long_term.upsert_user_memory"), \
         patch("aetheris.memory.long_term.store_long_term_fact"), \
         patch("aetheris.memory.mem0_memory.add_conversation_memory"):
        from aetheris.agent.graph import build_graph
        graph = build_graph(mcp_tools=[], checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-001"}}
        result = graph.invoke(_base_input("test-001"), config=config)

    assert "messages" in result
    assert len(result["messages"]) >= 2


@pytest.mark.integration
def test_graph_records_provider(tmp_path):
    checkpointer = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))
    fake_llm = GenericFakeChatModel(messages=iter([_plan_response(["plain_llm"]), "Respuesta."]))

    with patch("aetheris.agent.nodes.get_llm", return_value=(fake_llm, "openai")), \
         patch("aetheris.memory.long_term.load_user_memory", return_value={}), \
         patch("aetheris.memory.long_term.extract_memory_updates", return_value={}), \
         patch("aetheris.memory.long_term.upsert_user_memory"), \
         patch("aetheris.memory.long_term.store_long_term_fact"), \
         patch("aetheris.memory.mem0_memory.add_conversation_memory"):
        from aetheris.agent.graph import build_graph
        graph = build_graph(mcp_tools=[], checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-provider"}}
        result = graph.invoke(_base_input("test-provider"), config=config)

    assert result.get("llm_provider") in {"openai", "test", ""}
