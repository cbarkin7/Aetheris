"""FastAPI dependency injection."""
from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from aetheris.config import Settings, get_settings


def get_app_settings() -> Settings:
    return get_settings()


def get_compiled_graph(request: Request) -> CompiledStateGraph:
    """Retrieve the compiled LangGraph agent from app state."""
    return request.app.state.graph


def get_mcp_tools(request: Request) -> list:
    """Retrieve loaded MCP tools from app state."""
    return getattr(request.app.state, "mcp_tools", [])
