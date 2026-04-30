"""
Integration test: MCP client configuration (no real subprocess spawned).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.integration
async def test_get_mcp_tools_no_keys_returns_empty(monkeypatch, override_settings):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "")
    from aetheris.config import get_settings
    get_settings.cache_clear()

    from aetheris.mcp_tools.client import get_mcp_tools
    clients, tools = await get_mcp_tools()
    assert clients == []
    assert tools == []
    get_settings.cache_clear()


@pytest.mark.integration
async def test_get_mcp_tools_calls_get_tools_directly(monkeypatch, override_settings):
    """Con la nueva API, get_mcp_tools llama a client.get_tools() directamente (no async with)."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "")
    from aetheris.config import get_settings
    get_settings.cache_clear()

    mock_tool = MagicMock()
    mock_tool.name = "tavily_search"
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    with patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=mock_client):
        from aetheris.mcp_tools.client import get_mcp_tools
        clients, tools = await get_mcp_tools()

    mock_client.get_tools.assert_called_once()
    assert len(clients) == 1
    assert tools == [mock_tool]
    get_settings.cache_clear()


@pytest.mark.integration
def test_tavily_server_config_contains_api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    from aetheris.config import get_settings
    get_settings.cache_clear()

    from aetheris.mcp_tools.tavily_tools import get_tavily_server_config
    config = get_tavily_server_config()
    assert config["env"]["TAVILY_API_KEY"] == "tvly-test-key"
    assert "npx" in config["command"]
    get_settings.cache_clear()


@pytest.mark.integration
def test_google_server_config_contains_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "refresh-token")
    from aetheris.config import get_settings
    get_settings.cache_clear()

    from aetheris.mcp_tools.google_tools import get_google_server_config
    config = get_google_server_config()
    assert config["env"]["GOOGLE_CLIENT_ID"] == "client-id"
    assert config["env"]["GOOGLE_REFRESH_TOKEN"] == "refresh-token"
    get_settings.cache_clear()
