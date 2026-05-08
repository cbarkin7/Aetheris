"""
Test funcional: Gmail MCP Python nativo — lista herramientas disponibles.

Verifica que gmail_mcp_server.py arranca correctamente como servidor stdio
y expone las 7 herramientas esperadas.

Ejecución manual:
    python tests/functional_google/test_09_mcp_gmail_tools.py
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
GMAIL_MCP_SERVER = str(ROOT / "aetheris" / "mcp_tools" / "gmail_mcp_server.py")
GMAIL_TOKEN_PATH = str(ROOT / "data" / "google" / ".gmail-token.json")
CLIENT_SECRET_PATH = str(ROOT / "data" / "google" / "client_secret_aetheris.json")

EXPECTED_TOOLS = {
    "list_emails", "get_email", "search_emails",
    "send_email", "create_draft", "delete_email", "reply_to_email",
}


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        "gmail": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [GMAIL_MCP_SERVER],
            "env": {
                "GMAIL_TOKEN_PATH": GMAIL_TOKEN_PATH,
                "GMAIL_CLIENT_SECRET_PATH": CLIENT_SECRET_PATH,
            },
        }
    })

    tools = await client.get_tools()
    tool_names = {t.name for t in tools}

    print(f"\nHerramientas Gmail MCP ({len(tools)}):")
    for t in tools:
        print(f"  ✓ {t.name}")

    missing = EXPECTED_TOOLS - tool_names
    if missing:
        print(f"\n✗ Herramientas faltantes: {missing}")
        sys.exit(1)

    print(f"\n✓ Todas las {len(EXPECTED_TOOLS)} herramientas esperadas están disponibles.")


asyncio.run(main())
