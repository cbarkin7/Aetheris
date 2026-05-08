"""
Test funcional: Gmail MCP Python nativo — invoca search_emails en tiempo real.

Verifica el ciclo completo: MCP stdio → gmail_mcp_server.py → Gmail REST API.
Requiere token OAuth2 válido en data/google/.gmail-token.json.

Ejecución manual:
    python tests/functional_google/test_10_mcp_gmail_call.py
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
    print(f"\nTools Gmail ({len(tools)}):")
    for t in tools:
        print(f"  - {t.name}")

    search_tool = next((t for t in tools if t.name == "search_emails"), None)
    if not search_tool:
        print("\n✗ Herramienta 'search_emails' no encontrada.")
        sys.exit(1)

    print(f"\nInvocando: {search_tool.name} (query='in:inbox', maxResults=3) ...")
    result = await search_tool.ainvoke({
        "query": "in:inbox",
        "maxResults": 3,
    })

    print("\nResultado:")
    print(result)
    print("\n✓ Invocación completada correctamente.")


asyncio.run(main())
