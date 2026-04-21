"""Script puntual para verificar la conexion con Tavily MCP."""
import asyncio
import sys
import io
from pathlib import Path

# Forzar UTF-8 en stdout (Windows cp1252 no soporta los simbolos)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from aetheris.mcp.tavily_tools import get_tavily_server_config
from langchain_mcp_adapters.client import MultiServerMCPClient


async def test_tools():
    config = get_tavily_server_config()
    print("== Configuracion Tavily ==")
    print(f"  command : {config['command']}")
    print(f"  args    : {config['args']}")
    key = config['env'].get('TAVILY_API_KEY', '')
    print(f"  API key : {key[:12]}... (len={len(key)})")

    print("\n== Conectando al servidor MCP... ==")
    client = MultiServerMCPClient({"tavily": config})
    tools = await client.get_tools()
    print(f"  Tools disponibles: {len(tools)}")
    for t in tools:
        print(f"    * {t.name}: {t.description[:70]}")
    return tools


async def test_search(tools):
    search_tool = next((t for t in tools if "search" in t.name.lower()), None)
    if not search_tool:
        print("\nWARN: No se encontro herramienta de busqueda.")
        return

    print(f"\n== Prueba de busqueda con '{search_tool.name}' ==")
    result = await search_tool.ainvoke({"query": "LangGraph agent architecture 2025"})
    preview = str(result)[:400]
    print(f"  Resultado:\n{preview}")
    print("\nOK - Tavily MCP funciona correctamente.")


async def main():
    try:
        tools = await test_tools()
        if tools:
            await test_search(tools)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
