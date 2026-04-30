import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

env = os.environ.copy()
env["GOOGLE_OAUTH_CREDENTIALS"] = str(
    ROOT / "data" / "google" / "client_secret_aetheris.json"
)
env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(
    ROOT / "data" / "google" / ".calendar-token.json"
)

async def main():
    client = MultiServerMCPClient({
        "calendar": {
            "transport": "stdio",
            "command": "cmd",
            "args": ["/c", "npx", "-y", "@cocal/google-calendar-mcp"],
            "env": env,
        }
    })

    tools = await client.get_tools()

    print("Herramientas encontradas:")
    for tool in tools:
        print("-", tool.name)

asyncio.run(main())