import asyncio
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

env = os.environ.copy()
env["GOOGLE_OAUTH_CREDENTIALS"] = str(
    ROOT / "data" / "google" / "client_secret_aetheris.json"
)
env["GOOGLE_GMAIL_MCP_TOKEN_PATH"] = str(
    ROOT / "data" / "google" / ".gmail-token.json"
)

# 1. Arrancar Gmail MCP como servidor HTTP
process = subprocess.Popen(
    ["cmd", "/c", "npx", "-y", "@gongrzhe/server-gmail-mcp"],
    env=env,
)

time.sleep(5)

async def main():
    client = MultiServerMCPClient({
        "gmail": {
            "transport": "http",
            "url": "http://localhost:30000/mcp",
        }
    })

    tools = await client.get_tools()

    print("Herramientas Gmail:")
    for tool in tools:
        print("-", tool.name)

asyncio.run(main())

process.terminate()