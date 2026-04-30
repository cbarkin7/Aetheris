import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

access_token = os.getenv("GOOGLE_ACCESS_TOKEN")

async def main():
    client = MultiServerMCPClient({
        "gmail": {
            "transport": "http",
            "url": "http://localhost:30000/mcp",
            "headers": {
                "Authorization": f"Bearer {access_token}"
            }
        }
    })

    tools = await client.get_tools()

    print("Tools Gmail:")
    for t in tools:
        print("-", t.name)

    tool = next(t for t in tools if t.name == "search_emails")

    print("Invocando:", tool.name)

    result = await tool.ainvoke({
        "query": "in:inbox",
        "max_results": 5
    })

    print(result)

asyncio.run(main())