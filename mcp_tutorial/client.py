from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import asyncio
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

async def main():
    """Summary"""

    client = MultiServerMCPClient(

        {
            "calculator": {
                "command": "python",
                "args": ["mcp_tutorial/calculator.py"],
                "transport": "stdio",
            },

            "weather": {
                "url":"http://localhost:8000/mcp",
                "transport":"streamable_http",
            },
        }
    )

    tools = await client.get_tools()

    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_output_tokens=1024)

    agent = create_agent(model, tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather in New York and what is 123*456?"}]}
    )

    print(response)

asyncio.run(main())