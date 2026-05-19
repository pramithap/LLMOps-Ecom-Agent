from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Return a (stub) weather report for the given location."""
    return f"weather in {location} is rainy"

if __name__=="__main__":
    mcp.run(transport="streamable-http")