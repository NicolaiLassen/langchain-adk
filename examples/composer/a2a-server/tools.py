"""Example tools for the A2A server."""


async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found relevant information about the topic."


async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22C."
