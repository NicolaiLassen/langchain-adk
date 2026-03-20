"""Example tools for the pipeline composer example."""


async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles about the topic."
