"""MCP client - thin wrapper around FastMCP.

Uses a fresh session per call. Suitable for request-scoped usage.
For long-lived connections, extend with a shared client lifecycle.
"""

from __future__ import annotations

from typing import Any


class MCPClient:
    """Async wrapper around the FastMCP Client.

    Parameters
    ----------
    url : str
        The HTTP URL of the MCP server
        (e.g. "http://localhost:8001/mcp").
    """

    def __init__(self, url: str) -> None:
        self._url = url

    async def list_tools(self) -> list[Any]:
        """Return the list of tools exposed by the MCP server.

        Returns
        -------
        list[Any]
            The tool definitions returned by the MCP server.
        """
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        async with Client(StreamableHttpTransport(self._url)) as client:
            return await client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name with the given arguments.

        Parameters
        ----------
        name : str
            The MCP tool name.
        arguments : dict[str, Any]
            Key-value arguments for the tool.

        Returns
        -------
        Any
            Raw result from the MCP server (list of content items).
        """
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        async with Client(StreamableHttpTransport(self._url)) as client:
            return await client.call_tool(name, arguments)
