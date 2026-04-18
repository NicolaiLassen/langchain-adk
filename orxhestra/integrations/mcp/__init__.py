"""Model Context Protocol (MCP) integration.

Two thin wrappers over FastMCP:

- :class:`MCPClient` — connects to an MCP server (HTTP URL or
  in-memory FastMCP instance) and forwards ``list_tools``,
  ``call_tool``, ``list_resources``, and ``read_resource`` calls.
- :class:`MCPToolAdapter` — consumes an :class:`MCPClient` and
  returns one :class:`~langchain_core.tools.BaseTool` per tool
  exposed by the server.  The Pydantic input model is generated from
  each tool's JSON Schema.

Requires the ``mcp`` extra::

    pip install 'orxhestra[mcp]'

Examples
--------
>>> from orxhestra.integrations.mcp import MCPClient, MCPToolAdapter
>>> client = MCPClient("http://localhost:8001/mcp")
>>> adapter = MCPToolAdapter(client)
>>> tools = await adapter.load_tools()
>>> agent = LlmAgent("agent", model=model, tools=tools)
"""

from orxhestra.integrations.mcp.adapter import MCPToolAdapter
from orxhestra.integrations.mcp.client import MCPClient

__all__ = ["MCPClient", "MCPToolAdapter"]
