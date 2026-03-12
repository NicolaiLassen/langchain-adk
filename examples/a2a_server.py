"""A2A server example - expose an agent over HTTP with SSE streaming.

Run with:
    cd agent-sdk
    uvicorn examples.a2a_server:app

Then call:
    curl -N -X POST http://localhost:8000/run \\
      -H "Content-Type: application/json" \\
      -d '{"message": "What is 2+2?"}'
"""

from __future__ import annotations

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.a2a.server import A2AServer
from langchain_adk.sessions.in_memory_session_service import InMemorySessionService

# --- Replace with a real LLM ---
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-5-haiku-latest")

# Uncomment and set your LLM, then comment out the raise:
raise NotImplementedError("Set llm= below with a real LangChain model.")

agent = LlmAgent(
    name="AssistantAgent",
    llm=llm,  # noqa: F821
    instructions="You are a helpful assistant.",
)

server = A2AServer(
    agent,
    session_service=InMemorySessionService(),
    app_name="demo",
)

app = server.as_fastapi_app()
