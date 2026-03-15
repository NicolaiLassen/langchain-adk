"""Memory agent example - persistent memory across sessions.

Demonstrates:
  - InMemoryMemoryStore for storing and recalling facts
  - Session management with InMemorySessionService
  - Agent remembers information from previous conversations
"""

from __future__ import annotations

import asyncio

from langchain_adk import LlmAgent, InvocationContext, InMemorySessionService
from langchain_adk.events.event import Event, EventType
from langchain_adk.memory.in_memory_store import InMemoryMemoryStore
from langchain_adk.memory.memory import Memory


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # --- Set up memory and sessions ---
    memory_store = InMemoryMemoryStore()
    session_service = InMemorySessionService()

    # Pre-populate some memories (simulating past conversations)
    key = ("my-app", "user-42")
    memory_store._store[key] = [
        Memory(content="User's name is Alice.", author="agent"),
        Memory(content="Alice is a backend engineer working on microservices.", author="agent"),
        Memory(content="Alice prefers Python over JavaScript.", author="agent"),
        Memory(content="Alice's team is migrating from REST to gRPC.", author="agent"),
    ]

    # Search for relevant memories
    result = await memory_store.search_memory(
        app_name="my-app",
        user_id="user-42",
        query="alice",
    )

    # Build context from memories
    memory_context = "\n".join(
        f"- {m.content}" for m in result.memories
    )

    agent = LlmAgent(
        name="MemoryAgent",
        llm=llm,  # noqa: F821
        instructions=(
            "You are a helpful assistant with memory of past conversations.\n\n"
            "Here is what you remember about this user:\n"
            f"{memory_context}\n\n"
            "Use this context to personalize your responses. "
            "Reference what you know when relevant."
        ),
    )

    # --- Session 1: Ask a question that uses memory ---
    session = await session_service.create_session(
        app_name="my-app",
        user_id="user-42",
    )

    ctx = InvocationContext(
        session_id=session.id,
        user_id="user-42",
        app_name="my-app",
        agent_name=agent.name,
    )

    print("Session 1: Asking a personalized question")
    print("=" * 50)

    async for event in agent.astream(
        "What programming language should I use for my new service?",
        ctx=ctx,
    ):
        if event.is_final_response():
            print(f"\n[ANSWER] {event.text}")

    # Save session to memory for future recall
    session.events = []  # In a real app, events accumulate automatically
    await memory_store.add_session_to_memory(session)

    print(f"\n\nMemories stored: {len(memory_store._store[key])}")


if __name__ == "__main__":
    asyncio.run(main())
