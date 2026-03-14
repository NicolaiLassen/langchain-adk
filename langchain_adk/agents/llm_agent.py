"""LlmAgent - the primary LangChain-powered agent.

Implements a manual tool-call loop using LangChain's BaseChatModel.
No LangGraph - orchestration is pure Python async generators.

The loop:
  1. Build system prompt from instructions (or instruction provider)
  2. If a planner is attached, append its planning instruction
  3. Build an LlmRequest and call llm.bind_tools(tools).ainvoke(messages)
     (or astream if streaming_mode=SSE)
  4. Wrap the AIMessage in LlmResponse
  5. If response has tool_calls -> execute each -> append ToolMessages -> loop
  6. Yield typed events throughout
  7. Repeat until no tool calls or max_iterations
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import reduce
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import (
    ErrorEvent,
    Event,
    EventType,
    FinalAnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from langchain_adk.events.event_actions import EventActions
from langchain_adk.models.llm_request import LlmRequest
from langchain_adk.models.llm_response import LlmResponse
from langchain_adk.models.part import Content, DataPart, TextPart
from langchain_adk.tools.exit_loop import EXIT_LOOP_SENTINEL
from langchain_adk.tools.transfer_tool import TRANSFER_SENTINEL

if TYPE_CHECKING:
    from langchain_adk.planners.base_planner import BasePlanner

# Type alias for instruction providers - either a static string or a callable
# that receives the current InvocationContext and returns a string.
InstructionProvider = str | Callable[[InvocationContext], str | Awaitable[str]]

_DEFAULT_INSTRUCTIONS = """\
You are a helpful assistant. Answer the user's questions clearly and concisely.
When you have enough information to answer, provide a direct response.
Only use tools when necessary to complete the task.
"""


class LlmAgent(BaseAgent):
    """LangChain-powered agent with a manual tool-call loop.

    Uses any LangChain ``BaseChatModel`` as the LLM backend. Supports
    static or dynamic system instructions, arbitrary LangChain tools,
    before/after callbacks at the model and tool level, an optional planner
    for per-turn planning instructions, and SSE streaming.

    Attributes
    ----------
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool]
        Tools available to the agent.
    instructions : str or callable
        System prompt string or callable returning one.
    planner : BasePlanner, optional
        Planner that injects planning instructions before each LLM call.
    output_schema : type, optional
        Optional Pydantic model for structured final output.
    max_iterations : int
        Maximum tool-call loop iterations before stopping.
    before_model_callback : callable, optional
        Called with ``(ctx, request: LlmRequest)`` before each LLM call.
    after_model_callback : callable, optional
        Called with ``(ctx, response: LlmResponse)`` after each LLM call.
    on_model_error_callback : callable, optional
        Called with ``(ctx, request: LlmRequest, exception)`` when an LLM
        call raises. Return a ``LlmResponse`` to recover, or ``None`` to
        yield an ErrorEvent.
    before_tool_callback : callable, optional
        Called with ``(ctx, tool_name, tool_args)`` before each tool execution.
    after_tool_callback : callable, optional
        Called with ``(ctx, tool_name, result)`` after each tool execution.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        instructions: InstructionProvider = _DEFAULT_INSTRUCTIONS,
        description: str = "",
        planner: BasePlanner | None = None,
        output_schema: type | None = None,
        max_iterations: int = 10,
        before_model_callback: (
            Callable[[InvocationContext, LlmRequest], Awaitable[None]] | None
        ) = None,
        after_model_callback: (
            Callable[[InvocationContext, LlmResponse], Awaitable[None]] | None
        ) = None,
        on_model_error_callback: (
            Callable[
                [InvocationContext, LlmRequest, Exception],
                Awaitable[LlmResponse | None],
            ]
            | None
        ) = None,
        before_tool_callback: (
            Callable[[InvocationContext, str, dict], Awaitable[None]] | None
        ) = None,
        after_tool_callback: (
            Callable[[InvocationContext, str, Any], Awaitable[None]] | None
        ) = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._instructions = instructions
        self._planner = planner
        self._output_schema = output_schema
        self.max_iterations = max_iterations
        self.before_model_callback = before_model_callback
        self.after_model_callback = after_model_callback
        self.on_model_error_callback = on_model_error_callback
        self.before_tool_callback = before_tool_callback
        self.after_tool_callback = after_tool_callback

    async def _resolve_instructions(self, ctx: InvocationContext) -> str:
        """Resolve the system prompt from a string or instruction provider.

        When ``output_schema`` is set, appends JSON schema format instructions
        so the LLM knows the exact structure to return.

        Parameters
        ----------
        ctx : InvocationContext
            The current invocation context.

        Returns
        -------
        str
            The resolved system prompt.
        """
        if callable(self._instructions):
            result = self._instructions(ctx)
            if asyncio.iscoroutine(result):
                prompt = await result
            else:
                prompt = result
        else:
            prompt = self._instructions

        # Append output schema instructions using LangChain's PydanticOutputParser
        if self._output_schema is not None:
            parser = PydanticOutputParser(pydantic_object=self._output_schema)
            prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

        return prompt

    def _build_bound_llm(self) -> BaseChatModel:
        """Return the LLM with tools and/or structured output bound.

        When ``output_schema`` is set and the model supports
        ``with_structured_output`` (OpenAI, Anthropic, etc.), binds it
        for API-level schema enforcement. Otherwise falls back to
        prompt-based JSON instructions + ``parse_json_markdown`` parsing.

        Returns
        -------
        BaseChatModel
            The LLM, optionally with tools and/or structured output bound.
        """
        llm = self._llm
        if self._tools:
            llm = llm.bind_tools(list(self._tools.values()))
        return llm

    def _build_structured_llm(self) -> Any:
        """Return the LLM bound with structured output for the final answer.

        Uses ``with_structured_output(method="json_schema")`` when supported,
        which gives API-level schema enforcement (guaranteed valid JSON).
        Falls back to ``method="json_mode"`` then ``None`` for models that
        don't support it.

        Returns
        -------
        Any
            A Runnable that returns a Pydantic model instance, or None
            if the model does not support structured output.
        """
        if self._output_schema is None:
            return None
        # Try json_schema first (OpenAI strict mode), then json_mode, then None
        for method in ("json_schema", "json_mode"):
            try:
                return self._llm.with_structured_output(
                    self._output_schema, method=method
                )
            except (NotImplementedError, TypeError, ValueError):
                continue
        return None

    def _build_request(
        self,
        system_instruction: str,
        messages: list[BaseMessage],
    ) -> LlmRequest:
        """Package the current turn into an LlmRequest.

        Parameters
        ----------
        system_instruction : str
            The resolved system prompt for this invocation.
        messages : list[BaseMessage]
            The current message history.

        Returns
        -------
        LlmRequest
            A populated request ready to be passed to the model.
        """
        return LlmRequest(
            model=getattr(self._llm, "model_name", None)
            or getattr(self._llm, "model", None),
            system_instruction=system_instruction,
            messages=list(messages),
            tools=list(self._tools.values()),
            tools_dict=dict(self._tools),
            output_schema=self._output_schema,
        )

    def _apply_planner_instruction(
        self,
        base_prompt: str,
        ctx: InvocationContext,
        request: LlmRequest,
    ) -> str:
        """Append the planner's instruction to the system prompt for this turn.

        Parameters
        ----------
        base_prompt : str
            The resolved base system prompt.
        ctx : InvocationContext
            The current invocation context.
        request : LlmRequest
            The LLM request for this turn.

        Returns
        -------
        str
            The system prompt with planning instruction appended, or the
            base prompt unchanged if no planner is attached.
        """
        if self._planner is None:
            return base_prompt
        from langchain_adk.agents.readonly_context import ReadonlyContext
        readonly = ReadonlyContext(ctx)
        instruction = self._planner.build_planning_instruction(readonly, request)
        if instruction:
            return f"{base_prompt}\n\n{instruction}"
        return base_prompt

    @staticmethod
    def _events_to_messages(events: list[Event]) -> list[BaseMessage]:
        """Convert session events to LangChain messages for multi-turn context.

        Reconstructs the conversation history from persisted events so the
        LLM sees previous turns. Mirrors Google ADK's approach where
        session events *are* the conversation memory.
        """
        messages: list[BaseMessage] = []
        for event in events:
            if event.type == EventType.USER_MESSAGE:
                messages.append(HumanMessage(content=event.text))
            elif isinstance(event, FinalAnswerEvent) and not event.partial:
                messages.append(AIMessage(content=event.text))
            elif isinstance(event, ToolCallEvent):
                tool_call_id = event.metadata.get("tool_call_id", "")
                messages.append(AIMessage(
                    content="",
                    tool_calls=[{
                        "id": tool_call_id,
                        "name": event.tool_name,
                        "args": event.tool_input or {},
                    }],
                ))
            elif isinstance(event, ToolResultEvent):
                tool_call_id = event.metadata.get("tool_call_id", "")
                messages.append(ToolMessage(
                    content=event.text or event.error or "",
                    tool_call_id=tool_call_id,
                ))
        return messages

    async def _call_llm(
        self,
        llm: BaseChatModel,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> AsyncIterator[FinalAnswerEvent | AIMessage]:
        """Call the LLM, yielding partial events live when streaming.

        Mirrors LangChain's ``ainvoke`` / ``astream`` pattern:

        - Not streaming → ``ainvoke``, yields ``AIMessage``
        - Streaming → ``astream``, yields partial ``FinalAnswerEvent``
          for each text token, then final ``AIMessage``

        Tool-call responses skip partial events (tool args must be complete).

        Parameters
        ----------
        llm : BaseChatModel
            The bound LLM instance.
        messages : list[BaseMessage]
            The current message history.
        ctx : InvocationContext
            The invocation context (controls streaming, carries trace config).

        Yields
        ------
        FinalAnswerEvent
            Partial events with accumulated text (``partial=True``).
        AIMessage
            The final aggregated message (always last).
        """
        rc = ctx.langchain_run_config or {}

        if not self.is_streaming(ctx):
            yield await llm.ainvoke(messages, config=rc)
            return

        chunks: list[AIMessageChunk] = []
        accumulated_text = ""
        has_tool_calls = False

        async for chunk in llm.astream(messages, config=rc):
            chunks.append(chunk)

            if not has_tool_calls and (
                getattr(chunk, "tool_calls", None)
                or getattr(chunk, "tool_call_chunks", None)
            ):
                has_tool_calls = True

            if has_tool_calls:
                continue

            chunk_text = ""
            if isinstance(chunk.content, str):
                chunk_text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict):
                        chunk_text += part.get("text", "")

            if chunk_text:
                accumulated_text += chunk_text
                yield FinalAnswerEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    content=Content.from_text(accumulated_text),
                    partial=True,
                )

        if not chunks:
            yield AIMessage(content="")
            return

        yield reduce(lambda x, y: x + y, chunks)

    async def astream(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the agent with a manual tool-call loop.

        Parameters
        ----------
        input : str
            The user message or task.
        ctx : InvocationContext
            The invocation context. ``ctx.run_config`` controls streaming.
            ``ctx.langchain_run_config`` carries child callbacks from the
            parent trace (set up by ``_run_with_callbacks``).

        Yields
        ------
        Event
            ToolCallEvent, ToolResultEvent, FinalAnswerEvent, or ErrorEvent.
        """
        system_prompt = await self._resolve_instructions(ctx)
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # Rebuild conversation history from session events (multi-turn)
        if ctx.session and ctx.session.events:
            messages.extend(self._events_to_messages(ctx.session.events))

        messages.append(HumanMessage(content=input))

        llm = self._build_bound_llm()

        for _ in range(self.max_iterations):
            request = self._build_request(system_prompt, messages)

            # Apply planner instruction - update the system message in place
            effective_prompt = self._apply_planner_instruction(system_prompt, ctx, request)
            prompt_changed = effective_prompt != system_prompt
            if prompt_changed and messages and isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=effective_prompt)

            if self.before_model_callback:
                await self.before_model_callback(ctx, request)

            # Call LLM — yields partial events live, then the final AIMessage
            raw_response: AIMessage | None = None
            try:
                async for item in self._call_llm(llm, messages, ctx):
                    if isinstance(item, FinalAnswerEvent):
                        yield item  # live partial
                    else:
                        raw_response = item  # final AIMessage
            except Exception as exc:
                if self.on_model_error_callback:
                    recovery = await self.on_model_error_callback(ctx, request, exc)
                    if recovery is not None:
                        raw_response = AIMessage(content=recovery.text or "")
                    else:
                        yield ErrorEvent(
                            session_id=ctx.session_id,
                            agent_name=self.name,
                            message=str(exc),
                            exception_type=type(exc).__name__,
                        )
                        return
                else:
                    yield ErrorEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                    return

            llm_response = LlmResponse.from_ai_message(raw_response)

            # Let the planner post-process the response
            if self._planner is not None:
                from langchain_adk.agents.readonly_context import ReadonlyContext
                readonly = ReadonlyContext(ctx)
                replacement = self._planner.process_planning_response(readonly, llm_response)
                if replacement is not None:
                    llm_response = replacement

            if self.after_model_callback:
                await self.after_model_callback(ctx, llm_response)

            messages.append(raw_response)

            # No tool calls -> final answer
            if not llm_response.has_tool_calls:
                answer_text = llm_response.text
                parts: list[TextPart | DataPart] = []
                if answer_text:
                    parts.append(TextPart(text=answer_text))

                # Parse structured output if schema is set
                if self._output_schema is not None:
                    structured_output = None
                    parser = PydanticOutputParser(pydantic_object=self._output_schema)
                    # 1) Try parsing the LLM text directly (fast, streaming-safe)
                    if answer_text:
                        try:
                            structured_output = parser.parse(answer_text)
                        except Exception:
                            pass
                    # 2) Fall back to with_structured_output (API-level enforcement)
                    if structured_output is None:
                        structured_llm = self._build_structured_llm()
                        if structured_llm is not None:
                            try:
                                structured_output = await structured_llm.ainvoke(
                                    messages, config=ctx.langchain_run_config or {},
                                )
                            except Exception:
                                pass
                    if structured_output is not None:
                        parts.append(DataPart(data=structured_output.model_dump()))

                yield FinalAnswerEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    content=Content(parts=parts),
                    llm_response=llm_response,
                    partial=False,
                )
                return

            # Execute tool calls
            tool_messages: list[ToolMessage] = []

            for tool_call in llm_response.tool_calls:
                tool_name: str = tool_call["name"]
                tool_args: dict = tool_call["args"]
                tool_call_id: str = tool_call["id"]

                yield ToolCallEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    tool_name=tool_name,
                    tool_input=tool_args,
                    llm_response=llm_response,
                    metadata={"tool_call_id": tool_call_id},
                )

                tool = self._tools.get(tool_name)
                if tool is None:
                    error_msg = f"Tool '{tool_name}' not found. Available: {list(self._tools)}"
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        error=error_msg,
                        metadata={"tool_call_id": tool_call_id},
                    )
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    )
                    continue

                # Inject context into tools that support it (AgentTool, ManageTasksTool)
                if hasattr(tool, "inject_context"):
                    tool.inject_context(ctx)

                if self.before_tool_callback:
                    await self.before_tool_callback(ctx, tool_name, tool_args)

                try:
                    result = await tool.ainvoke(tool_args, config=ctx.langchain_run_config or {})

                    if self.after_tool_callback:
                        await self.after_tool_callback(ctx, tool_name, result)

                    actions = EventActions()
                    result_str = str(result)
                    if result_str.startswith(TRANSFER_SENTINEL):
                        target = result_str.removeprefix(TRANSFER_SENTINEL).strip()
                        actions = EventActions(transfer_to_agent=target)
                    elif result_str == EXIT_LOOP_SENTINEL:
                        actions = EventActions(escalate=True)

                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        content=Content.from_text(result_str),
                        actions=actions,
                        metadata={"tool_call_id": tool_call_id},
                    )
                    tool_messages.append(
                        ToolMessage(content=result_str, tool_call_id=tool_call_id)
                    )

                except Exception as exc:
                    error_msg = str(exc)
                    if self.after_tool_callback:
                        await self.after_tool_callback(ctx, tool_name, None)
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        error=error_msg,
                        metadata={"tool_call_id": tool_call_id},
                    )
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    )

            messages.extend(tool_messages)

        yield ErrorEvent(
            session_id=ctx.session_id,
            agent_name=self.name,
            message=f"Max iterations ({self.max_iterations}) reached without a final answer.",
        )
