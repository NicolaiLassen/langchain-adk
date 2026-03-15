"""Content and Part models — typed multimodal content for events.

Aligns with the A2A protocol's Part types (TextPart, DataPart, FilePart).
Events carry a ``Content`` object with a list of typed parts instead of
loose string fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    """A text content part."""

    type: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataPart(BaseModel):
    """A structured data content part (JSON-serializable dict).

    Used for structured output, tool results with structured data, etc.
    """

    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilePart(BaseModel):
    """A file content part — either inline bytes or a URI reference.

    Attributes
    ----------
    uri : str, optional
        URI pointing to the file (e.g. GCS, S3, HTTP).
    inline_bytes : str, optional
        Base64-encoded file content for inline transfer.
    mime_type : str, optional
        MIME type of the file (e.g. "image/png").
    name : str, optional
        Filename.
    """

    type: Literal["file"] = "file"
    uri: str | None = None
    inline_bytes: str | None = None
    mime_type: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallPart(BaseModel):
    """A tool/function call part — the agent wants to invoke a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResponsePart(BaseModel):
    """A tool/function response part — result from a tool execution."""

    type: Literal["tool_response"] = "tool_response"
    tool_call_id: str
    tool_name: str
    result: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


Part = TextPart | DataPart | FilePart | ToolCallPart | ToolResponsePart


class Content(BaseModel):
    """Container for multimodal content — a list of typed parts.

    Mirrors the A2A protocol's ``Message.parts``.

    Attributes
    ----------
    role : str, optional
        Who produced this content: "user", "model", or "agent".
    parts : list[Part]
        Ordered list of content parts.
    """

    role: str | None = None
    parts: list[Part] = Field(default_factory=list)

    @staticmethod
    def from_text(text: str, *, role: str | None = None) -> Content:
        """Create a Content with a single TextPart."""
        return Content(role=role, parts=[TextPart(text=text)])

    @staticmethod
    def from_data(data: dict[str, Any], *, role: str | None = None) -> Content:
        """Create a Content with a single DataPart."""
        return Content(role=role, parts=[DataPart(data=data)])

    @property
    def text(self) -> str:
        """Concatenate all text parts."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def data(self) -> dict[str, Any] | None:
        """Return the first DataPart's data, or None."""
        for p in self.parts:
            if isinstance(p, DataPart):
                return p.data
        return None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Return all ToolCallPart entries."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def tool_responses(self) -> list[ToolResponsePart]:
        """Return all ToolResponsePart entries."""
        return [p for p in self.parts if isinstance(p, ToolResponsePart)]

    @property
    def has_tool_calls(self) -> bool:
        """Return True if content contains any tool call parts."""
        return any(isinstance(p, ToolCallPart) for p in self.parts)
