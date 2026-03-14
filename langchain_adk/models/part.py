"""Content and Part models — typed multimodal content for events.

Aligns with Google ADK's Content/Part model and the A2A protocol's
Part types (TextPart, DataPart, FilePart). Events carry a ``Content``
object with a list of typed parts instead of loose string fields.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

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
    uri: Optional[str] = None
    inline_bytes: Optional[str] = None
    mime_type: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


Part = Union[TextPart, DataPart, FilePart]


class Content(BaseModel):
    """Container for multimodal content — a list of typed parts.

    Mirrors Google ADK's ``Content`` and A2A's ``Message.parts``.

    Attributes
    ----------
    role : str, optional
        Who produced this content: "user", "model", or "agent".
    parts : list[Part]
        Ordered list of content parts.
    """

    role: Optional[str] = None
    parts: list[Part] = Field(default_factory=list)

    @staticmethod
    def from_text(text: str, *, role: Optional[str] = None) -> Content:
        """Create a Content with a single TextPart."""
        return Content(role=role, parts=[TextPart(text=text)])

    @staticmethod
    def from_data(data: dict[str, Any], *, role: Optional[str] = None) -> Content:
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
