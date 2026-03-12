"""A2A protocol types — spec-compliant models (v0.3.0).

Implements the core A2A wire-format types based on:
  https://a2a-protocol.org/latest/specification/

All models use camelCase aliases for JSON serialization to match the spec.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


def _to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class A2AModel(BaseModel):
    """Base for all A2A models with camelCase alias support."""

    model_config = {"populate_by_name": True, "alias_generator": _to_camel}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


class TextPart(A2AModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: Optional[dict[str, Any]] = None


class FileWithBytes(A2AModel):
    bytes: str  # base64
    mime_type: Optional[str] = None
    name: Optional[str] = None


class FileWithUri(A2AModel):
    uri: str
    mime_type: Optional[str] = None
    name: Optional[str] = None


class FilePart(A2AModel):
    kind: Literal["file"] = "file"
    file: Union[FileWithBytes, FileWithUri]
    metadata: Optional[dict[str, Any]] = None


class DataPart(A2AModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: Optional[dict[str, Any]] = None


Part = Union[TextPart, FilePart, DataPart]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Role(str, Enum):
    user = "user"
    agent = "agent"


class Message(A2AModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    parts: list[Part]
    kind: Literal["message"] = "message"
    context_id: Optional[str] = None
    task_id: Optional[str] = None
    reference_task_ids: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class Artifact(A2AModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parts: list[Part]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    submitted = "submitted"
    working = "working"
    input_required = "input-required"
    completed = "completed"
    canceled = "canceled"
    failed = "failed"
    rejected = "rejected"
    auth_required = "auth-required"
    unknown = "unknown"


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class TaskStatus(A2AModel):
    state: TaskState
    message: Optional[Message] = None
    timestamp: Optional[str] = None


class Task(A2AModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    kind: Literal["task"] = "task"
    history: Optional[list[Message]] = None
    artifacts: Optional[list[Artifact]] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class TaskStatusUpdateEvent(A2AModel):
    kind: Literal["status-update"] = "status-update"
    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False
    metadata: Optional[dict[str, Any]] = None


class TaskArtifactUpdateEvent(A2AModel):
    kind: Literal["artifact-update"] = "artifact-update"
    task_id: str
    context_id: str
    artifact: Artifact
    append: Optional[bool] = None
    last_chunk: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------


class AgentProvider(A2AModel):
    organization: str
    url: str


class AgentSkill(A2AModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: Optional[list[str]] = None
    input_modes: Optional[list[str]] = None
    output_modes: Optional[list[str]] = None


class AgentCapabilities(A2AModel):
    streaming: Optional[bool] = None
    push_notifications: Optional[bool] = None
    state_transition_history: Optional[bool] = None


class AgentInterface(A2AModel):
    url: str
    transport: str = "JSONRPC"


class AgentCard(A2AModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocol_version: str = "0.3.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["application/json"],
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["application/json"],
    )
    provider: Optional[AgentProvider] = None
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON-RPC 2.0
# ---------------------------------------------------------------------------


class JSONRPCError(A2AModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCResponse(A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


# ---------------------------------------------------------------------------
# A2A-specific request params
# ---------------------------------------------------------------------------


class MessageSendConfiguration(A2AModel):
    accepted_output_modes: Optional[list[str]] = None
    history_length: Optional[int] = None
    blocking: Optional[bool] = None


class MessageSendParams(A2AModel):
    message: Message
    configuration: Optional[MessageSendConfiguration] = None
    metadata: Optional[dict[str, Any]] = None


class TaskQueryParams(A2AModel):
    id: str
    history_length: Optional[int] = None


class TaskIdParams(A2AModel):
    id: str
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class A2AErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    UNSUPPORTED_OPERATION = -32004
