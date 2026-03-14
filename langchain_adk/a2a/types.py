"""A2A protocol types — spec-compliant models (v0.3.0).

Implements the core A2A wire-format types based on:
  https://a2a-protocol.org/latest/specification/

All models use camelCase aliases for JSON serialization to match the spec.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

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
    metadata: dict[str, Any] | None = None


class FileWithBytes(A2AModel):
    bytes: str  # base64
    mime_type: str | None = None
    name: str | None = None


class FileWithUri(A2AModel):
    uri: str
    mime_type: str | None = None
    name: str | None = None


class FilePart(A2AModel):
    kind: Literal["file"] = "file"
    file: FileWithBytes | FileWithUri
    metadata: dict[str, Any] | None = None


class DataPart(A2AModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


Part = TextPart | FilePart | DataPart


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
    context_id: str | None = None
    task_id: str | None = None
    reference_task_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class Artifact(A2AModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parts: list[Part]
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


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
    message: Message | None = None
    timestamp: str | None = None


class Task(A2AModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    kind: Literal["task"] = "task"
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class TaskStatusUpdateEvent(A2AModel):
    kind: Literal["status-update"] = "status-update"
    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(A2AModel):
    kind: Literal["artifact-update"] = "artifact-update"
    task_id: str
    context_id: str
    artifact: Artifact
    append: bool | None = None
    last_chunk: bool | None = None
    metadata: dict[str, Any] | None = None


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
    examples: list[str] | None = None
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None


class AgentCapabilities(A2AModel):
    streaming: bool | None = None
    push_notifications: bool | None = None
    state_transition_history: bool | None = None


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
    provider: AgentProvider | None = None
    documentation_url: str | None = None
    icon_url: str | None = None


# ---------------------------------------------------------------------------
# JSON-RPC 2.0
# ---------------------------------------------------------------------------


class JSONRPCError(A2AModel):
    code: int
    message: str
    data: Any | None = None


class JSONRPCRequest(A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    result: Any | None = None
    error: JSONRPCError | None = None


# ---------------------------------------------------------------------------
# A2A-specific request params
# ---------------------------------------------------------------------------


class MessageSendConfiguration(A2AModel):
    accepted_output_modes: list[str] | None = None
    history_length: int | None = None
    blocking: bool | None = None


class MessageSendParams(A2AModel):
    message: Message
    configuration: MessageSendConfiguration | None = None
    metadata: dict[str, Any] | None = None


class TaskQueryParams(A2AModel):
    id: str
    history_length: int | None = None


class TaskIdParams(A2AModel):
    id: str
    metadata: dict[str, Any] | None = None


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
