"""Push-notification config storage + webhook delivery for the A2A server.

Implements the four spec methods that manage push-notification
configs:

* ``CreateTaskPushNotificationConfig`` (a.k.a. ``set``)
* ``GetTaskPushNotificationConfig``
* ``ListTaskPushNotificationConfigs``
* ``DeleteTaskPushNotificationConfig``

…plus the runtime piece: a :class:`PushNotificationDispatcher` that
fires HTTP POSTs to every config's webhook URL on every task
lifecycle event.

Security notes
--------------
* Default :meth:`PushNotificationDispatcher._validate_url` blocks
  ``http://``, ``localhost``, ``127.0.0.0/8``, ``169.254.0.0/16``
  (link-local), ``::1``, and any RFC1918 private range.  Override
  with ``allow_insecure_webhooks=True`` for local development.
* This release ships **Bearer-token** authentication only.  JWT +
  JWKS, HMAC, and Basic schemes are accepted on the model union but
  ignored when dispatching.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlparse

from orxhestra.a2a.types import PushNotificationConfig

if TYPE_CHECKING:
    from orxhestra.a2a.types import (
        Task,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )

_log = logging.getLogger(__name__)


@runtime_checkable
class PushNotificationStore(Protocol):
    """Async storage protocol for push-notification configs."""

    async def set(self, task_id: str, config: PushNotificationConfig) -> None:
        """Insert or update a config by ``(task_id, config.id)``."""

    async def get(
        self, task_id: str, config_id: str | None = None,
    ) -> PushNotificationConfig | None:
        """Return one config, or ``None`` if not found.

        ``config_id`` is optional — when ``None``, returns the first
        config registered for the task (the spec allows multiple but
        most clients only register one).
        """

    async def list(self, task_id: str) -> list[PushNotificationConfig]:
        """Return every config registered for a task."""

    async def delete(self, task_id: str, config_id: str) -> bool:
        """Remove a config.  Returns ``True`` if it existed."""


class InMemoryPushNotificationStore:
    """Default :class:`PushNotificationStore` — in-process dict-of-dicts."""

    def __init__(self) -> None:
        # task_id -> {config_id -> config}
        self._configs: dict[str, dict[str, PushNotificationConfig]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def set(self, task_id: str, config: PushNotificationConfig) -> None:
        async with self._lock:
            self._configs[task_id][config.id] = config

    async def get(
        self, task_id: str, config_id: str | None = None,
    ) -> PushNotificationConfig | None:
        async with self._lock:
            bucket = self._configs.get(task_id) or {}
            if config_id is not None:
                return bucket.get(config_id)
            return next(iter(bucket.values()), None)

    async def list(self, task_id: str) -> list[PushNotificationConfig]:
        async with self._lock:
            return list(self._configs.get(task_id, {}).values())

    async def delete(self, task_id: str, config_id: str) -> bool:
        async with self._lock:
            bucket = self._configs.get(task_id)
            if bucket is None or config_id not in bucket:
                return False
            del bucket[config_id]
            return True


class PushNotificationDispatcher:
    """Fire-and-forget webhook delivery for task lifecycle events.

    The server calls :meth:`dispatch` for every emitted streaming
    event.  For each registered config on the task, the dispatcher
    schedules a background HTTP POST.  Failures are logged but never
    re-raised — the SSE loop must never block on a slow webhook
    consumer.

    Parameters
    ----------
    store
        Backing :class:`PushNotificationStore` to read configs from.
    timeout
        HTTP request timeout in seconds (default 5).
    max_retries
        Re-attempts on connection / 5xx errors (default 2 — three
        attempts total).
    allow_insecure_webhooks
        When ``False`` (default), reject ``http://`` URLs and any
        loopback / link-local / RFC1918 destination as SSRF guard.
    """

    def __init__(
        self,
        store: PushNotificationStore,
        *,
        timeout: float = 5.0,
        max_retries: int = 2,
        allow_insecure_webhooks: bool = False,
    ) -> None:
        self._store = store
        self._timeout = timeout
        self._max_retries = max_retries
        self._allow_insecure = allow_insecure_webhooks

    async def dispatch(
        self,
        task: Task,
        event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    ) -> None:
        """Fire one POST per registered config.  Never raises."""
        configs = await self._store.list(task.id)
        if not configs:
            return
        # Schedule each delivery as a background task so the caller
        # (the SSE producer) is not blocked by webhook latency.
        for cfg in configs:
            try:
                self._validate_url(cfg.url)
            except ValueError as exc:
                _log.warning("rejecting push-notification URL %s: %s", cfg.url, exc)
                continue
            asyncio.create_task(self._post(cfg, task, event))

    async def _post(
        self,
        config: PushNotificationConfig,
        task: Task,
        event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    ) -> None:
        import httpx  # local import — keeps httpx optional at module load

        payload = {
            "taskId": task.id,
            "contextId": task.context_id,
            "event": event.model_dump(by_alias=True, exclude_none=True),
        }
        headers = {"Content-Type": "application/json"}
        token = self._extract_bearer_token(config)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        attempt = 0
        delay = 0.5
        while attempt <= self._max_retries:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(config.url, json=payload, headers=headers)
                if 200 <= resp.status_code < 300:
                    return
                if resp.status_code < 500 and resp.status_code != 408:
                    # Client error other than timeout — don't retry.
                    _log.warning(
                        "push-notification rejected by %s: HTTP %d",
                        config.url,
                        resp.status_code,
                    )
                    return
            except Exception as exc:  # noqa: BLE001 — fire-and-forget
                _log.warning("push-notification to %s failed: %s", config.url, exc)
            attempt += 1
            if attempt <= self._max_retries:
                await asyncio.sleep(delay)
                delay *= 2

        _log.warning(
            "push-notification to %s exhausted %d retries",
            config.url,
            self._max_retries,
        )

    def _extract_bearer_token(self, config: PushNotificationConfig) -> str | None:
        if config.authentication is not None:
            if config.authentication.type.lower() == "bearer":
                return config.authentication.credential
            return None  # other auth types not implemented this release
        return config.token

    def _validate_url(self, url: str) -> None:
        """Raise :class:`ValueError` for URLs that fail SSRF checks."""
        if self._allow_insecure:
            return
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(
                f"push-notification URLs must use https:// (got {parsed.scheme!r})",
            )
        host = parsed.hostname
        if host is None:
            raise ValueError("push-notification URL has no hostname")
        # Block obvious local hostnames first; falls through to the IP
        # check below for the rest.
        if host.lower() in {"localhost", "ip6-localhost"}:
            raise ValueError(f"localhost not allowed: {host!r}")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return  # Hostname (DNS name) — caller validated above.
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_multicast
            or ip.is_reserved
        ):
            raise ValueError(f"non-public IP not allowed: {host!r}")
