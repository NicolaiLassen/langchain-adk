"""Middleware that drives an :class:`AttestationProvider` from the agent loop.

Records every event in the provider's audit log.  On notable actions
— tool invocations, agent transfers — issues typed claims so
downstream verifiers have a receipt for each side-effect.

The middleware is best-effort: provider failures are logged but do
not abort the event stream.  Agents that must halt on audit failure
should wrap their provider in an adapter that raises instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orxhestra.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event
    from orxhestra.trust.attestation.protocol import AttestationProvider


logger: logging.Logger = logging.getLogger(__name__)


class AttestationMiddleware(BaseMiddleware):
    """Route agent events through an :class:`AttestationProvider`.

    Parameters
    ----------
    provider : AttestationProvider
        Backend to append events and issue claims to.  Use
        :class:`NoOpAttestationProvider` for a zero-overhead default
        or :class:`LocalAttestationProvider` for on-disk persistence.

    See Also
    --------
    orxhestra.trust.attestation.protocol.AttestationProvider : Contract.
    orxhestra.trust.attestation.local.LocalAttestationProvider : Reference
        hash-chained store.
    orxhestra.trust.attestation.noop.NoOpAttestationProvider : Default.
    orxhestra.middleware.trust.TrustMiddleware : Sibling middleware —
        pair with this one for verify + audit.

    Examples
    --------
    >>> from orxhestra import Runner
    >>> from orxhestra.middleware import AttestationMiddleware
    >>> from orxhestra.trust import LocalAttestationProvider
    >>> provider = LocalAttestationProvider("./audit", signing_key, did)
    >>> runner = Runner(
    ...     agent=my_agent,
    ...     middleware=[AttestationMiddleware(provider)],
    ... )
    """

    def __init__(self, provider: AttestationProvider) -> None:
        self.provider = provider

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Audit the event and issue claims for interesting side-effects.

        Parameters
        ----------
        ctx : InvocationContext
        event : Event

        Returns
        -------
        Event
            Always returns the event unchanged — attestation is
            non-intrusive.
        """
        try:
            await self.provider.append_audit(event)
        except Exception as exc:
            logger.warning("AttestationMiddleware append_audit failed: %s", exc)

        if event.partial:
            return event

        subject_did = event.signer_did or (event.agent_name or "")

        if event.actions.transfer_to_agent:
            try:
                await self.provider.issue_claim(
                    subject_did=subject_did,
                    claim_type="agent.transfer",
                    claims={
                        "from_agent": event.agent_name or "",
                        "to_agent": event.actions.transfer_to_agent,
                        "branch": event.branch,
                        "timestamp": event.timestamp,
                        "event_id": event.id,
                    },
                )
            except Exception as exc:
                logger.warning("agent.transfer claim failed: %s", exc)

        for tc in event.tool_calls:
            try:
                await self.provider.issue_claim(
                    subject_did=subject_did,
                    claim_type="tool.invoke",
                    claims={
                        "tool_name": tc.tool_name,
                        "tool_call_id": tc.tool_call_id,
                        "branch": event.branch,
                        "timestamp": event.timestamp,
                        "event_id": event.id,
                    },
                )
            except Exception as exc:
                logger.warning("tool.invoke claim failed: %s", exc)

        return event
