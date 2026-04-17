"""Attestation providers and middleware.

Orxhestra ships a protocol plus two reference implementations:

- :class:`AttestationProvider` — the stable seam.  External services
  (blockchain anchors, enterprise audit stores, compliance backends)
  plug in by implementing this protocol in their own packages — no
  named-provider adapter is shipped in the core repo.
- :class:`NoOpAttestationProvider` — drop-in default that records
  nothing.  Keeps the framework lean when attestation isn't needed.
- :class:`LocalAttestationProvider` — append-only JSON-on-disk audit
  log with SHA-256 hash chaining and detached Ed25519 signatures on
  each entry.  Zero external dependencies beyond
  :mod:`orxhestra.auth.crypto`.
- :class:`AttestationMiddleware` — records events as audit entries
  and issues structured claims on interesting actions (tool
  invocations, agent transfers).

Like the trust layer, attestation is **opt-in**: no middleware is
auto-registered.  You construct an :class:`AttestationMiddleware` with
your provider and pass it to :class:`Runner(middleware=...)`.

See Also
--------
orxhestra.trust : Signature verification layer that pairs naturally
    with attestation for full end-to-end coverage.
"""

from orxhestra.attestation.local import LocalAttestationProvider
from orxhestra.attestation.middleware import AttestationMiddleware
from orxhestra.attestation.noop import NoOpAttestationProvider
from orxhestra.attestation.protocol import (
    AttestationProvider,
    Claim,
)

__all__ = [
    "AttestationMiddleware",
    "AttestationProvider",
    "Claim",
    "LocalAttestationProvider",
    "NoOpAttestationProvider",
]
