"""Trust policy and signature verification middleware.

Turns the signing primitives in :mod:`orxhestra.auth.crypto` into an
end-to-end enforcement path:

- :class:`TrustPolicy` — declarative allow/deny rules and verification
  requirements.
- :class:`TrustMiddleware` — a :class:`Middleware` that verifies every
  event passing through the stack against a policy.
- :class:`PolicyDecision` — the verdict returned for each event.

The layer is **opt-in**.  Agents run unchanged with no trust middleware
registered; you only get enforcement when you construct a
:class:`TrustMiddleware` and pass it to :class:`Runner(middleware=...)`
(or declare it via composer YAML).

See Also
--------
orxhestra.auth.did : DID resolvers consumed by :class:`TrustMiddleware`.
orxhestra.attestation : Audit / claim issuance built on top of trust.
"""

from orxhestra.trust.middleware import TrustMiddleware
from orxhestra.trust.policy import PolicyDecision, TrustPolicy

__all__ = [
    "PolicyDecision",
    "TrustMiddleware",
    "TrustPolicy",
]
