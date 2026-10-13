"""Verify-before-fold: who gets audited, who gets ejected, and why.

Never fold an unverified gradient; eject on the first invalid one. Replication
is adaptive after BOINC (Anderson 2019, arXiv:1903.01699): count a node's
consecutively validated results, replicate everything from a new node, and
decay the replication probability as the count grows. Comparison is exact
(bitwise determinism per declared device map), and the boundary digest chain
names the faulty stage, step and microbatch in one lookup. Sortition must be
unpredictable to the audited node: `AuditPolicy` draws from an HMAC of a
secret seed, revealed at window close so anyone can re-derive the sample.
"""

import hashlib
import hmac
from dataclasses import dataclass


def digest(tensors) -> str:
    """Exact commitment to a gradient or boundary: sha256 over bytes, shape
    and dtype of every tensor, in order. Equality means equality of bits --
    there is no tolerance band for an adversary to live inside."""
    h = hashlib.sha256()
    for t in tensors:
        c = t.detach().contiguous().cpu()
        h.update(str(tuple(c.shape)).encode())
        h.update(str(c.dtype).encode())
        h.update(c.numpy().tobytes())
    return h.hexdigest()


@dataclass
class Reputation:
    validated: int = 0  # consecutive audits passed (BOINC's N)
    rejected: int = 0
    banned: bool = False


@dataclass
class Verdict:
    node: str
    step: int
    mb: int
    ok: bool
    claimed: str  # digest the worker committed to
    recomputed: str  # digest the auditor produced


class AuditPolicy:
    """Decides, deterministically under a secret seed, which contributions are
    replicated, and tracks reputations and ejections.
    warmup    : audits before a node earns spot-check status; everything
                before that is replicated, so a new node never folds unverified.
    floor     : spot-check rate veterans keep forever (a veteran can turn);
                it bounds a turncoat's expected damage to ~1/floor nudges.
    half_life : audits over which the replication probability halves after
                warmup, decaying from 1.0 toward the floor.
    """

    def __init__(
        self,
        seed: bytes,
        warmup: int = 16,
        floor: float = 0.02,
        half_life: int = 32,
        absolute_every: int = 0,
    ):
        self.seed = seed
        self.warmup = warmup
        self.floor = floor
        self.half_life = half_life
        # A percentage floor lets absolute damage grow with fleet size.
        # `absolute_every` forces an audit at least once per this many of a
        # node's items regardless of the draw, capping a single turncoat's
        # leak at a constant independent of how many peers hide it.
        self.absolute_every = absolute_every
        self.rep = {}  # node -> Reputation
        self.ledger = []  # list[Verdict], the public evidence trail
        self._since = {}  # node -> items since last audit

    def _r(self, node) -> Reputation:
        if node not in self.rep:
            self.rep[node] = Reputation()
        return self.rep[node]

    def replication_p(self, node) -> float:
        r = self._r(node)
        if r.banned:
            return 1.0  # banned nodes should not be here at all
        if r.validated < self.warmup:
            return 1.0
        p = 0.5 ** ((r.validated - self.warmup) / self.half_life)
        return max(self.floor, p)

    def replicate(self, node, step: int, mb: int) -> bool:
        """Unpredictable-but-reproducible sortition: HMAC(seed, item) mapped
        to [0,1) against the node's current replication probability, with a
        hard absolute-frequency backstop on top."""
        p = self.replication_p(node)
        forced = self.absolute_every and self._since.get(node, 0) >= self.absolute_every - 1
        msg = f"{node}|{step}|{mb}".encode()
        draw = int.from_bytes(hmac.new(self.seed, msg, hashlib.sha256).digest()[:8], "big")
        hit = p >= 1.0 or forced or (draw / 2**64) < p
        self._since[node] = 0 if hit else self._since.get(node, 0) + 1
        return hit

    def record(self, node, step: int, mb: int, claimed: str, recomputed: str) -> bool:
        """Auditor result. Returns ok. A single mismatch bans: with exact
        digests there are no honest near-misses inside a device class, so
        one bad answer is proof, not evidence (Bitcoin's banscore collapses
        to threshold 1)."""
        ok = claimed == recomputed
        r = self._r(node)
        if ok:
            r.validated += 1
        else:
            r.rejected += 1
            r.banned = True
        self.ledger.append(Verdict(node, step, mb, ok, claimed, recomputed))
        return ok

    def banned(self, node) -> bool:
        return self._r(node).banned

    def reveal(self) -> bytes:
        """Window close: publish the seed so the sample is re-derivable."""
        return self.seed
