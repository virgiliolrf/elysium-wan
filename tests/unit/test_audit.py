"""Verify-before-fold: a quarantined microbatch never folds until released, an
attacked fleet's step gradient is bit-identical to the honest fleet's, and the
audit policy replicates newcomers fully, spot-checks veterans at a floor, bans
on one mismatch, and draws its sortition from the seed."""

# WOULD-FAIL-IF: a quarantined microbatch folds before release, a rejected poisoned gradient changes the step's bits, or the policy stops replicating newcomers, banning on one mismatch or drawing sortition from the seed

import pytest
import torch

from wantrain.fault_tolerance.audit import AuditPolicy, digest
from wantrain.runtime.accumulate import OrderedAccumulator


def grads(seed, k=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 5, generator=g) for _ in range(k)]


# ---------------- the quarantine gate ----------------


def test_quarantine_blocks_fold_until_release():
    # covers: RUNTIME-I4
    a = OrderedAccumulator(range(3))
    a.add(0, grads(0))
    a.add(1, grads(1), verified=False)
    a.add(2, grads(2))
    assert a.pos == 1  # 0 folded; 1 quarantined blocks 2
    a.release(1)
    assert a.complete()


def test_rejected_poison_yields_honest_bits():
    """The theorem: attacked fleet == honest fleet, hex for hex."""
    # covers: RUNTIME-I4, CONTROL-I12
    honest = OrderedAccumulator(range(4))
    for mb in range(4):
        honest.add(mb, grads(mb))
    want = honest.result()

    attacked = OrderedAccumulator(range(4))
    attacked.add(0, grads(0))
    poison = [g + 1e-4 for g in grads(1)]  # sub-noise poison
    attacked.add(1, poison, verified=False)
    attacked.add(2, grads(2))
    attacked.add(3, grads(3))
    # auditor recomputes mb 1, digest differs, replacement folds instead
    assert digest(poison) != digest(grads(1))
    attacked.reject(1, grads(1))
    got = attacked.result()

    for w, g in zip(want, got):
        assert torch.equal(w, g)  # bitwise, not allclose


def test_release_path_is_also_exact():
    # covers: RUNTIME-I4
    honest = OrderedAccumulator(range(2))
    honest.add(0, grads(0))
    honest.add(1, grads(1))
    gated = OrderedAccumulator(range(2))
    gated.add(0, grads(0), verified=False)
    gated.add(1, grads(1), verified=False)
    assert gated.pos == 0
    gated.release(0)
    gated.release(1)
    for w, g in zip(honest.result(), gated.result()):
        assert torch.equal(w, g)


def test_result_refuses_while_quarantined():
    # covers: RUNTIME-I4
    a = OrderedAccumulator(range(1))
    a.add(0, grads(0), verified=False)
    with pytest.raises(AssertionError, match="quarantine"):
        a.result()


# ---------------- digests ----------------


def test_digest_is_exact_and_shape_aware():
    # covers: CONTROL-I12
    assert digest(grads(7)) == digest(grads(7))
    bumped = [g.clone() for g in grads(7)]
    bumped[0][0, 0] += 2**-20  # one ulp-ish nudge, one tensor
    assert digest(bumped) != digest(grads(7))
    flat = [g.reshape(-1) for g in grads(7)]
    assert digest(flat) != digest(grads(7))  # same bytes, different shape


# ---------------- the policy ----------------


def test_new_node_is_always_replicated():
    # covers: CONTROL-I12
    p = AuditPolicy(seed=b"s")
    assert all(p.replicate("newbie", s, m) for s in range(4) for m in range(8))


def test_reputation_decays_to_floor_not_zero():
    # covers: CONTROL-I12
    p = AuditPolicy(seed=b"s", warmup=4, floor=0.05, half_life=8)
    for i in range(4):
        p.record("vet", 0, i, "d", "d")
    assert p.replication_p("vet") == 1.0  # warmup just met
    for i in range(200):
        p.record("vet", 1, i, "d", "d")
    assert p.replication_p("vet") == 0.05  # floor holds forever


def test_one_mismatch_is_a_ban():
    # covers: CONTROL-I12
    p = AuditPolicy(seed=b"s")
    for i in range(50):
        p.record("t", 0, i, "d", "d")
    assert not p.banned("t")
    ok = p.record("t", 1, 0, "claimed", "different")
    assert not ok and p.banned("t")
    assert p.replication_p("t") == 1.0


def test_sortition_needs_the_seed():
    # covers: CONTROL-I12
    a = AuditPolicy(seed=b"secret-a", warmup=0, floor=0.3, half_life=1)
    b = AuditPolicy(seed=b"secret-b", warmup=0, floor=0.3, half_life=1)
    for pol in (a, b):
        for i in range(64):
            pol.record("v", 0, i, "d", "d")
    draws_a = [a.replicate("v", 1, m) for m in range(400)]
    draws_b = [b.replicate("v", 1, m) for m in range(400)]
    assert draws_a != draws_b  # seed changes the sample
    assert 0.15 < sum(draws_a) / 400 < 0.45  # rate ~ floor
    assert draws_a == [a.replicate("v", 1, m) for m in range(400)]  # reproducible


def test_ledger_records_evidence():
    # covers: CONTROL-I12
    p = AuditPolicy(seed=b"s")
    p.record("n", 3, 7, "aa", "bb")
    v = p.ledger[-1]
    assert (v.node, v.step, v.mb, v.ok) == ("n", 3, 7, False)
    assert v.claimed != v.recomputed
