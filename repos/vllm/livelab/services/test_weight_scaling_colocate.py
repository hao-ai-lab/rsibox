"""Tests for weight_scaling_colocate.py.

CPU (always): ledger, digest comparison and gate logic, no torch needed.
GPU (FASTRSI_GPU_E2E=1, FASTRSI_VLLM_MODEL=<path>, FASTRSI_VLLM_PYTHON=<venv python>,
FASTRSI_GPUS=g0,g1): real holders (vllm preload), real vLLM engines leased over CUDA IPC, a
holder seeded over NCCL, and the pool. One regression test per bug of the PR #125 model:
  (a) a passing patch that was never accepted leaked into the active state;
  (b) rollback was a shallow copy;
  (c) a corruption fault failed over onto the same corrupted holder.
Run under the vLLM venv:  FASTRSI_GPU_E2E=1 ... python -m pytest -x -v this_file
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_scaling_colocate as W  # noqa: E402


# ------------------------------------------------------------------------------------- CPU --

def test_ledger_is_append_only_and_frozen():
    led = W.Ledger()
    p = W.Patch("p1", r"x", "scale", 2.0)
    led.append(p)
    with pytest.raises(ValueError):
        led.append(W.Patch("p1", r"y"))
    with pytest.raises(Exception):
        led.entries[0].value = 3.0            # frozen dataclass
    assert led.ids == ["p1"] and isinstance(led.entries, tuple)


def test_digests_agree_reports_differences():
    a = {"w1": "1:2", "w2": "3:4"}
    assert W.digests_agree(a, dict(a))["ok"]
    r = W.digests_agree(a, {"w1": "1:2", "w2": "9:9"})
    assert not r["ok"] and r["differ"] == ["w2"]
    assert not W.digests_agree(a, {"w1": "1:2"})["ok"]


def test_gate_compare_tokens_and_tolerance():
    ref = [{"tokens": ["a", "b"], "logprobs": [-0.1, -0.2]}]
    assert W.gate_compare(ref, ref, 0.0)["pass"]
    assert W.gate_compare(ref, [{"tokens": ["a", "b"], "logprobs": [-0.1, -0.24]}], 0.05)["pass"]
    assert not W.gate_compare(ref, [{"tokens": ["a", "c"], "logprobs": [-0.1, -0.2]}], 1.0)["pass"]
    assert not W.gate_compare(ref, [{"tokens": ["a", "b"], "logprobs": [-0.1, -0.9]}], 0.05)["pass"]


# CPU stand-ins for the GPU side of the pool, so Pool's own state machine (failover, rollback,
# accept) runs for real: each fake engine leases one fake holder whose "weights" are one number,
# lease_apply scales it, lease_digest reports it, W.copy copies it between holders (the NCCL
# copy), and W.reference answers with logprobs that follow it (the HTTP reference prompts).

class _FakeHolder:
    def __init__(self, gpu):
        self.gpu, self.socket_dir, self.weight = gpu, "/nonexistent", 1.0


class _FakeEngine:
    def __init__(self, name, gpu):
        self.name, self.gpu, self.port, self.pid = name, gpu, 40000 + gpu, None
        self.holder = _FakeHolder(gpu)

    def reset_prefix_cache(self):
        return 200

    def rpc(self, method, arg="{}", timeout=300):
        if method == "lease_digest":
            return {"ok": True, "seconds": 0.0, "digest": {"w": repr(self.holder.weight)}}
        if method == "lease_apply":
            self.holder.weight *= float(__import__("json").loads(arg)["value"])
            return {"ok": True, "seconds": 0.0, "touched": 1}
        raise AssertionError(method)


@pytest.fixture
def cpu_pool(monkeypatch):
    engines = {}

    def fake_reference(port, model, max_tokens=16):
        eng = next(e for e in engines.values() if e.port == port)
        return [{"tokens": ["ok"], "logprobs": [-0.1 * eng.holder.weight]}]

    def fake_copy(model, socket_dir, src_gpu, dst_gpu, *args, **kwargs):
        holders = {e.holder.gpu: e.holder for e in engines.values()}
        holders[dst_gpu].weight = holders[src_gpu].weight
        return {"ok": True}

    monkeypatch.setattr(W, "reference", fake_reference)
    monkeypatch.setattr(W, "copy", fake_copy)
    engines.update(A=_FakeEngine("A", 0), S=_FakeEngine("S", 1))
    p = W.Pool("m", sys.executable, tempfile.gettempdir(), engines["A"], engines["S"],
               W._free_port(), tol=1.0)
    p._stop.set()
    yield p, engines["A"], engines["S"]
    p.close()


def test_rollback_after_a_failover_restores_the_patched_holder_not_the_clean_one(cpu_pool):
    """Input: a passing patch x2 pending on A, then a failover A -> S (watchdog or a 5xx).
    Action: rollback(). Expect: S keeps serving and both holders hold the ledger (weight 1.0).
    Before: rollback failed back over to A and copied A's unaccepted weights onto S (both 2.0)."""
    pool, a, s = cpu_pool
    assert pool.evaluate(W.Patch("p1", "w", "scale", 2.0))["state"] == "pending"
    pool.failover("transient 500", a)
    pool.rollback("discard p1")
    assert pool.active is s and (a.holder.weight, s.holder.weight) == (1.0, 1.0)
    assert pool.check_integrity(pool.active)["ok"] and pool.check_integrity(pool.standby)["ok"]


def test_accept_refuses_when_the_patched_engine_no_longer_serves(cpu_pool):
    """Input: a patch x2 pending on A, then a failover A -> S. Action: accept(). Expect: a
    refusal, nothing in the ledger, and S (serving) still unpatched at 1.0; rollback() then
    repairs A. Before: accept replayed the patch onto A again (4.0), left S unpatched and
    recorded p1 in the ledger."""
    pool, a, s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    pool.failover("transient 500", a)
    with pytest.raises(RuntimeError, match="no longer serves"):
        pool.accept()
    assert pool.ledger.ids == [] and s.holder.weight == 1.0 and a.holder.weight == 2.0
    pool.rollback("discard p1")
    assert (a.holder.weight, s.holder.weight) == (1.0, 1.0)


def test_failover_never_returns_to_a_holder_that_was_not_repaired(cpu_pool):
    """Input: A fails over to S (A's holder corrupted, not yet re-copied), then S reports a 5xx.
    Action: failover again. Expect: refused (S keeps serving); after restore_holder(A, S) a
    failover to A is allowed. Before: the second failover routed traffic onto the corrupted A."""
    pool, a, s = cpu_pool
    a.holder.weight = -999.0
    pool.failover("digest mismatch", a)
    assert pool.failover("503 on S", s) is None and pool.active is s
    assert pool.restore_holder(a, s)["digest_ok"]
    assert pool.failover("503 on S", s) is not None and pool.active is a


# ------------------------------------------------------------------------------------- GPU --

GPU = os.environ.get("FASTRSI_GPU_E2E") == "1"
gpu = pytest.mark.skipif(not GPU, reason="needs FASTRSI_GPU_E2E=1 and a vLLM venv with GPUs")
MML = 4096
GMU = 0.35


@pytest.fixture(scope="module")
def pool():
    model = os.environ["FASTRSI_VLLM_MODEL"]
    python = os.environ.get("FASTRSI_VLLM_PYTHON", sys.executable)
    g0, g1 = (int(x) for x in os.environ.get("FASTRSI_GPUS", "0,1").split(","))
    work = tempfile.mkdtemp(prefix="wsc-test-", dir=os.environ.get("FASTRSI_TEST_TMP"))
    # AF_UNIX paths are limited to 107 bytes: the socket dir stays short, outside `work`
    sock = tempfile.mkdtemp(prefix="wsc-")
    h0 = W.Holder(g0, sock, os.path.join(work, "h0.log"))
    h1 = W.Holder(g1, sock, os.path.join(work, "h1.log"))
    engines = []
    try:
        h0.start(model, python, MML)
        h1.start(model, python, MML, dummy=True)
        seed = W.copy(model, sock, g0, g1, MML, python, work)
        assert seed["ok"], seed
        a = W.Engine("A", g0, 8851, work, h0)
        s = W.Engine("S", g1, 8852, work, h1)
        engines += [a, s]
        a.start(model, python, GMU, MML)
        s.start(model, python, GMU, MML)
        p = W.Pool(model, python, work, a, s, 8859)
        # the seeded holder serves the same model as the disk-loaded one
        assert W.digests_agree(p.ref_digest, p.digest(s))["ok"]
        yield p
        p.close()
    finally:
        for e in engines:
            e.stop()
        h0.stop()
        h1.stop()


def _clean(pool, eng):
    return pool.check_integrity(eng)["ok"]


@gpu
def test_bug_a_unaccepted_patch_never_leaks(pool):
    p1, p2 = W.GOOD
    r1 = pool.evaluate(p1)
    assert r1["state"] == "pending", r1
    assert pool.ledger.ids == []
    # the standby never saw p1
    assert _clean(pool, pool.standby)
    r2 = pool.evaluate(p2)               # p1 was not accepted: rolled back first
    assert "rolled_back_pending" in r2 and r2["rolled_back_pending"]["patch"] == "good-1"
    assert r2["state"] == "pending"
    # the engine that ran p1 is the standby now, restored to the ledger (no p1 left)
    assert _clean(pool, pool.standby)
    acc = pool.accept()
    assert acc["ledger"] == ["good-2"] and acc["standby_digest_ok"]
    assert _clean(pool, pool.active) and _clean(pool, pool.standby)


@gpu
def test_bug_b_rollback_restores_every_byte(pool):
    ref = list(pool.ref)
    victim = pool.active
    r = pool.evaluate(W.Patch("bad", r"layers\.(1[0-9])\.mlp\.down_proj", "scale", 0.0))
    assert r["state"] == "rejected" and not r["gate"]["pass"], r
    assert pool.active is not victim             # the standby took over
    assert r["restore"]["digest_ok"]             # every tensor of the patched holder restored
    # the restored engine answers exactly like the ledger reference (tol 0)
    assert W.gate_compare(ref, W.reference(victim.port, pool.model), 0.0)["pass"]
    assert W.gate_compare(ref, W.reference(pool.router.port, pool.model), 0.0)["pass"]


@gpu
def test_bug_c_corruption_fails_over_to_a_clean_holder(pool):
    victim = pool.active
    victim.rpc("lease_fault", "corrupt")
    integ = pool.check_integrity(victim)
    assert not integ["ok"]
    pool.failover("digest mismatch", victim)
    assert pool.active is not victim and pool.active.holder is not victim.holder
    assert _clean(pool, pool.active)
    assert W.gate_compare(pool.ref, W.reference(pool.router.port, pool.model), 0.0)["pass"]
    rs = pool.restore_holder(victim, pool.active)
    assert rs["digest_ok"]
    # a harmless patch after the corruption is not a fault any more (the old model failed over)
    r = pool.evaluate(W.Patch("harmless", r"layers\.7\.mlp\.down_proj", "scale", 1.0))
    assert r["state"] == "pending" and "failover" not in r, r
    pool.rollback("test cleanup")
    assert _clean(pool, pool.active) and _clean(pool, pool.standby)
