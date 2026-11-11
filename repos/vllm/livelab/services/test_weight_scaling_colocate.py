"""Tests for weight_scaling_colocate.py.

CPU (always): ledger, digest comparison and gate logic, no torch needed; the pool's failover,
rollback and accept with fake engines and holders; copy() when one rank fails.
GPU (FASTRSI_GPU_E2E=1, FASTRSI_VLLM_MODEL=<path>, FASTRSI_VLLM_PYTHON=<venv python>,
FASTRSI_GPUS=g0,g1, default the first two of CUDA_VISIBLE_DEVICES, else 0,1; free ports;
the temp folders are removed afterwards): real holders (vllm preload), real vLLM engines leased over CUDA IPC, a
holder seeded over NCCL, and the pool. One regression test per bug of the PR #125 model:
  (a) a passing patch that was never accepted leaked into the active state;
  (b) rollback was a shallow copy;
  (c) a corruption fault failed over onto the same corrupted holder.
Run under the vLLM venv:  FASTRSI_GPU_E2E=1 ... python -m pytest -x -v this_file
"""
import os
import shutil
import sys
import tempfile
import time
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
        engine = next(item for item in engines.values() if item.port == port)
        return [{"tokens": ["ok"], "logprobs": [-0.1 * engine.holder.weight]}]

    def fake_copy(model, socket_dir, src_gpu, dst_gpu, *args, **kwargs):
        holders = {item.holder.gpu: item.holder for item in engines.values()}
        holders[dst_gpu].weight = holders[src_gpu].weight
        return {"ok": True}

    monkeypatch.setattr(W, "reference", fake_reference)
    monkeypatch.setattr(W, "copy", fake_copy)
    engines.update(A=_FakeEngine("A", 0), S=_FakeEngine("S", 1))
    pool = W.Pool("m", sys.executable, tempfile.gettempdir(), engines["A"], engines["S"],
                  W._free_port(), tol=1.0)
    pool._stop.set()
    yield pool, engines["A"], engines["S"]
    pool.close()


def test_rollback_after_a_failover_restores_the_patched_holder_not_the_clean_one(cpu_pool):
    """Input: a passing patch x2 pending on A, then a failover A -> S (watchdog or a 5xx).
    Action: rollback(). Expect: S keeps serving and both holders hold the ledger (weight 1.0).
    Before: rollback failed back over to A and copied A's unaccepted weights onto S (both 2.0)."""
    pool, engine_a, engine_s = cpu_pool
    assert pool.evaluate(W.Patch("p1", "w", "scale", 2.0))["state"] == "pending"
    pool.failover("transient 500", engine_a)
    pool.rollback("discard p1")
    assert pool.active is engine_s and (engine_a.holder.weight, engine_s.holder.weight) == (1.0, 1.0)
    assert pool.check_integrity(pool.active)["ok"] and pool.check_integrity(pool.standby)["ok"]


def test_accept_refuses_when_the_patched_engine_no_longer_serves(cpu_pool):
    """Input: a patch x2 pending on A, then a failover A -> S. Action: accept(). Expect: a
    refusal, nothing in the ledger, and S (serving) still unpatched at 1.0; rollback() then
    repairs A. Before: accept replayed the patch onto A again (4.0), left S unpatched and
    recorded p1 in the ledger."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    pool.failover("transient 500", engine_a)
    with pytest.raises(RuntimeError, match="no longer serves"):
        pool.accept()
    assert pool.ledger.ids == [] and engine_s.holder.weight == 1.0 and engine_a.holder.weight == 2.0
    pool.rollback("discard p1")
    assert (engine_a.holder.weight, engine_s.holder.weight) == (1.0, 1.0)


def test_failover_never_returns_to_a_holder_that_was_not_repaired(cpu_pool):
    """Input: A fails over to S (A's holder corrupted, not yet re-copied), then S reports a 5xx.
    Action: failover again. Expect: refused (S keeps serving); after restore_holder(A, S) a
    failover to A is allowed. Before: the second failover routed traffic onto the corrupted A."""
    pool, engine_a, engine_s = cpu_pool
    engine_a.holder.weight = -999.0
    pool.failover("digest mismatch", engine_a)
    assert pool.failover("503 on S", engine_s) is None and pool.active is engine_s
    assert pool.restore_holder(engine_a, engine_s)["digest_ok"]
    assert pool.failover("503 on S", engine_s) is not None and pool.active is engine_a


def test_evaluate_refuses_while_a_holder_is_not_repaired(cpu_pool):
    """Input: A fails over to S and A's holder is not re-copied yet. Action: evaluate a patch
    the gate would reject (x100), then one it would pass (x2). Expect: both refused before
    anything is applied; S serves at 1.0. Before: the rejected patch could not fail over and S
    kept serving it (restore refused); the passing one could then only be accepted."""
    pool, engine_a, engine_s = cpu_pool
    pool.failover("transient 500", engine_a)
    for value in (100.0, 2.0):
        with pytest.raises(RuntimeError, match="not repaired"):
            pool.evaluate(W.Patch(f"x{value}", "w", "scale", value))
    assert pool.active is engine_s and engine_s.holder.weight == 1.0 and pool.pending is None


@pytest.mark.parametrize("fails_at", ["replay", "digest"])
def test_a_standby_whose_accept_replay_did_not_complete_is_tainted(cpu_pool, fails_at):
    """Input: a patch x2 pending on A; S's replay half-applies and answers not ok, or S's
    digest after the replay raises. Action: accept(), then A fails. Expect: accept raises, S
    is tainted and the failover to S is refused. Before: S stayed untainted and the next
    failover routed traffic onto its half-patched weights."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    real_rpc = engine_s.rpc

    def broken_rpc(method, arg="{}", timeout=300):
        if method == "lease_apply" and fails_at == "replay":
            engine_s.holder.weight *= 1.5
            return {"ok": False, "error": "replay died halfway"}
        if method == "lease_digest" and fails_at == "digest" and engine_s.holder.weight != 1.0:
            raise RuntimeError("engine S stopped answering")
        return real_rpc(method, arg, timeout)
    engine_s.rpc = broken_rpc
    with pytest.raises(RuntimeError):
        pool.accept()
    assert engine_s.name in pool.tainted
    assert pool.failover("A died", engine_a) is None and pool.active is engine_a


def test_a_failover_cannot_land_between_accepts_check_and_its_replay(cpu_pool):
    """Input: a patch x2 pending on A; a failover A -> S is forced the moment accept() first
    releases the pool lock. Action: accept(). Expect: the replay reached S, not A: A and S at
    2.0 and the ledger holds p1. Before: accept released the lock after its check, the
    failover swapped the engines, and the replay patched A a second time (4.0) while S stayed
    at 1.0."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    inner = pool.lock

    class FailoverOnFirstRelease:
        def __enter__(self):
            return inner.__enter__()

        def __exit__(self, *exc_info):
            result = inner.__exit__(*exc_info)
            pool.lock = inner
            pool.failover("A 500 during accept", engine_a)
            return result
    pool.lock = FailoverOnFirstRelease()
    pool.accept()
    assert (engine_a.holder.weight, engine_s.holder.weight) == (2.0, 2.0)
    assert pool.ledger.ids == ["p1"]


def test_restore_never_copies_from_the_engine_with_the_pending_patch(cpu_pool):
    """Input: a patch x2 pending on A. Action: restore_holder(S, A). Expect: refused, S stays
    at 1.0. Before: S got A's unaccepted weights and stayed untainted."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    with pytest.raises(RuntimeError, match="unaccepted"):
        pool.restore_holder(engine_s, engine_a)
    assert engine_s.holder.weight == 1.0


def test_a_verified_standby_takes_over_when_the_active_dies_inside_accept(cpu_pool, monkeypatch):
    """Input: a patch x2 pending on A; A stops answering (digest and reference prompts) right
    after the replay reached S and the ledger took p1. Action: accept(), then fail A over.
    Expect: accept verifies S against A's digest taken before the replay and returns; the
    failover goes to S, which serves at 2.0 and matches the ledger digest. Before (97b6ee6):
    accept raised on the dead A, S stayed tainted with the old ledger digest, and the pool had
    no engine it would fail over to."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    real_rpc, real_reference = engine_a.rpc, W.reference

    def dead_after_ledger(method, arg="{}", timeout=300):
        if pool.ledger.ids:
            raise ConnectionError("engine A stopped answering")
        return real_rpc(method, arg, timeout)

    def reference_without_a(port, model, max_tokens=16):
        if pool.ledger.ids and port == engine_a.port:
            raise ConnectionError("engine A stopped answering")
        return real_reference(port, model, max_tokens)
    engine_a.rpc = dead_after_ledger
    monkeypatch.setattr(W, "reference", reference_without_a)
    result = pool.accept()
    assert result["standby_digest_ok"] is True and pool.ledger.ids == ["p1"]
    assert pool.failover("A died", engine_a) is not None and pool.active is engine_s
    assert engine_s.holder.weight == 2.0 and pool.check_integrity(engine_s)["ok"]


def test_a_standby_with_wrong_weights_after_accept_is_still_refused(cpu_pool):
    """Input: a patch x2 pending on A; S's replay answers ok but leaves S at 3.0. Action:
    accept(), then fail A over. Expect: standby_digest_ok False, S tainted, failover refused
    (A keeps serving). Proves the accept-time check compares real digests, not a flag."""
    pool, engine_a, engine_s = cpu_pool
    pool.evaluate(W.Patch("p1", "w", "scale", 2.0))
    real_rpc = engine_s.rpc

    def wrong_replay(method, arg="{}", timeout=300):
        if method == "lease_apply":
            engine_s.holder.weight = 3.0
            return {"ok": True, "seconds": 0.0, "touched": 1}
        return real_rpc(method, arg, timeout)
    engine_s.rpc = wrong_replay
    assert pool.accept()["standby_digest_ok"] is False and engine_s.name in pool.tainted
    assert pool.failover("A 500", engine_a) is None and pool.active is engine_a


def test_rollback_with_nothing_pending_is_a_clear_refusal(cpu_pool):
    """Before: rollback() with no pending patch raised AttributeError ('NoneType' ... 'id')."""
    pool, _, _ = cpu_pool
    with pytest.raises(RuntimeError, match="nothing pending"):
        pool.rollback("no patch")


# Stands in for the vLLM venv's python in copy(): it is run as `python <script> copy-rank
# --rank R ...`. Rank 1 fails at once (as when its holder does not answer); rank 0 would wait
# in NCCL init for its peer for 15 minutes (here: 60 s).
FAKE_RANK_PYTHON = """#!{py}
import os, sys, time
rank = sys.argv[sys.argv.index("--rank") + 1]
open(os.path.join({out!r}, "rank" + rank), "w").write(" ".join(sys.argv[2:]))
if rank == "1":
    sys.exit("holder did not answer")
time.sleep(60)
"""


def test_copy_stops_the_peer_rank_when_one_rank_fails(tmp_path):
    """Input: a copy whose rank 1 exits 1 right away while rank 0 blocks. Action: copy() with
    log_dir set. Expect: within seconds, ok False, returncodes [killed, 1], rank 0's process
    gone; and no --max-model-len passed when none was asked (the holder's own default).
    Before: copy() waited on rank 0 first, up to 900 s, and copy-rank then defaulted to 4096,
    a different cache key than a holder started without --max-model-len. Does not prove: the
    NCCL copy itself (GPU tests below)."""
    fake = tmp_path / "python"
    fake.write_text(FAKE_RANK_PYTHON.format(py=sys.executable, out=str(tmp_path)))
    fake.chmod(0o755)
    started = time.monotonic()
    rep = W.copy("m", "/nonexistent", 0, 1, python=str(fake), log_dir=str(tmp_path))
    assert time.monotonic() - started < 20, rep
    assert rep["ok"] is False and rep["returncodes"][1] == 1 and rep["returncodes"][0] != 0, rep
    assert "--max-model-len" not in (tmp_path / "rank0").read_text()
    args = W.parser().parse_args(["copy-rank", "--rank", "0", "--init", "tcp://x:1",
                                  "--model", "m", "--socket-dir", "/s"])
    assert args.max_model_len is None


def test_restore_holder_copies_with_the_pools_max_model_len(cpu_pool, monkeypatch):
    """Input: a pool built with max_model_len=8192 (its holders' cache key). Action:
    restore_holder. Expect: copy() is asked for max_model_len 8192. Before: restore_holder
    omitted it and copy-rank used 4096, so the restore asked the holders for another key."""
    pool, engine_a, engine_s = cpu_pool
    pool.max_model_len = 8192
    asked = []
    monkeypatch.setattr(W, "copy", lambda *args, **kwargs: asked.append(
        kwargs.get("max_model_len")) or {"ok": True})
    pool.restore_holder(engine_a, engine_s)
    assert asked == [8192]


# ------------------------------------------------------------------------------------- GPU --

GPU = os.environ.get("FASTRSI_GPU_E2E") == "1"
gpu = pytest.mark.skipif(not GPU, reason="needs FASTRSI_GPU_E2E=1 and a vLLM venv with GPUs")
MML = 4096
GMU = 0.35


@pytest.fixture(scope="module")
def pool():
    model = os.environ["FASTRSI_VLLM_MODEL"]
    python = os.environ.get("FASTRSI_VLLM_PYTHON", sys.executable)
    # the GPUs this job was given (CUDA_VISIBLE_DEVICES), not GPUs 0 and 1 of the host
    visible = [gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    gpus = (os.environ.get("FASTRSI_GPUS") or ",".join((visible or ["0", "1"])[:2])).split(",")
    if len(gpus) < 2 or not all(gpu.strip().isdigit() for gpu in gpus[:2]):
        pytest.skip(f"needs two GPU indices (FASTRSI_GPUS or CUDA_VISIBLE_DEVICES), got {gpus}: "
                    "set FASTRSI_GPUS=i,j")
    g0, g1 = int(gpus[0]), int(gpus[1])
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
        engine_a = W.Engine("A", g0, W._free_port(), work, h0)
        engine_s = W.Engine("S", g1, W._free_port(), work, h1)
        engines += [engine_a, engine_s]
        engine_a.start(model, python, GMU, MML)
        engine_s.start(model, python, GMU, MML)
        gpu_pool = W.Pool(model, python, work, engine_a, engine_s, W._free_port(),
                          max_model_len=MML)
        # the seeded holder serves the same model as the disk-loaded one
        assert W.digests_agree(gpu_pool.ref_digest, gpu_pool.digest(engine_s))["ok"]
        yield gpu_pool
        gpu_pool.close()
    finally:
        for started_engine in engines:
            started_engine.stop()
        h0.stop()
        h1.stop()
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(sock, ignore_errors=True)


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
