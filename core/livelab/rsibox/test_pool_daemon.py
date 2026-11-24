"""Golden tests for anti-starvation priority scheduling, failover hooks, and Unix-socket PoolServer/PoolClient.

Runs entirely on CPU against real `ToyLab` slots and real OS worker subprocesses (pure stdlib).

Run from the repository root:
    python3 -m unittest discover -s core/livelab/rsibox -t core/livelab -v
"""
from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from kernel import toy_lab
from rsibox import (
    PRIORITY_RANKS,
    PoolClient,
    PoolClosedError,
    PoolError,
    PoolServer,
    PoolSlot,
    PoolTimeoutError,
    SandboxPool,
    edit_once,
    pool_server as pool_server_module,
    short_sock_path,
    verify_trace,
)
from rsibox.box import Box

LIVELAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patch_activate_offset(base_ops_source: str, offset: int) -> dict[str, dict[str, str]]:
    patched = edit_once(base_ops_source, "    return x + 1", f"    return x + {offset}")
    return {"files": {"toy.ops": patched}}


def _expected_toy_output(ranks: list[str], offset: int) -> dict[str, list[int]]:
    inputs = toy_lab.WORKLOAD["inputs"]
    return {rank: [(value * 2) + offset + 10 for value in inputs] for rank in ranks}


def _subproc_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{LIVELAB_ROOT}{os.pathsep}{existing}" if existing else LIVELAB_ROOT
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


_WORKER_SCRIPT = """
import json, sys
from rsibox import PoolClient, edit_once

sock_path, agent_id, priority, offset = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
client = PoolClient(sock_path, timeout_s=30.0)
with client.acquire(agent_id=agent_id, priority=priority, idea_id=f"idea-{offset}", timeout_s=30.0) as lease:
    st0 = lease.status()
    assert st0["ok"] and st0["state"] == "base" and not st0["txns"], st0
    src = lease.source("toy.ops")
    assert src["ok"], src
    patched = edit_once(src["text"], "    return x + 1", f"    return x + {offset}")
    inst = lease.install({"files": {"toy.ops": patched}})
    assert inst["ok"], inst
    chk = lease.check()
    assert chk["ok"] and chk["proven"], chk
    # Intentionally leave the edit installed to prove server-side clean-slate rollback on release.
    print(json.dumps({
        "agent_id": agent_id,
        "slot_id": lease.slot_id,
        "priority": lease.priority,
        "proven": chk["proven"],
        "output": chk["output"],
    }), flush=True)
"""

_CRASH_WORKER_SCRIPT = """
import json, sys, time
from rsibox import PoolClient, edit_once

sock_path, agent_id = sys.argv[1], sys.argv[2]
client = PoolClient(sock_path, timeout_s=30.0)
lease = client.acquire(agent_id=agent_id, priority="dev", timeout_s=10.0)
src = lease.source("toy.ops")
assert src["ok"], src
patched = edit_once(src["text"], "    return x + 1", "    return x + 77")
inst = lease.install({"files": {"toy.ops": patched}})
assert inst["ok"], inst
print(json.dumps({"ready": True, "slot_id": lease.slot_id}), flush=True)
# Sleep without releasing; parent will SIGKILL this process.
time.sleep(60.0)
"""


class PoolPriorityFailoverDaemonTests(unittest.TestCase):
    def _make_toy_lab(self) -> toy_lab.ToyLab:
        lab = toy_lab.ToyLab(ranks=2, check_timeout=2.0, hang_s=0.5)
        self.addCleanup(lab.close)
        return lab

    def _wait_for_waiters(self, pool: SandboxPool, expected_count: int, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        observed = 0
        while time.monotonic() < deadline:
            with pool._lock:
                observed = len(pool._waiters)
            if observed >= expected_count:
                return
            time.sleep(0.002)
        self.fail(f"expected at least {expected_count} waiter(s), observed {observed}")

    def test_priority_lanes_and_anti_starvation_never_starve_dev(self):
        """Pain point: high-priority eval/fault bursts must jump ahead of queued dev jobs,
        while anti-starvation guarantees dev receives a slot after `max_priority_streak` jumps
        and every priority is recorded in the SHA-256 hash-chained trace."""
        self.assertLess(PRIORITY_RANKS["fault"], PRIORITY_RANKS["eval"])
        self.assertEqual(PRIORITY_RANKS["fault"], PRIORITY_RANKS["recovery"])
        self.assertLess(PRIORITY_RANKS["eval"], PRIORITY_RANKS["repair"])
        self.assertLess(PRIORITY_RANKS["repair"], PRIORITY_RANKS["dev"])

        lab = self._make_toy_lab()
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("bad-streak", lab)], max_priority_streak=-1)
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("bad-bool", lab)], max_priority_streak=True)

        with tempfile.TemporaryDirectory(prefix="rsibox-priority-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "priority.jsonl")

            # Part A: Full lane ordering (fault == recovery > eval > repair > dev) with streak=5.
            lane_pool = SandboxPool([PoolSlot("slot-0", lab)], max_priority_streak=5)
            self.addCleanup(lane_pool.close)
            with self.assertRaises(ValueError):
                lane_pool.acquire("bad-prio", priority="urgent")

            lane_order: list[str] = []
            holder = lane_pool.acquire("holder", priority="dev")
            queued_specs = [
                ("dev-1", "dev"),
                ("repair-1", "repair"),
                ("eval-1", "eval"),
                ("fault-1", "fault"),
                ("recovery-1", "recovery"),
            ]
            threads: list[threading.Thread] = []
            for idx, (name, prio) in enumerate(queued_specs, start=1):
                def worker(worker_name: str, worker_prio: str) -> None:
                    with lane_pool.lease(worker_name, priority=worker_prio, timeout_s=10.0) as leased:
                        self.assertEqual(leased.priority, worker_prio)
                        lane_order.append(worker_name)

                t = threading.Thread(target=worker, args=(name, prio))
                t.start()
                threads.append(t)
                self._wait_for_waiters(lane_pool, idx)
            holder.release()
            for t in threads:
                t.join(timeout=10.0)
                self.assertFalse(t.is_alive())
            self.assertEqual(lane_order, ["fault-1", "recovery-1", "eval-1", "repair-1", "dev-1"])
            lane_pool.close()

            # Part B: Anti-starvation with max_priority_streak=2.
            # Enqueue [dev-1, dev-2, eval-1, eval-2, eval-3, eval-4] behind a held slot.
            # Exact expected order: eval-1, eval-2, (yield!) dev-1, eval-3, eval-4, dev-2.
            pool = SandboxPool([PoolSlot("slot-0", lab)], trace_path=trace_path, max_priority_streak=2)
            self.addCleanup(pool.close)

            service_order: list[tuple[str, str]] = []
            hold_lease = pool.acquire("initial-holder", priority="dev")
            burst_specs = [
                ("dev-1", "dev"),
                ("dev-2", "dev"),
                ("eval-1", "eval"),
                ("eval-2", "eval"),
                ("eval-3", "eval"),
                ("eval-4", "eval"),
            ]
            burst_threads: list[threading.Thread] = []
            for idx, (name, prio) in enumerate(burst_specs, start=1):
                def burst_worker(worker_name: str, worker_prio: str) -> None:
                    outcome = pool.run(
                        worker_name,
                        lambda leased_lab, leased: (
                            service_order.append((worker_name, leased.priority)),
                            leased_lab.status()["state"],
                        )[1],
                        priority=worker_prio,
                        timeout_s=10.0,
                    )
                    self.assertTrue(outcome["ok"], outcome)
                    self.assertEqual(outcome["priority"], worker_prio)

                t = threading.Thread(target=burst_worker, args=(name, prio))
                t.start()
                burst_threads.append(t)
                self._wait_for_waiters(pool, idx)

            hold_lease.release()
            for t in burst_threads:
                t.join(timeout=10.0)
                self.assertFalse(t.is_alive())

            self.assertEqual(
                service_order,
                [
                    ("eval-1", "eval"),
                    ("eval-2", "eval"),
                    ("dev-1", "dev"),
                    ("eval-3", "eval"),
                    ("eval-4", "eval"),
                    ("dev-2", "dev"),
                ],
            )
            self.assertEqual(pool.metrics()["starvation_yields"], 1)

            # Verify SHA-256 hash-chained trace includes `priority` and detects priority tampering.
            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["count"], 7)
            recorded_priorities = [(rec["agent_id"], rec["priority"]) for rec in verified["records"]]
            self.assertEqual(
                recorded_priorities,
                [
                    ("initial-holder", "dev"),
                    ("eval-1", "eval"),
                    ("eval-2", "eval"),
                    ("dev-1", "dev"),
                    ("eval-3", "eval"),
                    ("eval-4", "eval"),
                    ("dev-2", "dev"),
                ],
            )

            with open(trace_path, encoding="utf-8") as handle:
                lines = handle.readlines()
            tampered = json.loads(lines[1])
            tampered["priority"] = "fault"
            lines[1] = json.dumps(tampered, sort_keys=True) + "\n"
            with open(trace_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
            tamper_result = verify_trace(trace_path)
            self.assertFalse(tamper_result["ok"])
            self.assertEqual(tamper_result["line"], 2)

    def test_failover_hook_fast_promotion_fallback_to_recover_and_sibling_isolation(self):
        """Pain point: poisoned slot triggers `failover_fn` (recording `recovery_mode='failover'`)
        without stalling sibling slots, falls back to `recover_fn` (`recovery_mode='recover'`) if
        `failover_fn` raises, and quarantines fail-closed if both fail."""
        slot_zero_lab = self._make_toy_lab()
        slot_one_lab = self._make_toy_lab()
        standby_lab = self._make_toy_lab()
        base_output = slot_zero_lab.check()["output"]

        failover_entered = threading.Event()
        allow_failover_finish = threading.Event()
        self.addCleanup(allow_failover_finish.set)
        failover_calls: list[tuple[str, str]] = []
        recover_calls: list[tuple[str, str]] = []
        failover_mode = ["promote"]

        def custom_failover(slot: PoolSlot, lab, reason: str):
            failover_calls.append((slot.slot_id, reason))
            if failover_mode[0] == "promote":
                failover_entered.set()
                allow_failover_finish.wait(timeout=10.0)
                return standby_lab  # Shadow promotion returning a clean replacement Lab!
            raise RuntimeError("shadow pool empty")

        def custom_recover(slot: PoolSlot, lab, reason: str):
            recover_calls.append((slot.slot_id, reason))
            return lab.reset("respawn")

        with tempfile.TemporaryDirectory(prefix="rsibox-failover-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "failover.jsonl")
            pool = SandboxPool(
                [PoolSlot("slot-0", slot_zero_lab), PoolSlot("slot-1", slot_one_lab)],
                trace_path=trace_path,
                failover_fn=custom_failover,
                recover_fn=custom_recover,
            )
            self.addCleanup(pool.close)

            # Case 1: Poison slot-0 -> fast failover promotes standby_lab while slot-1 keeps serving.
            poison_box: dict = {}

            def poison_slot_zero() -> None:
                poison_box["outcome"] = pool.run_isolated(
                    "poison-agent",
                    lambda lab, lease: (lab.inject("crash"), lab.check())[1],
                    timeout_s=10.0,
                )

            t = threading.Thread(target=poison_slot_zero)
            t.start()
            self.assertTrue(failover_entered.wait(timeout=10.0), "expected slot-0 to invoke failover_fn")

            # Healthy sibling slot-1 must serve immediately with recovery_mode='none'.
            sibling = pool.run_isolated(
                "healthy-sibling",
                lambda lab, lease: (
                    lab.install(_patch_activate_offset(lab.base_text["toy.ops"], offset=11)),
                    lab.check(),
                )[1],
                timeout_s=5.0,
            )
            self.assertTrue(sibling["ok"], sibling)
            self.assertEqual(sibling["slot_id"], "slot-1")
            self.assertEqual(sibling["recovery_mode"], "none")
            self.assertFalse(sibling["recovered"])
            self.assertTrue(sibling["result"]["proven"])

            allow_failover_finish.set()
            t.join(timeout=10.0)
            self.assertFalse(t.is_alive())

            failover_outcome = poison_box["outcome"]
            self.assertTrue(failover_outcome["ok"], failover_outcome)
            self.assertEqual(failover_outcome["slot_id"], "slot-0")
            self.assertTrue(failover_outcome["recovered"])
            self.assertEqual(failover_outcome["recovery_mode"], "failover")
            self.assertEqual(failover_outcome["status_after"], "base")
            self.assertEqual(len(failover_calls), 1)
            self.assertEqual(len(recover_calls), 0)
            self.assertEqual(pool.metrics()["failovers"], 1)
            self.assertEqual(pool.metrics()["recoveries"], 1)

            # Verify slot-0 now serves cleanly from the promoted standby_lab!
            promoted_check = pool.run_isolated(
                "post-failover",
                lambda lab, lease: (lease.slot_id, lab.status()["state"], lab.check()["output"]),
                timeout_s=5.0,
            )
            self.assertTrue(promoted_check["ok"], promoted_check)
            self.assertEqual(promoted_check["result"], ("slot-0", "base", base_output))

            # Case 2: failover_fn raises -> automatic fallback to recover_fn (`recovery_mode='recover'`).
            failover_mode[0] = "raise"
            fallback_outcome = pool.run_isolated(
                "fallback-agent",
                lambda lab, lease: (lab.inject("crash"), lab.check())[1],
                timeout_s=10.0,
            )
            self.assertTrue(fallback_outcome["ok"], fallback_outcome)
            self.assertTrue(fallback_outcome["recovered"])
            self.assertEqual(fallback_outcome["recovery_mode"], "recover")
            self.assertEqual(fallback_outcome["status_after"], "base")
            self.assertEqual(len(failover_calls), 2)
            self.assertEqual(len(recover_calls), 1)
            self.assertEqual(pool.metrics()["failovers"], 1)
            self.assertEqual(pool.metrics()["recoveries"], 2)

            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            modes_by_agent = {rec["agent_id"]: rec["recovery_mode"] for rec in verified["records"]}
            self.assertEqual(modes_by_agent["poison-agent"], "failover")
            self.assertEqual(modes_by_agent["healthy-sibling"], "none")
            self.assertEqual(modes_by_agent["fallback-agent"], "recover")

    def test_multiprocess_pool_server_contention_witness_fencing_and_sigkill_recovery(self):
        """Pain point: independent OS processes contending over `PoolServer`/`PoolClient` must get
        fenced leases, clean-slate witness proofs, valid SHA-256 traces, post-release rejection,
        and instant server-side cleanup & recovery when a worker is `SIGKILL`ed mid-lease."""
        lab_zero = self._make_toy_lab()
        lab_one = self._make_toy_lab()
        base_output = lab_zero.check()["output"]
        sock_path = short_sock_path("rsibox-pool-mp")

        with tempfile.TemporaryDirectory(prefix="rsibox-daemon-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "daemon_leases.jsonl")
            pool = SandboxPool(
                [PoolSlot("slot-0", lab_zero), PoolSlot("slot-1", lab_one)],
                trace_path=trace_path,
                max_priority_streak=2,
            )
            self.addCleanup(pool.close)
            server = PoolServer(pool, sock_path)
            self.addCleanup(server.close)

            # 1. Launch 6 real subprocess workers concurrently against 2 warm ToyLab slots.
            procs: list[tuple[int, subprocess.Popen[str]]] = []
            env = _subproc_env()
            for idx in range(6):
                offset = idx + 4
                prio = "eval" if idx % 2 == 0 else "dev"
                proc = subprocess.Popen(
                    [sys.executable, "-c", _WORKER_SCRIPT, sock_path, f"proc-agent-{idx}", prio, str(offset)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                procs.append((offset, proc))

            seen_slots: set[str] = set()
            for offset, proc in procs:
                stdout, stderr = proc.communicate(timeout=30.0)
                self.assertEqual(proc.returncode, 0, f"worker failed: stderr={stderr}\nstdout={stdout}")
                payload = json.loads(stdout.strip().splitlines()[-1])
                self.assertTrue(payload["proven"], payload)
                self.assertEqual(payload["output"], _expected_toy_output(lab_zero.ranks, offset=offset))
                seen_slots.add(payload["slot_id"])

            self.assertEqual(seen_slots, {"slot-0", "slot-1"})
            client = PoolClient(sock_path, timeout_s=10.0)
            stats = client.stats()
            self.assertTrue(stats["ok"], stats)
            self.assertEqual(stats["total_leases"], 6)
            self.assertEqual(stats["active_leases"], 0)
            health = client.health()
            self.assertTrue(health["ok"], health)
            self.assertEqual(health["healthy_slots"], 2)
            self.assertEqual(health["in_use_slots"], 0)

            # 2. Verify client-side and server-side fencing after release.
            lease = client.acquire("fence-agent", priority="eval", timeout_s=5.0)
            captured_token = lease.token
            self.assertEqual(lease.status()["state"], "base")
            record = lease.release()
            self.assertEqual(record["agent_id"], "fence-agent")
            self.assertEqual(record["priority"], "eval")
            with self.assertRaises(PoolError):
                lease.status()
            with self.assertRaises(PoolError):
                lease.check()

            # Raw socket attempt with released token must be rejected server-side.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw_sock:
                raw_sock.settimeout(5.0)
                raw_sock.connect(sock_path)
                raw_sock.sendall((json.dumps({"op": "status", "token": captured_token}) + "\n").encode("utf-8"))
                raw_reply = json.loads(raw_sock.makefile("rb").readline().decode("utf-8"))
            self.assertFalse(raw_reply["ok"])
            self.assertEqual(raw_reply.get("error_type"), "PoolError")

            # 3. Abrupt SIGKILL crash test on a 1-slot server to prove ZERO slot leak and full rollback.
            crash_lab = self._make_toy_lab()
            crash_sock = short_sock_path("rsibox-pool-kill")
            crash_trace = os.path.join(temp_dir, "crash_leases.jsonl")
            crash_pool = SandboxPool([PoolSlot("solo-0", crash_lab)], trace_path=crash_trace)
            self.addCleanup(crash_pool.close)
            crash_server = PoolServer(crash_pool, crash_sock)
            self.addCleanup(crash_server.close)

            crash_proc = subprocess.Popen(
                [sys.executable, "-c", _CRASH_WORKER_SCRIPT, crash_sock, "doomed-agent"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                ready_line = crash_proc.stdout.readline()
                if not ready_line:
                    self.fail(f"crash worker exited early: {crash_proc.stderr.read()}")
                ready_msg = json.loads(ready_line)
                self.assertEqual(ready_msg["slot_id"], "solo-0")

                crash_client = PoolClient(crash_sock, timeout_s=10.0)
                self.assertEqual(crash_client.health()["in_use_slots"], 1)
                # While doomed-agent holds solo-0, another acquire with short timeout must time out.
                with self.assertRaises(PoolTimeoutError):
                    crash_client.acquire("blocked-agent", timeout_s=0.1)

                # Hard-kill the worker with SIGKILL (no finally, no release RPC!).
                os.kill(crash_proc.pid, signal.SIGKILL)
                crash_proc.wait(timeout=5.0)
            finally:
                if crash_proc.poll() is None:
                    crash_proc.kill()
                    crash_proc.wait(timeout=5.0)
                if crash_proc.stdout:
                    crash_proc.stdout.close()
                if crash_proc.stderr:
                    crash_proc.stderr.close()

            # Server must immediately detect EOF, roll back the uncommitted edit, and serve next client!
            with crash_client.acquire("survivor-agent", priority="recovery", timeout_s=5.0) as survivor:
                self.assertEqual(survivor.slot_id, "solo-0")
                st = survivor.status()
                self.assertEqual(st["state"], "base")
                self.assertEqual(st["txns"], [])
                chk = survivor.check()
                self.assertTrue(chk["ok"], chk)
                self.assertEqual(chk["output"], base_output)

            # Verify both trace files have unbroken SHA-256 chains!
            daemon_verified = verify_trace(trace_path)
            self.assertTrue(daemon_verified["ok"], daemon_verified)
            self.assertEqual(daemon_verified["count"], 7)

            crash_verified = verify_trace(crash_trace)
            self.assertTrue(crash_verified["ok"], crash_verified)
            self.assertEqual(crash_verified["count"], 2)
            self.assertEqual(crash_verified["records"][0]["agent_id"], "doomed-agent")
            self.assertIn("disconnected", crash_verified["records"][0]["error"])
            self.assertEqual(crash_verified["records"][0]["status_after"], "base")
            self.assertEqual(crash_verified["records"][1]["agent_id"], "survivor-agent")
            self.assertEqual(crash_verified["records"][1]["priority"], "recovery")

    def test_cli_pool_serve_health_and_stats_end_to_end(self):
        """Proves `rsibox pool serve`, `rsibox pool health`, and `rsibox pool stats` work
        end-to-end as real CLI subcommands and shut down cleanly on SIGTERM."""
        sock_path = short_sock_path("rsibox-pool-cli")
        env = _subproc_env()
        with tempfile.TemporaryDirectory(prefix="rsibox-cli-pool-") as temp_dir:
            trace_path = os.path.join(temp_dir, "cli_pool.jsonl")
            daemon = subprocess.Popen(
                [
                    sys.executable, "-m", "rsibox", "pool", "serve",
                    "--sock", sock_path, "--lab", "toy", "--slots", "2", "--trace", trace_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                ready_line = daemon.stdout.readline()
                if not ready_line:
                    self.fail(f"daemon failed to start: {daemon.stderr.read()}")
                ready = json.loads(ready_line)
                self.assertTrue(ready["ok"], ready)
                self.assertEqual(ready["slots"], 2)

                health_proc = subprocess.run(
                    [sys.executable, "-m", "rsibox", "pool", "health", "--sock", sock_path],
                    capture_output=True, text=True, timeout=10.0, env=env, check=True,
                )
                health = json.loads(health_proc.stdout)
                self.assertTrue(health["ok"], health)
                self.assertEqual(health["healthy_slots"], 2)

                client = PoolClient(sock_path, timeout_s=10.0)
                with client.acquire("cli-agent", priority="eval", timeout_s=5.0) as lease:
                    self.assertTrue(lease.check()["ok"])

                stats_proc = subprocess.run(
                    [sys.executable, "-m", "rsibox", "pool", "stats", "--sock", sock_path],
                    capture_output=True, text=True, timeout=10.0, env=env, check=True,
                )
                stats = json.loads(stats_proc.stdout)
                self.assertTrue(stats["ok"], stats)
                self.assertEqual(stats["total_leases"], 1)

                daemon.send_signal(signal.SIGTERM)
                daemon.wait(timeout=10.0)
                self.assertEqual(daemon.returncode, 0)
            finally:
                if daemon.poll() is None:
                    daemon.kill()
                    daemon.wait(timeout=5.0)
                if daemon.stdout:
                    daemon.stdout.close()
                if daemon.stderr:
                    daemon.stderr.close()

            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["count"], 1)
            self.assertEqual(verified["records"][0]["priority"], "eval")

    def test_p1_1_failover_replacement_lifecycle_aliasing_and_quarantine(self):
        """Regression for P1-1: same-Box failover preserves slot.box in-place, sibling-aliased
        replacements are rejected without closing sibling, dirty replacements are closed before
        fallback, clean swaps close displaced old labs when close_labs=True, and double failure
        quarantines fail-closed."""
        # Case A: Returning the same Box treats recovery as in-place without clearing slot.box.
        boxed = Box(lambda receipts=None: toy_lab.ToyLab(ranks=2, receipts=receipts), spec="toy")
        boxed.boot()
        self.addCleanup(boxed.close)
        box_closed = [0]
        orig_box_close = boxed.close
        boxed.close = lambda: (box_closed.__setitem__(0, box_closed[0] + 1), orig_box_close())[1]

        def inplace_box_failover(slot, lab, reason):
            lab.reset("respawn")
            return boxed

        box_slot = PoolSlot("box-0", boxed)
        box_pool = SandboxPool([box_slot], close_labs=True, failover_fn=inplace_box_failover)
        out = box_pool.run_isolated("a-box", lambda lab, lease: (lab.inject("crash"), lab.check())[1])
        self.assertTrue(out["ok"] and out["recovered"] and out["recovery_mode"] == "failover", out)
        self.assertIs(box_slot.box, boxed)
        self.assertEqual(box_closed[0], 0)
        box_pool.close(close_labs=True)
        self.assertEqual(box_closed[0], 1)

        # Case B/C/D: Alias rejection, dirty-candidate close + fallback, clean swap closes old lab,
        # and both hooks failing quarantines fail-closed.
        lab0, lab1 = self._make_toy_lab(), self._make_toy_lab()
        dirty_cand = self._make_toy_lab()
        dirty_cand.inject("crash")
        dirty_cand.check()
        standby_lab = self._make_toy_lab()

        closed_tags: list[str] = []
        for tag, target in (("lab0", lab0), ("lab1", lab1), ("dirty", dirty_cand)):
            orig = target.close
            target.close = (lambda t=tag, fn=orig: (closed_tags.append(t), fn())[1])

        mode = ["alias"]
        recover_calls = [0]

        def failover_hook(slot, lab, reason):
            if mode[0] == "alias":
                return lab1
            if mode[0] == "dirty":
                return dirty_cand
            if mode[0] == "swap":
                return standby_lab
            return {"ok": False, "error": "failover unavailable"}

        def recover_hook(slot, lab, reason):
            recover_calls[0] += 1
            if mode[0] == "both_fail":
                return {"ok": False, "error": "recover unavailable"}
            return lab.reset("respawn")

        pool = SandboxPool(
            [PoolSlot("s0", lab0), PoolSlot("s1", lab1)],
            close_labs=True, failover_fn=failover_hook, recover_fn=recover_hook,
        )
        self.addCleanup(pool.close)

        # B: Alias to s1 rejected without closing s1; falls back to recover_fn.
        r_alias = pool.run_isolated("u-alias", lambda lab, lease: (lab.inject("crash"), lab.check())[1])
        self.assertEqual(r_alias["recovery_mode"], "recover")
        self.assertEqual(closed_tags, [])
        self.assertEqual(lab1.status()["state"], "base")

        # C1: Dirty candidate rejected AND closed before falling back to recover_fn.
        mode[0] = "dirty"
        r_dirty = pool.run_isolated("u-dirty", lambda lab, lease: (lab.inject("crash"), lab.check())[1])
        self.assertEqual(r_dirty["recovery_mode"], "recover")
        self.assertEqual(closed_tags, ["dirty"])

        # C2: Clean swap promotes standby_lab AND closes displaced lab0.
        mode[0] = "swap"
        r_swap = pool.run_isolated("u-swap", lambda lab, lease: (lab.inject("crash"), lab.check())[1])
        self.assertEqual(r_swap["recovery_mode"], "failover")
        self.assertEqual(closed_tags, ["dirty", "lab0"])

        # D: Both hooks fail -> s0 quarantined fail-closed, s1 unaffected.
        mode[0] = "both_fail"
        r_fail = pool.run_isolated("u-fail", lambda lab, lease: (lab.inject("crash"), lab.check())[1])
        self.assertEqual(r_fail["status_after"], "quarantined")
        self.assertEqual(r_fail["recovery_mode"], "none")
        self.assertFalse(r_fail["recovered"])
        self.assertEqual(pool.metrics()["quarantined_slots"], ["s0"])
        r_sib = pool.run_isolated("u-sib", lambda lab, lease: (lease.slot_id, lab.check()["ok"]))
        self.assertEqual(r_sib["result"], ("s1", True))

    def test_p1_2_remote_lease_timeout_closes_socket_and_prevents_desync(self):
        """Regression for P1-2: timed-out RemoteLease RPC marks lease released, closes the
        socket so the server rolls back the edit, blocks subsequent RPCs, avoids desync on
        release(), and PoolClient.acquire wraps transport errors as PoolError."""
        with self.assertRaises(PoolError):
            PoolClient(short_sock_path("rsibox-missing")).acquire("a0", timeout_s=0.5)

        lab = self._make_toy_lab()
        base_output = lab.check()["output"]
        entered_check = threading.Event()
        unblock_check = threading.Event()
        self.addCleanup(unblock_check.set)
        orig_check = lab.check
        slow_once = [True]

        def gated_check(workload=None):
            if slow_once[0]:
                slow_once[0] = False
                entered_check.set()
                unblock_check.wait(timeout=5.0)
            return orig_check(workload)

        lab.check = gated_check
        sock_path = short_sock_path("rsibox-p12")
        with tempfile.TemporaryDirectory(prefix="rsibox-p12-") as temp_dir:
            trace_path = os.path.join(temp_dir, "p12.jsonl")
            pool = SandboxPool([PoolSlot("s0", lab)], trace_path=trace_path)
            self.addCleanup(pool.close)
            server = PoolServer(pool, sock_path)
            self.addCleanup(server.close)

            client = PoolClient(sock_path, timeout_s=0.15)
            lease = client.acquire("timeout-agent", timeout_s=5.0)
            src = lease.source("toy.ops")
            self.assertTrue(lease.install(_patch_activate_offset(src["text"], offset=55))["ok"])

            with self.assertRaises(PoolError):
                lease.check()
            self.assertTrue(lease.released)
            with self.assertRaises(PoolError):
                lease.status()
            self.assertEqual(lease.release(), {})

            self.assertTrue(entered_check.wait(timeout=2.0))
            unblock_check.set()

            next_client = PoolClient(sock_path, timeout_s=5.0)
            with next_client.acquire("next-agent", timeout_s=5.0) as clean:
                self.assertEqual(clean.status()["state"], "base")
                self.assertEqual(clean.check()["output"], base_output)
            self.assertTrue(verify_trace(trace_path)["ok"])

    def test_p1_3_socket_safety_permissions_inode_unlink_and_frame_cap(self):
        """Regression for P1-3: PoolServer refuses regular files and live sockets, enforces
        0o600 mode, preserves unrelated inode on close(), and enforces max JSON line cap."""
        lab1, lab2 = self._make_toy_lab(), self._make_toy_lab()
        pool1 = SandboxPool([PoolSlot("s1", lab1)])
        pool2 = SandboxPool([PoolSlot("s2", lab2)])
        self.addCleanup(pool1.close)
        self.addCleanup(pool2.close)

        with tempfile.TemporaryDirectory(prefix="rsibox-p13-") as temp_dir:
            reg_file = os.path.join(temp_dir, "keep.txt")
            with open(reg_file, "w", encoding="utf-8") as fh:
                fh.write("do-not-delete")
            with self.assertRaises(PoolError):
                PoolServer(pool1, reg_file)
            with open(reg_file, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "do-not-delete")

        sock_path = short_sock_path("rsibox-p13")
        server1 = PoolServer(pool1, sock_path)
        self.addCleanup(server1.close)
        self.assertEqual(stat.S_IMODE(os.lstat(sock_path).st_mode), 0o600)

        # Live daemon must NOT be clobbered by a second PoolServer on the same sock_path.
        with self.assertRaises(PoolError):
            PoolServer(pool2, sock_path)
        self.assertTrue(PoolClient(sock_path, timeout_s=5.0).health()["ok"])

        # Oversized frame rejected cleanly without crashing the server.
        with mock.patch.object(pool_server_module, "_MAX_LINE_BYTES", 128):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
                raw.settimeout(5.0)
                raw.connect(sock_path)
                raw.sendall((json.dumps({"op": "stats", "pad": "x" * 256}) + "\n").encode("utf-8"))
                reply = json.loads(raw.makefile("rb").readline().decode("utf-8"))
            self.assertFalse(reply["ok"])
            self.assertEqual(reply.get("error_type"), "ValueError")
        self.assertTrue(PoolClient(sock_path, timeout_s=5.0).health()["ok"])

        # If path inode is replaced externally before close(), close() must NOT unlink the new file.
        os.unlink(sock_path)
        with open(sock_path, "w", encoding="utf-8") as fh:
            fh.write("replaced")
        self.addCleanup(lambda: os.path.exists(sock_path) and os.unlink(sock_path))
        server1.close()
        self.assertTrue(os.path.isfile(sock_path))

    def test_p1_4_server_close_without_closing_pool_unblocks_blocked_acquire_fast(self):
        """Regression for P1-4: PoolServer.close(close_pool=False) returns in <2s even when a
        remote client is blocked inside acquire(), leaving zero zombie waiters on SandboxPool."""
        lab = self._make_toy_lab()
        pool = SandboxPool([PoolSlot("s0", lab)])
        self.addCleanup(pool.close)
        sock_path = short_sock_path("rsibox-p14")
        server = PoolServer(pool, sock_path, close_pool=False)

        held = pool.acquire("direct-holder")
        errors: list[BaseException] = []

        def blocked_remote() -> None:
            try:
                PoolClient(sock_path, timeout_s=30.0).acquire("remote-waiter", timeout_s=30.0)
            except BaseException as exc:
                errors.append(exc)

        waiter_thread = threading.Thread(target=blocked_remote)
        waiter_thread.start()
        deadline = time.monotonic() + 3.0
        while server.stats()["active_connections"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)

        t0 = time.monotonic()
        server.close(close_pool=False)
        elapsed = time.monotonic() - t0
        waiter_thread.join(timeout=3.0)

        self.assertLess(elapsed, 2.0, f"server.close(close_pool=False) took {elapsed:.2f}s")
        self.assertFalse(waiter_thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], (PoolClosedError, PoolError))
        self.assertEqual(pool.health()["queued_waiters"], 0)

        # Pool itself remains open and fully functional after releasing direct holder.
        held.release()
        with pool.lease("after-close", timeout_s=2.0) as post_lease:
            self.assertEqual(post_lease.slot_id, "s0")
            self.assertTrue(post_lease.lab.check()["ok"])

    def test_remote_waiters_preserve_fifo_anti_starvation_and_true_queue_wait_s(self):
        """Regression for Round-2 P1: in-place `cancel_check` keeps remote waiters in
        `SandboxPool._waiters` across waits >0.25s so FIFO ordering, `max_priority_streak`
        anti-starvation, and true `queue_wait_s` are all preserved."""
        lab = self._make_toy_lab()
        sock_path = short_sock_path("rsibox-r2-fifo")
        pool = SandboxPool([PoolSlot("s0", lab)], max_priority_streak=2)
        self.addCleanup(pool.close)
        server = PoolServer(pool, sock_path)
        self.addCleanup(server.close)

        held = pool.acquire("initial-holder")
        order: list[str] = []
        waits: dict[str, float] = {}
        order_lock = threading.Lock()

        def remote_worker(agent_id: str, priority: str) -> None:
            client = PoolClient(sock_path, timeout_s=15.0)
            with client.acquire(agent_id, priority=priority, timeout_s=15.0) as leased:
                with order_lock:
                    order.append(agent_id)
                    waits[agent_id] = leased.queue_wait_s

        # Queue remote `dev` first, then 3 remote `eval` waiters while holding the slot > 0.25s.
        threads: list[threading.Thread] = []
        specs = [("remote-dev-1", "dev"), ("remote-eval-1", "eval"), ("remote-eval-2", "eval"), ("remote-eval-3", "eval")]
        t_hold_start = time.monotonic()
        for expected_q, (aid, prio) in enumerate(specs, start=1):
            t = threading.Thread(target=remote_worker, args=(aid, prio))
            t.start()
            threads.append(t)
            self._wait_for_waiters(pool, expected_q, timeout_s=5.0)

        remaining_hold = 0.28 - (time.monotonic() - t_hold_start)
        if remaining_hold > 0:
            time.sleep(remaining_hold)
        held.release()

        for t in threads:
            t.join(timeout=10.0)
            self.assertFalse(t.is_alive())

        # Anti-starvation: remote-dev-1 MUST run right after 2 evals (index 2, not starved to last!).
        self.assertEqual(order, ["remote-eval-1", "remote-eval-2", "remote-dev-1", "remote-eval-3"])
        self.assertGreaterEqual(pool.metrics()["starvation_yields"], 1)
        # True total queue wait (> 0.20s), NOT truncated to <= 0.1s!
        self.assertGreater(waits["remote-dev-1"], 0.20, waits)


if __name__ == "__main__":
    unittest.main()
