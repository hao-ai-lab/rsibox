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
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from kernel import toy_lab
from rsibox import (
    PRIORITY_RANKS,
    PoolClient,
    PoolError,
    PoolServer,
    PoolSlot,
    PoolTimeoutError,
    SandboxPool,
    edit_once,
    short_sock_path,
    verify_trace,
)

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


if __name__ == "__main__":
    unittest.main()
