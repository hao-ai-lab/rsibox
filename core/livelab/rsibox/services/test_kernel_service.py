"""Unit, concurrency, process-quiesce, and SandboxPool tests for `KernelService` (`S7`)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import rsibox
from rsibox import (
    GateTimeoutError,
    KernelClosedError,
    KernelService,
    KernelServiceError,
    PoolError,
    PoolSlot,
    RegionGate,
    RegionTicket,
    SandboxPool,
    SlotNotFoundError,
    services,
)
from rsibox.services.kernel_service import (
    _set_cgroup_freeze,
    select_numa_cpu_slice,
)


class _DummyLab:
    def __init__(self) -> None:
        self.txns: list[str] = []

    def status(self) -> dict[str, object]:
        return {"ok": True, "state": "base", "txns": list(self.txns), "poison": False}

    def check(self, workload=None) -> dict[str, object]:
        return {"ok": True, "workload": workload}

    def reset(self, kind: str = "state") -> dict[str, object]:
        self.txns.clear()
        return {"ok": True, "kind": kind}

    def uninstall(self, id: str | None = None) -> dict[str, object]:
        self.txns.clear()
        return {"ok": True}

    def close(self) -> dict[str, object]:
        return {"ok": True}


class KernelServiceTests(unittest.TestCase):
    def test_exports_and_zero_gpu_imports(self) -> None:
        self.assertIs(services.KernelService, KernelService)
        self.assertIs(services.RegionGate, RegionGate)
        self.assertIs(services.RegionTicket, RegionTicket)
        for sym in ("KernelService", "RegionGate", "RegionTicket", "KernelServiceError", "SlotNotFoundError"):
            self.assertIn(sym, rsibox.__all__)
            self.assertIn(sym, services.__all__)
        for mod in ("torch", "triton", "cuda"):
            self.assertNotIn(mod, sys.modules)

    def test_registration_validation_and_closed_fail_closed(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            reg = svc.register_slot(0, "s0")
            self.assertEqual(reg.gpu_id, 0)
            self.assertEqual(reg.slot_id, "s0")
            self.assertEqual(reg.pid, os.getpid())
            self.assertTrue(reg.alive)
            self.assertEqual(len(svc.list_slots(0)), 1)
            self.assertTrue(svc.unregister_slot(0, "s0"))
            self.assertFalse(svc.unregister_slot(0, "s0"))

            with self.assertRaises(SlotNotFoundError):
                with svc.critical_region(0, "missing"):
                    pass
            with self.assertRaises(ValueError):
                svc.register_slot(-1, "bad")
            with self.assertRaises(ValueError):
                svc.register_slot(0, " ")

        with self.assertRaises(KernelClosedError):
            svc.register_slot(0, "s1")
        with self.assertRaises(KernelClosedError):
            with svc.critical_region(0, "s0"):
                pass

    def test_epoch_drain_thaw_before_unlock_and_exception_cleanup(self) -> None:
        events: list[str] = []
        gate_state_during_thaw: list[bool] = []

        with KernelService(pin_cpus=False) as svc:
            svc.register_slot(0, "s0")
            svc.register_slot(
                0,
                "s1",
                drain_fn=lambda ep: (events.append(f"drain:{ep}"), ep)[1],
                freeze_fn=lambda: events.append("freeze"),
                thaw_fn=lambda: (
                    gate_state_during_thaw.append(svc.get_gate(0).is_blocked()),
                    events.append("thaw"),
                ),
            )

            with svc.critical_region(
                0,
                "s0",
                sync_gpu_fn=lambda: events.append("sync"),
                flush_l2_fn=lambda: events.append("flush"),
            ) as ticket:
                self.assertEqual(ticket.epoch, 1)
                self.assertEqual(ticket.drained_slots, ("s1",))
                self.assertEqual(ticket.frozen_slots, ("s1",))
                self.assertEqual(ticket.freeze_mode, "callback")
                self.assertTrue(svc.get_gate(0).is_blocked())
                events.append("body")

            self.assertFalse(svc.get_gate(0).is_blocked())
            self.assertEqual(gate_state_during_thaw, [True])
            self.assertEqual(events, ["drain:1", "sync", "freeze", "flush", "body", "sync", "thaw"])

            # Exception inside critical_region must still sync, thaw before unlock, and release gate.
            events.clear()
            gate_state_during_thaw.clear()
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with svc.critical_region(
                    0,
                    "s0",
                    sync_gpu_fn=lambda: events.append("sync"),
                    flush_l2_fn=lambda: events.append("flush"),
                ) as ticket2:
                    self.assertEqual(ticket2.epoch, 2)
                    raise RuntimeError("boom")

            self.assertFalse(svc.get_gate(0).is_blocked())
            self.assertEqual(gate_state_during_thaw, [True])
            self.assertEqual(events, ["drain:2", "sync", "freeze", "flush", "sync", "thaw"])

    def test_drain_epoch_mismatch_aborts_and_leaves_gate_clean(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            svc.register_slot(0, "s0")
            svc.register_slot(0, "s1", drain_fn=lambda ep: ep + 99)
            with self.assertRaisesRegex(KernelServiceError, "drain failed"):
                with svc.critical_region(0, "s0"):
                    pass
            self.assertFalse(svc.get_gate(0).is_blocked())

    def test_per_gpu_exclusivity_and_cross_gpu_parallelism(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            svc.register_slot(0, "g0-a")
            svc.register_slot(0, "g0-b")
            svc.register_slot(1, "g1-a")

            active_g0 = 0
            max_g0 = 0
            epochs_g0: list[int] = []
            lock = threading.Lock()
            both_gpus_active = threading.Event()
            g0_entered = threading.Event()

            def run_g0(slot: str) -> None:
                nonlocal active_g0, max_g0
                for _ in range(3):
                    with svc.critical_region(0, slot) as ticket:
                        with lock:
                            active_g0 += 1
                            max_g0 = max(max_g0, active_g0)
                            epochs_g0.append(ticket.epoch)
                        time.sleep(0.01)
                        with lock:
                            active_g0 -= 1

            def hold_g0_and_check_g1() -> None:
                with svc.critical_region(0, "g0-a"):
                    g0_entered.set()
                    self.assertTrue(both_gpus_active.wait(timeout=2.0))

            def run_g1() -> None:
                self.assertTrue(g0_entered.wait(timeout=2.0))
                with svc.critical_region(1, "g1-a") as ticket:
                    self.assertEqual(ticket.gpu_id, 1)
                    both_gpus_active.set()

            t_hold = threading.Thread(target=hold_g0_and_check_g1)
            t_g1 = threading.Thread(target=run_g1)
            t_hold.start()
            t_g1.start()
            t_hold.join(timeout=3.0)
            t_g1.join(timeout=3.0)
            self.assertTrue(both_gpus_active.is_set())

            t1 = threading.Thread(target=run_g0, args=("g0-a",))
            t2 = threading.Thread(target=run_g0, args=("g0-b",))
            t1.start()
            t2.start()
            t1.join(timeout=3.0)
            t2.join(timeout=3.0)
            self.assertEqual(max_g0, 1)
            self.assertEqual(len(epochs_g0), 6)
            self.assertEqual(epochs_g0, sorted(epochs_g0))

    def test_submission_gate_blocks_and_times_out(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            svc.register_slot(0, "s0")
            svc.register_slot(0, "s1")
            inside = threading.Event()
            release = threading.Event()
            unblocked = threading.Event()

            def holder() -> None:
                with svc.critical_region(0, "s0"):
                    inside.set()
                    release.wait(timeout=2.0)

            t = threading.Thread(target=holder)
            t.start()
            self.assertTrue(inside.wait(timeout=2.0))

            with self.assertRaises(GateTimeoutError):
                svc.submission_gate(0, "s1", timeout_s=0.05)

            def waiter() -> None:
                svc.submission_gate(0, "s1", timeout_s=2.0)
                unblocked.set()

            wt = threading.Thread(target=waiter)
            wt.start()
            time.sleep(0.03)
            self.assertFalse(unblocked.is_set())
            release.set()
            wt.join(timeout=2.0)
            t.join(timeout=2.0)
            self.assertTrue(unblocked.is_set())

    def test_real_subprocess_sigstop_sigcont_pause_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            beat_file = Path(tmp) / "beats.txt"
            child_code = (
                "import pathlib, time\n"
                f"p = pathlib.Path({str(beat_file)!r})\n"
                "while True:\n"
                "    with p.open('a') as f: f.write('1\\n')\n"
                "    time.sleep(0.015)\n"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", child_code],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 3.0
                while (not beat_file.exists() or beat_file.stat().st_size == 0) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertGreater(beat_file.stat().st_size, 0)

                with KernelService(state_dir=tmp, pin_cpus=False) as svc:
                    svc.register_slot(0, "holder")
                    reg = svc.register_slot(0, "sib", pid=proc.pid, pgid=os.getpgid(proc.pid))
                    self.assertEqual(reg.pgid, proc.pid)

                    with svc.critical_region(0, "holder") as ticket:
                        self.assertEqual(ticket.frozen_slots, ("sib",))
                        self.assertEqual(ticket.freeze_mode, "sigstop_pg")
                        time.sleep(0.03)
                        size_frozen = beat_file.stat().st_size
                        time.sleep(0.12)
                        self.assertEqual(beat_file.stat().st_size, size_frozen)

                    # Thawed on exit: beats resume promptly.
                    resume_deadline = time.monotonic() + 2.0
                    while beat_file.stat().st_size <= size_frozen and time.monotonic() < resume_deadline:
                        time.sleep(0.01)
                    self.assertGreater(beat_file.stat().st_size, size_frozen)
            finally:
                proc.terminate()
                proc.wait(timeout=3.0)

    def test_cgroup2_freeze_and_permission_fallback_plus_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cg_dir = Path(tmp) / "slot_cg"
            cg_dir.mkdir()
            (cg_dir / "cgroup.freeze").write_text("0\n", encoding="utf-8")
            (cg_dir / "cgroup.events").write_text("populated 1\nfrozen 0\n", encoding="utf-8")

            # Simulate kernel updating cgroup.events when cgroup.freeze is written.
            orig_set = _set_cgroup_freeze
            writes: list[bool] = []

            def fake_cg(path: str, freeze: bool, *, timeout_s: float = 1.0) -> bool:
                writes.append(freeze)
                (cg_dir / "cgroup.events").write_text(f"populated 1\nfrozen {1 if freeze else 0}\n", encoding="utf-8")
                return orig_set(path, freeze, timeout_s=timeout_s)

            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                with KernelService(pin_cpus=False) as svc:
                    svc.register_slot(0, "holder")
                    svc.register_slot(0, "sib", pid=proc.pid, pgid=os.getpgid(proc.pid), cgroup_path=str(cg_dir))
                    with mock.patch("rsibox.services.kernel_service._set_cgroup_freeze", side_effect=fake_cg):
                        with svc.critical_region(0, "holder") as ticket:
                            self.assertEqual(ticket.freeze_mode, "cgroup2")
                            self.assertEqual(ticket.frozen_slots, ("sib",))
                        self.assertEqual(writes, [True, False])

                    # If cgroup2 write fails (PermissionError / unwritable), falls back to SIGSTOP.
                    with mock.patch("rsibox.services.kernel_service._set_cgroup_freeze", return_value=False):
                        with svc.critical_region(0, "holder") as ticket:
                            self.assertEqual(ticket.freeze_mode, "sigstop_pg")

                    # Identity mismatch (PID reuse / wrong start_ticks) refuses to signal.
                    entry = svc._slots[0]["sib"]
                    entry.start_ticks = 999999999
                    with mock.patch(
                        "rsibox.services.kernel_service._read_proc_identity",
                        return_value=(proc.pid, proc.pid, 12345),
                    ):
                        mode = svc._freeze_entry(entry, holder_pid=os.getpid(), holder_pgid=os.getpgrp(), timeout_s=1.0)
                        self.assertEqual(mode, "none")
            finally:
                proc.terminate()
                proc.wait(timeout=3.0)

    def test_holder_died_and_stale_ipc_gate_recovery(self) -> None:
        thawed: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            svc = KernelService(state_dir=tmp, pin_cpus=False)
            try:
                svc.register_slot(0, "s0")
                svc.register_slot(0, "s1", freeze_fn=lambda: None, thaw_fn=lambda: thawed.append("s1"))
                gate = svc.get_gate(0)
                gate._epoch = 3
                gate._blocked = True
                gate._holder_slot_id = "s0"
                svc._slots[0]["s1"].frozen = True
                svc._slots[0]["s1"].freeze_method = "callback"
                gate._write_shared_state(99999999, 1)

                rec = svc.holder_died(0, "s0")
                self.assertTrue(rec["recovered_gate"])
                self.assertEqual(rec["removed_slots"], ["s0"])
                self.assertEqual(rec["thawed_slots"], ["s1"])
                self.assertFalse(gate.is_blocked())
                self.assertEqual(thawed, ["s1"])

                # Next critical_region on s1 succeeds cleanly with bumped epoch.
                with svc.critical_region(0, "s1") as ticket:
                    self.assertGreaterEqual(ticket.epoch, 4)

                # Idempotent repeat call does nothing.
                rec2 = svc.holder_died(0, "s0")
                self.assertFalse(rec2["recovered_gate"])
                self.assertEqual(rec2["removed_slots"], [])
            finally:
                svc.close()

    def test_numa_cpu_slicing_and_affinity_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            (node_root / "node0").mkdir(parents=True)
            (node_root / "node1").mkdir(parents=True)
            (node_root / "node0" / "cpulist").write_text("0-3,8-11\n", encoding="utf-8")
            (node_root / "node1" / "cpulist").write_text("4-7,12-15\n", encoding="utf-8")

            s0 = select_numa_cpu_slice(0, node_root=str(node_root), allowed_cpus=frozenset(range(16)), gpus_per_node=4, cores_per_gpu=2)
            s1 = select_numa_cpu_slice(1, node_root=str(node_root), allowed_cpus=frozenset(range(16)), gpus_per_node=4, cores_per_gpu=2)
            s4 = select_numa_cpu_slice(4, node_root=str(node_root), allowed_cpus=frozenset(range(16)), gpus_per_node=4, cores_per_gpu=2)
            self.assertEqual(s0, (0, 1))
            self.assertEqual(s1, (2, 3))
            self.assertEqual(s4, (4, 5))

            set_calls: list[tuple[int, set[int]]] = []
            with KernelService(pin_cpus=True, node_root=str(node_root), gpus_per_node=4, cores_per_gpu=2) as svc:
                svc.register_slot(0, "s0")
                with (
                    mock.patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True),
                    mock.patch.object(os, "sched_setaffinity", side_effect=lambda p, m: set_calls.append((p, set(m))), create=True),
                ):
                    with svc.critical_region(0, "s0") as ticket:
                        self.assertEqual(ticket.pinned_cpus, (0, 1))
                    self.assertEqual(set_calls, [(0, {0, 1}), (0, {0, 1, 2, 3})])

    def test_cross_process_ipc_ack_and_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with KernelService(state_dir=tmp, pin_cpus=False) as svc:
                svc.register_slot(0, "s0")
                svc.register_slot(0, "s1")
                gate = svc.get_gate(0)
                gate.ack_drained("s1", 0)

                def acker() -> None:
                    while not gate.is_blocked():
                        time.sleep(0.005)
                    gate.ack_drained("s1", gate.epoch)

                t = threading.Thread(target=acker)
                t.start()
                with svc.critical_region(0, "s0", timeout_s=2.0) as ticket:
                    self.assertEqual(ticket.epoch, 1)
                    self.assertEqual(ticket.drained_slots, ("s1",))
                t.join(timeout=2.0)

            # Separate KernelService instances in distinct processes discover & freeze each other via state_dir.
            with KernelService(state_dir=tmp, pin_cpus=False) as svc_a, KernelService(state_dir=tmp, pin_cpus=False) as svc_b:
                proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
                try:
                    svc_a.register_slot(0, "worker-a")
                    svc_b.register_slot(0, "worker-b", pid=proc.pid, pgid=os.getpgid(proc.pid))
                    self.assertEqual({s.slot_id for s in svc_a.list_slots(0)}, {"worker-a", "worker-b"})
                    with svc_a.critical_region(0, "worker-a") as ticket:
                        self.assertEqual(ticket.frozen_slots, ("worker-b",))
                        self.assertEqual(ticket.freeze_mode, "sigstop_pg")
                finally:
                    proc.terminate()
                    proc.wait(timeout=3.0)

    def test_sandbox_pool_multi_slot_integration(self) -> None:
        with KernelService(pin_cpus=False) as ksvc:
            slots = [
                PoolSlot("slot-a", _DummyLab(), role="eval", metadata={"gpu_id": 0}),
                PoolSlot("slot-b", _DummyLab(), role="eval", metadata={"gpu_id": 0}),
                PoolSlot("slot-c", _DummyLab(), role="eval"),
            ]
            with SandboxPool(slots, kernel_service=ksvc, slots_per_gpu=2) as pool:
                self.assertEqual(pool.slots_per_gpu, 2)
                self.assertIs(pool.kernel_service, ksvc)
                self.assertEqual(pool.metrics()["slots_per_gpu"], 2)
                self.assertEqual(len(ksvc.list_slots(0)), 2)
                self.assertEqual(len(ksvc.list_slots(1)), 1)

                with pool.lease("agent-0") as lease:
                    with lease.critical_region() as ticket:
                        self.assertIsInstance(ticket, RegionTicket)
                        self.assertEqual(ticket.slot_id, lease.slot_id)

                pool.add_slot(PoolSlot("slot-d", _DummyLab(), role="eval", metadata={"gpu_id": 1}))
                self.assertEqual(len(ksvc.list_slots(1)), 2)
                pool.drain_slot("slot-d")
                self.assertEqual(len(ksvc.list_slots(1)), 1)

            self.assertEqual(ksvc.list_slots(), [])

    def test_cross_process_sigkill_holder_recovery_and_live_holder_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            beats = Path(tmp) / "beats.txt"
            flag = Path(tmp) / "holding.flag"
            sib_code = (
                "import pathlib, time\n"
                f"p = pathlib.Path({str(beats)!r})\n"
                "while True:\n"
                "    with p.open('a') as f: f.write('1\\n')\n"
                "    time.sleep(0.015)\n"
            )
            hold_code = (
                "import os, pathlib, sys, time\n"
                f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
                "from rsibox import KernelService\n"
                f"svc = KernelService(state_dir={tmp!r}, pin_cpus=False)\n"
                "svc.register_slot(0, 'holder', pid=os.getpid(), pgid=os.getpgid(0))\n"
                "with svc.critical_region(0, 'holder'):\n"
                f"    pathlib.Path({str(flag)!r}).write_text('1')\n"
                "    time.sleep(30)\n"
            )
            sib = subprocess.Popen([sys.executable, "-c", sib_code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            hold = None
            try:
                deadline = time.monotonic() + 3.0
                while (not beats.exists() or beats.stat().st_size == 0) and time.monotonic() < deadline:
                    time.sleep(0.01)
                with KernelService(state_dir=tmp, pin_cpus=False) as svc_reg:
                    svc_reg.register_slot(0, "sib", pid=sib.pid, pgid=os.getpgid(sib.pid))
                    hold = subprocess.Popen([sys.executable, "-c", hold_code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    hold_dl = time.monotonic() + 5.0
                    while not flag.exists() and time.monotonic() < hold_dl:
                        if hold.poll() is not None:
                            self.fail(f"holder exited early: {hold.returncode}")
                        time.sleep(0.01)
                    time.sleep(0.04)
                    frozen_size = beats.stat().st_size
                    time.sleep(0.10)
                    self.assertEqual(beats.stat().st_size, frozen_size)

                    # Alive holder: holder_died and close() on peer KernelService must NOT clobber gate or thaw sibling.
                    with KernelService(state_dir=tmp, pin_cpus=False) as peer:
                        rec_alive = peer.holder_died(0, "holder")
                        self.assertFalse(rec_alive["recovered_gate"])
                        self.assertEqual(rec_alive["removed_slots"], [])
                    self.assertTrue(svc_reg.get_gate(0).is_blocked())
                    self.assertEqual(svc_reg.get_gate(0).holder_slot_id, "holder")
                    self.assertEqual(beats.stat().st_size, frozen_size)

                    # SIGKILL the holder mid-region; peer must recover gate and SIGCONT the frozen sibling.
                    hold.kill()
                    hold.wait(timeout=3.0)
                    with KernelService(state_dir=tmp, pin_cpus=False) as recoverer:
                        rec_dead = recoverer.holder_died(0)
                        self.assertTrue(rec_dead["recovered_gate"])
                        self.assertIn("holder", rec_dead["removed_slots"])
                        self.assertIn("sib", rec_dead["thawed_slots"])
                        self.assertFalse(recoverer.get_gate(0).is_blocked())
                        self.assertGreaterEqual(recoverer.get_gate(0).epoch, 1)

                    resume_dl = time.monotonic() + 2.0
                    while beats.stat().st_size <= frozen_size and time.monotonic() < resume_dl:
                        time.sleep(0.01)
                    self.assertGreater(beats.stat().st_size, frozen_size)
            finally:
                if hold is not None and hold.poll() is None:
                    hold.kill()
                    hold.wait(timeout=3.0)
                sib.terminate()
                sib.wait(timeout=3.0)

    def test_reentrant_critical_region_fails_cleanly_without_deadlock(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            svc.register_slot(0, "s0")
            t0 = time.monotonic()
            with svc.critical_region(0, "s0", timeout_s=2.0) as outer:
                self.assertEqual(outer.epoch, 1)
                with self.assertRaisesRegex(KernelServiceError, "re-entrant"):
                    with svc.critical_region(0, "s0", timeout_s=1.0):
                        pass
                self.assertTrue(svc.get_gate(0).is_blocked())
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertFalse(svc.get_gate(0).is_blocked())

    def test_cgroup_freeze_timeout_reverts_to_zero_and_rejects_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cg_dir = Path(tmp) / "cg"
            cg_dir.mkdir()
            (cg_dir / "cgroup.freeze").write_text("0\n", encoding="utf-8")
            (cg_dir / "cgroup.events").write_text("populated 1\nfrozen 0\n", encoding="utf-8")
            # Timeout waiting for frozen 1 must revert cgroup.freeze back to "0".
            self.assertFalse(_set_cgroup_freeze(str(cg_dir), True, timeout_s=0.06))
            self.assertEqual((cg_dir / "cgroup.freeze").read_text(encoding="utf-8").strip(), "0")

            # Ancestor cgroup containing holder must be rejected, falling back to sigstop_pg.
            sub_cg = cg_dir / "holder_sub"
            sub_cg.mkdir()
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
            try:
                with KernelService(pin_cpus=False) as svc:
                    svc.register_slot(0, "holder")
                    svc.register_slot(0, "sib", pid=proc.pid, pgid=os.getpgid(proc.pid), cgroup_path=str(cg_dir))
                    with mock.patch("rsibox.services.kernel_service._read_proc_cgroup", return_value=str(sub_cg)):
                        with svc.critical_region(0, "holder") as ticket:
                            self.assertEqual(ticket.freeze_mode, "sigstop_pg")
            finally:
                proc.terminate()
                proc.wait(timeout=3.0)

    def test_pool_add_slot_wakeup_and_lease_critical_region_fencing(self) -> None:
        with KernelService(pin_cpus=False) as ksvc:
            with SandboxPool([PoolSlot("s0", _DummyLab(), metadata={"gpu_id": 0})], kernel_service=ksvc, slots_per_gpu=2) as pool:
                held = pool.acquire("a0")
                waiter_ticket: list[RegionTicket] = []
                waiter_err: list[BaseException] = []

                def waiter() -> None:
                    try:
                        with pool.lease("a1", timeout_s=2.0) as lease:
                            with lease.critical_region() as t:
                                waiter_ticket.append(t)
                    except BaseException as exc:
                        waiter_err.append(exc)

                wt = threading.Thread(target=waiter)
                wt.start()
                time.sleep(0.03)
                pool.add_slot(PoolSlot("s1", _DummyLab(), metadata={"gpu_id": 0}))
                wt.join(timeout=2.0)
                self.assertEqual(waiter_err, [])
                self.assertEqual(len(waiter_ticket), 1)
                self.assertEqual(waiter_ticket[0].slot_id, "s1")
                held.release()

                # release() waits for active lease.critical_region(), and post-release critical_region() fails.
                lease2 = pool.acquire("a2")
                entered = threading.Event()
                allow_exit = threading.Event()
                released_done = threading.Event()

                def hold_region() -> None:
                    with lease2.critical_region():
                        entered.set()
                        allow_exit.wait(timeout=2.0)

                rt = threading.Thread(target=hold_region)
                rt.start()
                self.assertTrue(entered.wait(timeout=2.0))
                rel_thread = threading.Thread(target=lambda: (lease2.release(), released_done.set()))
                rel_thread.start()
                time.sleep(0.03)
                self.assertFalse(released_done.is_set())
                allow_exit.set()
                rt.join(timeout=2.0)
                rel_thread.join(timeout=2.0)
                self.assertTrue(released_done.is_set())
                with self.assertRaises(PoolError):
                    with lease2.critical_region():
                        pass

    def test_p2_validation_guards(self) -> None:
        with KernelService(pin_cpus=False) as svc:
            for bad_id in ("../escape", "a/b", "a\\b", "bad\x00id", " "):
                with self.assertRaises(ValueError):
                    svc.register_slot(0, bad_id)
            svc.register_slot(0, "ok")
            for bad_to in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True):
                with self.assertRaises(ValueError):
                    with svc.critical_region(0, "ok", timeout_s=bad_to):
                        pass
            for bad_gpn in (0, -1, True):
                with self.assertRaises(ValueError):
                    select_numa_cpu_slice(0, gpus_per_node=bad_gpn)

    def test_remote_unregister_and_close_evicts_shared_slot_without_stopping_live_proc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            beats = Path(tmp) / "beats.txt"
            live_code = (
                "import pathlib, time\n"
                f"p = pathlib.Path({str(beats)!r})\n"
                "while True:\n"
                "    with p.open('a') as f: f.write('1\\n')\n"
                "    time.sleep(0.015)\n"
            )
            live_proc = subprocess.Popen([sys.executable, "-c", live_code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                dl = time.monotonic() + 3.0
                while (not beats.exists() or beats.stat().st_size == 0) and time.monotonic() < dl:
                    time.sleep(0.01)
                with KernelService(state_dir=tmp, pin_cpus=False) as svc_a:
                    svc_a.register_slot(0, "worker-a")
                    svc_b = KernelService(state_dir=tmp, pin_cpus=False)
                    svc_b.register_slot(0, "worker-b", pid=live_proc.pid, pgid=os.getpgid(live_proc.pid))
                    self.assertEqual({s.slot_id for s in svc_a.list_slots(0)}, {"worker-a", "worker-b"})
                    svc_b.close()
                    self.assertEqual({s.slot_id for s in svc_a.list_slots(0)}, {"worker-a"})
                    self.assertEqual({s.slot_id for s in svc_a.list_slots()}, {"worker-a"})

                    before = beats.stat().st_size
                    with svc_a.critical_region(0, "worker-a") as ticket:
                        self.assertEqual(ticket.frozen_slots, ())
                        time.sleep(0.08)
                        self.assertGreater(beats.stat().st_size, before)
            finally:
                live_proc.terminate()
                live_proc.wait(timeout=3.0)


if __name__ == "__main__":
    unittest.main()
