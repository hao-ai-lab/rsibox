"""Unit and concurrency tests for `rsibox.services.memory` (S1 MemoryService).

Pure CPU / stdlib tests — zero GPU, torch, numpy, or engine imports.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from rsibox.services import (  # noqa: E402
    DEFAULT_S0_STATE_KEYS,
    HandleNotFoundError,
    LeaseError,
    MemoryHandle,
    MemoryPoolKind,
    MemoryService,
    PoolAccessError,
    SnapshotLease,
    StampMismatchError,
)
from rsibox.warm import DiskStore, DynamoGmsStore, LaunchDelta, StoreProbe, stamp_diff  # noqa: E402


class _FakeWeightStore:
    """In-memory `WeightStore` stub tracking probe, ensure_ready, and stop calls."""

    def __init__(
        self,
        name: str = "dynamo_gms",
        *,
        sockets: tuple[str, ...] = ("/tmp/rsibox-test-gms-0.sock",),
        stamp: dict[str, Any] | None = None,
        size_bytes: int = 8192,
        fail_probe: str | None = None,
        on_ensure_ready: Any = None,
    ) -> None:
        self.name = name
        self.sockets = sockets
        self.stamp = stamp
        self.size_bytes = size_bytes
        self.fail_probe = fail_probe
        self.on_ensure_ready = on_ensure_ready
        self.ensure_ready_calls = 0
        self.probe_calls = 0
        self.probe_expected_stamps: list[dict[str, Any] | None] = []
        self.stop_calls = 0

    def ensure_ready(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
        timeout_s: float = 300.0,
    ) -> StoreProbe:
        self.ensure_ready_calls += 1
        if self.on_ensure_ready is not None:
            self.on_ensure_ready()
        return StoreProbe(ok=True, backend=self.name, sockets=self.sockets)

    def probe(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
        expected_stamp: dict[str, Any] | None = None,
    ) -> StoreProbe:
        self.probe_calls += 1
        self.probe_expected_stamps.append(expected_stamp)
        if self.fail_probe is not None:
            return StoreProbe(ok=False, backend=self.name, error=self.fail_probe)
        if expected_stamp is not None and self.stamp is not None:
            diff = stamp_diff(expected_stamp, self.stamp)
            if diff:
                return StoreProbe(
                    ok=False,
                    backend=self.name,
                    sockets=self.sockets,
                    stale_keys=diff,
                    error="stale store stamp",
                )
        details: dict[str, Any] = {"size_bytes": self.size_bytes}
        if self.stamp is not None:
            details["stamp"] = self.stamp
        return StoreProbe(ok=True, backend=self.name, sockets=self.sockets, details=details)

    def launch_delta(
        self,
        *,
        engine: str,
        explicit_flags: set[str] | frozenset[str] = frozenset(),
        explicit_values: dict[str, Any] | None = None,
        gpus: Sequence[str] = (),
    ) -> LaunchDelta:
        return LaunchDelta()

    def stop(self) -> None:
        self.stop_calls += 1


class MemoryServiceTests(unittest.TestCase):
    def test_attach_disk_and_gms_store_with_stamp_validation(self) -> None:
        svc = MemoryService()
        base_stamp = {
            "files": {"model.py": "abc123"},
            "weights": {"model.safetensors": [1024, 100]},
            "env": {"BACKEND": "fa3"},
            "extra": "tp1",
        }
        drifted_stamp = {
            "files": {"model.py": "def456"},
            "weights": {"model.safetensors": [1024, 100]},
            "env": {"BACKEND": "fa3"},
            "extra": "tp1",
        }

        disk = DiskStore()
        # DiskStore with stamp=None and non-empty expected_stamp must fail closed.
        with self.assertRaises(StampMismatchError):
            svc.attach_store(
                MemoryPoolKind.RO_WEIGHTS,
                disk,
                engine="vllm",
                role="weights",
                expected_stamp=base_stamp,
            )

        h_disk = svc.attach_store(
            MemoryPoolKind.RO_WEIGHTS,
            disk,
            engine="vllm",
            role="weights",
            gpus=(0,),
            stamp=base_stamp,
            expected_stamp=base_stamp,
        )
        self.assertIsInstance(h_disk, MemoryHandle)
        self.assertEqual(h_disk.pool_kind, MemoryPoolKind.RO_WEIGHTS)
        self.assertTrue(h_disk.read_only)
        self.assertEqual(h_disk.size_bytes, 0)

        # Mutating caller stamp dict must not affect stored handle stamp.
        base_stamp["files"]["model.py"] = "mutated"
        self.assertEqual(svc.get_handle(h_disk.handle_id).stamp["files"]["model.py"], "abc123")

        # Stale DiskStore attachment fails closed via stamp_diff.
        with self.assertRaises(StampMismatchError):
            svc.attach_store(
                MemoryPoolKind.RO_WEIGHTS,
                disk,
                engine="vllm",
                role="weights",
                gpus=(0,),
                stamp=drifted_stamp,
                expected_stamp={"files": {"model.py": "abc123"}, "weights": {"model.safetensors": [1024, 100]}, "env": {"BACKEND": "fa3"}, "extra": "tp1"},
            )

        # Fake GMS store succeeds on matching stamp and receives stamp in probe even when expected_stamp=None.
        clean_stamp = {
            "files": {"model.py": "abc123"},
            "weights": {"model.safetensors": [1024, 100]},
            "env": {"BACKEND": "fa3"},
            "extra": "tp1",
        }
        gms_store = _FakeWeightStore(name="dynamo_gms", stamp=clean_stamp, size_bytes=65536)
        h_gms = svc.attach_store(
            MemoryPoolKind.RO_WEIGHTS,
            gms_store,
            engine="sglang",
            role="inference",
            gpus=(0, 1),
            stamp=clean_stamp,
            ensure_ready=True,
        )
        self.assertEqual(gms_store.ensure_ready_calls, 1)
        self.assertEqual(gms_store.probe_expected_stamps[-1], clean_stamp)
        self.assertEqual(h_gms.gms_handle, "/tmp/rsibox-test-gms-0.sock")
        self.assertEqual(h_gms.size_bytes, 65536)

        # Mutating store stamp is caught by probe_all() even without passing expected_stamp.
        gms_store.stamp = drifted_stamp
        probe_res = svc.probe_all()
        self.assertFalse(probe_res["ok"])
        self.assertIn("files:model.py", probe_res["handles"][h_gms.handle_id]["stale_keys"])

        svc.teardown()

    def test_attach_real_dynamo_gms_store_mocked_sockets_and_stamp_drift(self) -> None:
        svc = MemoryService()
        stamp_ok = {"files": {"loader.py": "sha1"}, "weights": {}, "env": {}, "extra": None}
        stamp_bad = {"files": {"loader.py": "sha2"}, "weights": {}, "env": {}, "extra": None}
        with tempfile.TemporaryDirectory(dir="/tmp" if os.path.isdir("/tmp") else None) as tmp:
            sock_dir = os.path.join(tmp, "g")
            os.makedirs(sock_dir, exist_ok=True)
            gms = DynamoGmsStore(socket_dir=sock_dir, read_only=True)
            gms.write_stamp(stamp_ok)
            uuids = ("GPU-00000000-0000-0000-0000-000000000000",)
            with (
                mock.patch("rsibox.warm.dynamo_gms.resolve_gpu_uuids", return_value=uuids),
                mock.patch.object(gms, "_probe_nofile", return_value=(None, None)),
                mock.patch(
                    "rsibox.warm.dynamo_gms._probe_gms_socket",
                    return_value=(True, {"type": "get_runtime_state_response", "committed": True}),
                ),
            ):
                handle = svc.attach_store(
                    MemoryPoolKind.RO_WEIGHTS,
                    gms,
                    engine="vllm",
                    role="weights",
                    gpus=(0,),
                    expected_stamp=stamp_ok,
                    own_store=False,
                )
                self.assertEqual(handle.stamp, stamp_ok)
                self.assertTrue(handle.gms_handle and handle.gms_handle.startswith(sock_dir))

                # Drift stamp file on disk: probe_all detects drift without explicit expected_stamp.
                gms.write_stamp(stamp_bad)
                p = svc.probe_all()
                self.assertFalse(p["ok"])
                self.assertIn("files:loader.py", p["handles"][handle.handle_id]["stale_keys"])

                # Attaching with mismatched expected_stamp fails closed.
                with self.assertRaises(StampMismatchError):
                    svc.attach_store(
                        MemoryPoolKind.RO_WEIGHTS,
                        gms,
                        engine="vllm",
                        role="weights2",
                        gpus=(0,),
                        expected_stamp=stamp_ok,
                        own_store=False,
                    )
        svc.teardown()

    def test_attach_store_releases_lock_during_io_and_stops_owned_store_on_failure(self) -> None:
        svc = MemoryService()
        other_thread_saw_unlocked: list[bool] = []

        def check_unlocked_during_io() -> None:
            done = threading.Event()
            def probe_lock() -> None:
                svc.list_handles()
                other_thread_saw_unlocked.append(True)
                done.set()
            t = threading.Thread(target=probe_lock)
            t.start()
            t.join(timeout=2.0)
            self.assertTrue(done.is_set(), "MemoryService._lock was held across store I/O")

        failing_owned = _FakeWeightStore(
            name="dynamo_gms",
            fail_probe="socket timeout",
            on_ensure_ready=check_unlocked_during_io,
        )
        with self.assertRaises(PoolAccessError):
            svc.attach_store(
                MemoryPoolKind.RO_WEIGHTS,
                failing_owned,
                engine="vllm",
                role="weights",
                own_store=True,
                ensure_ready=True,
            )
        self.assertEqual(other_thread_saw_unlocked, [True])
        self.assertEqual(failing_owned.stop_calls, 1)

        # Foreign store (own_store=False) is NOT stopped on failure.
        failing_foreign = _FakeWeightStore(name="dynamo_gms", fail_probe="socket timeout")
        with self.assertRaises(PoolAccessError):
            svc.attach_store(
                MemoryPoolKind.RO_WEIGHTS,
                failing_foreign,
                engine="vllm",
                role="weights",
                own_store=False,
                ensure_ready=True,
            )
        self.assertEqual(failing_foreign.stop_calls, 0)
        svc.teardown()

    def test_readonly_enforcement_and_no_ro_escalation_in_share_handle(self) -> None:
        svc = MemoryService()
        for kind in (MemoryPoolKind.RO_WEIGHTS, MemoryPoolKind.RO_AUX_WEIGHTS):
            with self.assertRaises(PoolAccessError):
                svc.register_region(kind, "verl", "reference", gpus=(0,), size_bytes=1024, read_only=False)
            handle = svc.register_region(kind, "verl", "reference", gpus=(0,), size_bytes=1024)
            self.assertTrue(handle.read_only)
            with self.assertRaises(PoolAccessError):
                svc.acquire_lease(handle.handle_id, "worker-1", writable=True)
            lease = svc.acquire_lease(handle.handle_id, "worker-1", writable=False)
            self.assertTrue(svc.release_lease(lease.lease_id))

        # Any read-only handle (including KV_POOL / ACTOR_ROLLOUT_SHARED) cannot escalate to read_only=False.
        for kind in (MemoryPoolKind.KV_POOL, MemoryPoolKind.ACTOR_ROLLOUT_SHARED):
            h_ro = svc.register_region(kind, "vllm", "kv", gpus=(0,), size_bytes=2048, read_only=True)
            with self.assertRaises(PoolAccessError):
                svc.share_handle(h_ro.handle_id, "rollout", read_only=False)
            h_shared = svc.share_handle(h_ro.handle_id, "rollout")
            self.assertTrue(h_shared.read_only)
        svc.teardown()

    def test_cow_s0_single_writer_multi_reader_and_restore_epoch(self) -> None:
        svc = MemoryService()
        s0_stamp = {"files": {"trainer.py": "v1"}, "weights": {}, "env": {}, "extra": None}
        h_s0 = svc.register_region(
            MemoryPoolKind.COW_S0_SNAPSHOT, "torchtitan", "trainer", gpus=(0, 1), size_bytes=4096, stamp=s0_stamp
        )
        self.assertFalse(h_s0.read_only)

        # Multi-reader concurrent access succeeds.
        r1 = svc.acquire_lease(h_s0.handle_id, "reader-a", writable=False)
        r2 = svc.acquire_lease(h_s0.handle_id, "reader-b", writable=False)
        self.assertEqual(r1.epoch, 0)
        self.assertEqual(r1.epoch_state_keys, DEFAULT_S0_STATE_KEYS)
        self.assertEqual(svc.get_handle(h_s0.handle_id).refcount, 2)

        # Writer blocked while readers hold leases; reader lease cannot restore_s0.
        with self.assertRaises(PoolAccessError):
            svc.acquire_lease(h_s0.handle_id, "writer-a", writable=True)
        with self.assertRaises(PoolAccessError):
            svc.restore_s0(r1.lease_id)

        self.assertTrue(svc.release_lease(r1.lease_id))
        self.assertTrue(svc.release_lease(r2.lease_id))

        # Single writer succeeds; second writer and concurrent reader are refused.
        w1 = svc.acquire_lease(h_s0.handle_id, "writer-a", writable=True)
        self.assertIsInstance(w1, SnapshotLease)
        with self.assertRaises(PoolAccessError):
            svc.acquire_lease(h_s0.handle_id, "writer-b", writable=True)
        with self.assertRaises(PoolAccessError):
            svc.acquire_lease(h_s0.handle_id, "reader-c", writable=False)

        # Bool expected_epoch rejected; stamp mismatch rejected.
        with self.assertRaises(TypeError):
            svc.restore_s0(w1.lease_id, expected_epoch=True)
        with self.assertRaises(StampMismatchError):
            svc.restore_s0(w1.lease_id, expected_epoch=0, expected_stamp={"files": {"trainer.py": "stale"}})

        # Valid restore increments epoch cleanly.
        receipt = svc.restore_s0(w1.lease_id, expected_epoch=0, expected_stamp=s0_stamp)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["previous_epoch"], 0)
        self.assertEqual(receipt["epoch"], 1)
        self.assertEqual(receipt["restored_keys"], list(DEFAULT_S0_STATE_KEYS))

        with self.assertRaises(LeaseError):
            svc.restore_s0(w1.lease_id, expected_epoch=0)

        self.assertTrue(svc.release_lease(w1.lease_id))
        self.assertFalse(svc.release_lease(w1.lease_id))
        with self.assertRaises(LeaseError):
            svc.restore_s0(w1.lease_id)

        r3 = svc.acquire_lease(h_s0.handle_id, "reader-d", writable=False)
        self.assertEqual(r3.epoch, 1)
        svc.release_lease(r3.lease_id)
        svc.teardown()

    def test_share_handle_shared_lock_group_and_incarnation_fence(self) -> None:
        svc = MemoryService()
        stamp = {"files": {"actor.py": "h1"}, "weights": {"ckpt": [512, 1]}, "env": {}, "extra": "step10"}
        h_actor = svc.register_region(
            MemoryPoolKind.ACTOR_ROLLOUT_SHARED, "verl", "actor", gpus=(0, 1, 2, 3),
            size_bytes=16384, stamp=stamp, ipc_token="ipc://actor-weights-step10",
            read_only=False, incarnation=4,
        )

        # Cannot share while an exclusive writer holds the handle.
        w_lease = svc.acquire_lease(h_actor.handle_id, "actor-sync", writable=True)
        with self.assertRaises(PoolAccessError):
            svc.share_handle(h_actor.handle_id, "rollout")
        svc.release_lease(w_lease.lease_id)

        # Safe handoff to rollout preserves stamp & incarnation and shares lock group.
        h_rollout = svc.share_handle(h_actor.handle_id, "rollout", target_engine="sglang")
        self.assertEqual(h_rollout.incarnation, 4)
        self.assertEqual(h_rollout.stamp, stamp)
        self.assertTrue(h_rollout.read_only)

        # Active reader on h_rollout blocks writer on h_actor (cross-alias RW exclusion).
        r_rollout = svc.acquire_lease(h_rollout.handle_id, "rollout-0", writable=False)
        with self.assertRaises(PoolAccessError):
            svc.acquire_lease(h_actor.handle_id, "actor-sync", writable=True)
        svc.release_lease(r_rollout.lease_id)

        # Active writer on h_actor blocks reader on h_rollout.
        w_actor = svc.acquire_lease(h_actor.handle_id, "actor-sync", writable=True)
        with self.assertRaises(PoolAccessError):
            svc.acquire_lease(h_rollout.handle_id, "rollout-0", writable=False)
        self.assertEqual(svc.holder_died("actor-sync"), (w_actor.lease_id,))

        # Bumping incarnation updates the shared lock group monotonically across aliases.
        h_next = svc.share_handle(h_actor.handle_id, "rollout", bump_incarnation=True)
        self.assertEqual(h_next.incarnation, 5)
        self.assertEqual(svc.get_handle(h_actor.handle_id).incarnation, 5)
        self.assertEqual(svc.get_handle(h_rollout.handle_id).incarnation, 5)
        svc.teardown()

    def test_auto_handle_id_skips_manual_collisions(self) -> None:
        svc = MemoryService()
        manual = svc.register_region(
            MemoryPoolKind.RO_WEIGHTS, "vllm", "weights", size_bytes=512, handle_id="ro_weights:vllm:weights-1"
        )
        auto = svc.register_region(MemoryPoolKind.RO_WEIGHTS, "vllm", "weights", size_bytes=512)
        self.assertEqual(manual.handle_id, "ro_weights:vllm:weights-1")
        self.assertEqual(auto.handle_id, "ro_weights:vllm:weights-2")
        svc.teardown()

    def test_holder_died_and_owner_scoped_teardown_dedupes_shared_stores(self) -> None:
        svc = MemoryService()
        owned_store = _FakeWeightStore(name="dynamo_gms", size_bytes=2048)
        foreign_store = _FakeWeightStore(name="external_gms", size_bytes=4096)

        h_owned = svc.attach_store(MemoryPoolKind.RO_WEIGHTS, owned_store, engine="vllm", role="weights", own_store=True)
        h_owned_alias = svc.share_handle(h_owned.handle_id, "rollout")
        h_foreign = svc.attach_store(MemoryPoolKind.RO_WEIGHTS, foreign_store, engine="vllm", role="ref", own_store=False)
        h_s0 = svc.register_region(MemoryPoolKind.COW_S0_SNAPSHOT, "miles", "trainer", gpus=(0,), size_bytes=1024)

        l_ro = svc.acquire_lease(h_owned.handle_id, "dead-worker", writable=False)
        l_w = svc.acquire_lease(h_s0.handle_id, "dead-worker", writable=True)
        l_other = svc.acquire_lease(h_foreign.handle_id, "live-worker", writable=False)

        released = svc.holder_died("dead-worker")
        self.assertEqual(set(released), {l_ro.lease_id, l_w.lease_id})
        self.assertEqual(svc.get_handle(h_owned.handle_id).refcount, 0)
        self.assertEqual(svc.get_handle(h_s0.handle_id).refcount, 0)
        self.assertEqual(svc.get_handle(h_foreign.handle_id).refcount, 1)
        self.assertEqual(l_other.holder_id, "live-worker")
        self.assertEqual(h_owned_alias.role, "rollout")

        l_w2 = svc.acquire_lease(h_s0.handle_id, "live-worker", writable=True)
        svc.release_lease(l_w2.lease_id)

        probe = svc.probe_all()
        self.assertTrue(probe["ok"])
        self.assertEqual(probe["owned_stores"], 1)
        self.assertEqual(probe["active_leases"], 1)

        summary = svc.teardown()
        self.assertFalse(summary["already_closed"])
        self.assertEqual(summary["stopped_stores"], ["dynamo_gms"])
        # Shared alias referencing owned_store must NOT appear in skipped_foreign_stores.
        self.assertEqual(summary["skipped_foreign_stores"], ["external_gms"])
        self.assertEqual(owned_store.stop_calls, 1)
        self.assertEqual(foreign_store.stop_calls, 0)

        summary2 = svc.teardown()
        self.assertTrue(summary2["already_closed"])
        self.assertFalse(svc.probe_all()["ok"])
        with self.assertRaises(PoolAccessError):
            svc.register_region(MemoryPoolKind.RO_WEIGHTS, "vllm", "weights", size_bytes=100)
        with self.assertRaises(HandleNotFoundError):
            svc.get_handle(h_owned.handle_id)

    def test_concurrent_readers_and_writers_on_same_s0_handle_and_alias(self) -> None:
        svc = MemoryService()
        h_s0 = svc.register_region(
            MemoryPoolKind.COW_S0_SNAPSHOT, "torchtitan", "trainer", gpus=(0,), size_bytes=4096
        )
        h_s0_ro = svc.share_handle(h_s0.handle_id, "observer")

        errors: list[BaseException] = []
        active_readers = 0
        active_writers = 0
        max_concurrent_writers = 0
        writer_successes = 0
        obs_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def reader_worker(idx: int) -> None:
            nonlocal active_readers
            try:
                barrier.wait(timeout=5.0)
                target = h_s0.handle_id if idx % 2 == 0 else h_s0_ro.handle_id
                for _ in range(50):
                    try:
                        lease = svc.acquire_lease(target, f"r-{idx}", writable=False)
                    except PoolAccessError:
                        continue
                    with obs_lock:
                        self.assertEqual(active_writers, 0, "reader observed active writer")
                        active_readers += 1
                    try:
                        pass
                    finally:
                        with obs_lock:
                            active_readers -= 1
                        svc.release_lease(lease.lease_id)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def s0_worker(idx: int) -> None:
            nonlocal active_writers, max_concurrent_writers, writer_successes
            try:
                barrier.wait(timeout=5.0)
                for _ in range(50):
                    try:
                        lease = svc.acquire_lease(h_s0.handle_id, f"w-{idx}", writable=True)
                    except PoolAccessError:
                        continue
                    with obs_lock:
                        self.assertEqual(active_readers, 0, "writer observed active reader")
                        active_writers += 1
                        max_concurrent_writers = max(max_concurrent_writers, active_writers)
                    try:
                        svc.restore_s0(lease.lease_id)
                        with obs_lock:
                            writer_successes += 1
                    finally:
                        with obs_lock:
                            active_writers -= 1
                        svc.release_lease(lease.lease_id)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            *(threading.Thread(target=reader_worker, args=(i,)) for i in range(4)),
            *(threading.Thread(target=s0_worker, args=(i,)) for i in range(4)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
            self.assertFalse(t.is_alive())

        self.assertEqual(errors, [])
        self.assertGreater(writer_successes, 0)
        self.assertEqual(max_concurrent_writers, 1)
        self.assertEqual(svc.get_handle(h_s0.handle_id).refcount, 0)
        self.assertEqual(svc.get_handle(h_s0_ro.handle_id).refcount, 0)
        self.assertEqual(svc.probe_all()["active_leases"], 0)
        svc.teardown()


if __name__ == "__main__":
    unittest.main()
