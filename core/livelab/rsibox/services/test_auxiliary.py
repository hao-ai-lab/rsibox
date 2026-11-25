"""Unit, concurrency, and MemoryService integration tests for `rsibox.services.auxiliary` (S4).

Pure CPU / stdlib tests — zero GPU, torch, numpy, or engine imports.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Any

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from rsibox.services import (  # noqa: E402
    AuxClosedError,
    AuxiliaryModelService,
    AuxInvocationError,
    AuxInvocationReceipt,
    AuxServiceError,
    AuxStageKind,
    AuxStageSpec,
    MemoryPoolKind,
    MemoryService,
    StageExistsError,
    StageNotFoundError,
    StampMismatchError,
)


@dataclass(frozen=True)
class _PayloadItem:
    prompt: str
    step: int


class _PipeResource:
    """Stub holding an open `os.pipe()` pair so any leaked close hook triggers `ResourceWarning`."""

    def __init__(self) -> None:
        r_fd, w_fd = os.pipe()
        self._r = os.fdopen(r_fd, "rb")
        self._w = os.fdopen(w_fd, "wb")
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        for fh in (self._r, self._w):
            if not fh.closed:
                fh.close()


class _InterceptingMemoryService:
    """Thin delegating wrapper around `MemoryService` for deterministic race hooks."""

    def __init__(self, inner: MemoryService, on_acquire: Any = None) -> None:
        self._inner = inner
        self._on_acquire = on_acquire

    def get_handle(self, handle_id: str) -> Any:
        return self._inner.get_handle(handle_id)

    def acquire_lease(self, handle_id: str, holder_id: str, writable: bool = False, *, expected_stamp: Any = None) -> Any:
        lease = self._inner.acquire_lease(handle_id, holder_id, writable=writable, expected_stamp=expected_stamp)
        if self._on_acquire is not None:
            self._on_acquire(handle_id, lease)
        return lease

    def release_lease(self, lease_id: str) -> bool:
        return self._inner.release_lease(lease_id)


class AuxiliaryModelServiceValidationTests(unittest.TestCase):
    def test_input_validation_metadata_deepcopy_and_error_guards(self) -> None:
        for bad_cap in (True, False, -1, "8"):
            with self.assertRaises((TypeError, ValueError)):
                AuxiliaryModelService(default_cache_entries=bad_cap)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            AuxiliaryModelService(memory_service=object())  # type: ignore[arg-type]

        with AuxiliaryModelService() as svc:
            for bad_str in ("", "   ", None, 42):
                with self.assertRaises(ValueError):
                    svc.register_stage(bad_str, kind="vae", engine="fastvideo", model_id="vae-1", infer_fn=lambda p, c: p)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    svc.register_stage("s-bad", kind="vae", engine=bad_str, model_id="vae-1", infer_fn=lambda p, c: p)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    svc.register_stage("s-bad", kind="vae", engine="fastvideo", model_id=bad_str, infer_fn=lambda p, c: p)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    svc.register_stage("s-bad", kind="vae", engine="fastvideo", model_id="vae-1", version=bad_str, infer_fn=lambda p, c: p)  # type: ignore[arg-type]

            with self.assertRaises(ValueError):
                svc.register_stage("s-kind", kind="invalid_kind", engine="vllm", model_id="m", infer_fn=lambda p, c: p)
            with self.assertRaises(TypeError):
                svc.register_stage("s-fn", kind="judge", engine="vllm", model_id="m", infer_fn="not-callable")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_stage("s-ck", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, cache_key_fn="bad")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_stage("s-cl", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, close_fn="bad")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_stage("s-own", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, own_handle="yes")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_stage("s-meta", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, metadata=["bad"])  # type: ignore[arg-type]
            for bad_gpu in ((True,), (-1,), "0"):
                with self.assertRaises((TypeError, ValueError)):
                    svc.register_stage("s-gpu", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, gpus=bad_gpu)  # type: ignore[arg-type]
            for bad_cap in (True, -1, 1.5):
                with self.assertRaises((TypeError, ValueError)):
                    svc.register_stage("s-cap", kind="judge", engine="vllm", model_id="m", infer_fn=lambda p, c: p, max_cache_entries=bad_cap)  # type: ignore[arg-type]

            # handle_id / new_handle_id / own_handle / hold_lease fail closed without memory_service.
            with self.assertRaises(AuxServiceError):
                svc.register_stage("s-no-mem", kind="judge", engine="vllm", model_id="m", handle_id="h1", infer_fn=lambda p, c: p)
            with self.assertRaises(AuxServiceError):
                svc.register_stage("s-own-no-mem", kind="judge", engine="vllm", model_id="m", own_handle=True, infer_fn=lambda p, c: p)

            raw_meta: dict[str, Any] = {"nested": ["a"]}
            spec = svc.register_stage(
                "judge-1", kind=AuxStageKind.JUDGE, engine="verl", model_id="rm-7b",
                infer_fn=lambda p, c: c["metadata"], metadata=raw_meta,
            )
            self.assertIsInstance(spec, AuxStageSpec)
            raw_meta["nested"].append("mutated-caller")
            spec.metadata["nested"].append("mutated-spec")
            self.assertEqual(svc.get_stage("judge-1").metadata, {"nested": ["a"]})

            with self.assertRaises(AuxServiceError):
                svc.bump_version("judge-1", "v2", new_handle_id="h2")
            with self.assertRaises(AuxServiceError):
                svc.invoke("judge-1", "worker-0", {"x": 1}, hold_lease=True)
            with self.assertRaises(StageExistsError):
                svc.register_stage("judge-1", kind="judge", engine="verl", model_id="rm-7b", infer_fn=lambda p, c: p)
            with self.assertRaises(StageNotFoundError):
                svc.get_stage("missing")
            with self.assertRaises(StageNotFoundError):
                svc.invoke("missing", "worker-0", {"x": 1})
            with self.assertRaises(AuxServiceError):
                svc.invoke("judge-1", "worker-0", {"x": 1}, expected_stamp={"weights": {"w": [1, 1]}})
            with self.assertRaises(TypeError):
                svc.invoke("judge-1", "worker-0", {"x": 1}, use_cache=1)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.invoke("judge-1", "worker-0", {"x": 1}, hold_lease=1)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.invoke("judge-1", "worker-0", {"x": 1}, expected_stamp="bad")  # type: ignore[arg-type]


class AuxiliaryMemoryIntegrationTests(unittest.TestCase):
    def test_ro_aux_weights_lease_stamp_hold_lease_and_own_handle(self) -> None:
        mem = MemoryService()
        stage_mem = MemoryService()
        aux_stamp = {"files": {"reward.py": "sha-1"}, "weights": {"rm.safetensors": [4096, 10]}, "env": {}, "extra": "v1"}
        stale_stamp = {"files": {"reward.py": "sha-2"}, "weights": {"rm.safetensors": [4096, 10]}, "env": {}, "extra": "v1"}
        try:
            h_aux = mem.register_region(
                MemoryPoolKind.RO_AUX_WEIGHTS, "verl", "reward", gpus=(2, 3), size_bytes=4096, stamp=aux_stamp, handle_id="aux-rm-1"
            )
            h_ro = mem.register_region(
                MemoryPoolKind.RO_WEIGHTS, "fastvideo", "vae", gpus=(0,), size_bytes=2048, stamp=aux_stamp, handle_id="ro-vae-1"
            )
            h_actor_rw = mem.register_region(
                MemoryPoolKind.ACTOR_ROLLOUT_SHARED, "verl", "actor", gpus=(0,), size_bytes=8192, read_only=False, handle_id="actor-rw"
            )
            h_actor_ro = mem.register_region(
                MemoryPoolKind.ACTOR_ROLLOUT_SHARED, "verl", "actor_ro", gpus=(0,), size_bytes=8192, read_only=True, handle_id="actor-ro"
            )
            h_s0 = mem.register_region(
                MemoryPoolKind.COW_S0_SNAPSHOT, "megatron", "s0", gpus=(0,), size_bytes=8192, handle_id="s0-rw"
            )
            stage_mem.register_region(
                MemoryPoolKind.RO_AUX_WEIGHTS, "verl", "owned_rm", gpus=(1,), size_bytes=1024, handle_id="owned-rm-1"
            )

            with AuxiliaryModelService(memory_service=mem) as svc:
                for bad_hid in (h_actor_rw.handle_id, h_actor_ro.handle_id, h_s0.handle_id):
                    with self.assertRaises(AuxServiceError):
                        svc.register_stage(
                            f"bad-{bad_hid}", kind="reward_model", engine="verl", model_id="rm", handle_id=bad_hid, infer_fn=lambda p, c: p
                        )

                observed_refcounts: list[int] = []
                call_count = 0

                def reward_infer(payload: Any, ctx: dict[str, Any]) -> dict[str, Any]:
                    nonlocal call_count
                    call_count += 1
                    observed_refcounts.append(mem.get_handle(h_aux.handle_id).refcount)
                    self.assertIsNotNone(ctx["lease_id"])
                    return {"score": len(payload["prompt"]) * 0.5, "caller": ctx["caller_id"]}

                spec = svc.register_stage(
                    "rm-main", kind=AuxStageKind.REWARD_MODEL, engine="verl", model_id="rm-7b",
                    handle_id=h_aux.handle_id, infer_fn=reward_infer,
                )
                self.assertEqual(spec.gpus, (2, 3))
                self.assertTrue(spec.read_only)

                vae_spec = svc.register_stage(
                    "vae-main", kind=AuxStageKind.VAE, engine="fastvideo", model_id="wan-vae",
                    handle_id=h_ro.handle_id, infer_fn=lambda p, c: {"latents": p["frames"]},
                )
                self.assertEqual(vae_spec.gpus, (0,))

                rec1 = svc.invoke("rm-main", "rollout-0", {"prompt": "hello"}, expected_stamp=aux_stamp)
                self.assertIsInstance(rec1, AuxInvocationReceipt)
                self.assertFalse(rec1.cache_hit)
                self.assertIsNotNone(rec1.lease_id)
                self.assertEqual(rec1.output["score"], 2.5)
                self.assertEqual(observed_refcounts, [1])
                self.assertEqual(mem.get_handle(h_aux.handle_id).refcount, 0)

                # Cache hit verifies expected_stamp via RO lease and releases refcount cleanly.
                rec2 = svc.invoke("rm-main", "rollout-1", {"prompt": "hello"}, expected_stamp=aux_stamp)
                self.assertTrue(rec2.cache_hit)
                self.assertIsNotNone(rec2.lease_id)
                self.assertEqual(call_count, 1)
                self.assertEqual(mem.get_handle(h_aux.handle_id).refcount, 0)

                # hold_lease=True retains RO lease on both cache hit and cache miss until caller releases.
                rec_held = svc.invoke("rm-main", "rollout-hold", {"prompt": "hello"}, hold_lease=True)
                self.assertTrue(rec_held.cache_hit)
                self.assertIsNotNone(rec_held.lease_id)
                self.assertEqual(mem.get_handle(h_aux.handle_id).refcount, 1)
                assert rec_held.lease_id is not None
                self.assertTrue(mem.release_lease(rec_held.lease_id))
                self.assertEqual(mem.get_handle(h_aux.handle_id).refcount, 0)

                # Stale expected_stamp must fail closed with StampMismatchError even on a cached input!
                with self.assertRaises(StampMismatchError):
                    svc.invoke("rm-main", "rollout-2", {"prompt": "hello"}, expected_stamp=stale_stamp)
                self.assertEqual(mem.get_handle(h_aux.handle_id).refcount, 0)
                self.assertEqual(call_count, 1)

                # own_handle=True on stage-owned MemoryService tears down stage_mem on unregister_stage.
                svc.register_stage(
                    "owned-rm", kind="reward_model", engine="verl", model_id="rm-small",
                    handle_id="owned-rm-1", memory_service=stage_mem, own_handle=True,
                    infer_fn=lambda p, c: p,
                )
                svc.unregister_stage("owned-rm")
                self.assertTrue(stage_mem.probe_all()["closed"])
                # Shared service-level mem remains open.
                self.assertFalse(mem.probe_all()["closed"])

                with self.assertRaises(AuxServiceError):
                    svc.bump_version("rm-main", "v2", new_handle_id=h_actor_rw.handle_id)
        finally:
            stage_mem.teardown()
            mem.teardown()


class AuxiliaryCacheAndDigestTests(unittest.TestCase):
    def test_type_tagged_canonical_digest_prevents_collisions_and_rejects_bad_types(self) -> None:
        with AuxiliaryModelService() as svc:
            svc.register_stage("echo", kind="custom", engine="vllm", model_id="m", infer_fn=lambda p, c: p)

            # Type collisions that plain JSON/default used to conflate must produce distinct digests.
            d_int = svc.invoke("echo", "w0", 1).input_digest
            d_bool = svc.invoke("echo", "w0", True).input_digest
            d_list = svc.invoke("echo", "w0", [1, 2]).input_digest
            d_tuple = svc.invoke("echo", "w0", (1, 2)).input_digest
            d_bytes = svc.invoke("echo", "w0", b"abc").input_digest
            import hashlib
            d_dict_fake = svc.invoke("echo", "w0", {"__bytes_sha256__": hashlib.sha256(b"abc").hexdigest()}).input_digest
            d_path = svc.invoke("echo", "w0", Path("/tmp/a")).input_digest
            d_str_path = svc.invoke("echo", "w0", "/tmp/a").input_digest
            self.assertEqual(len({d_int, d_bool}), 2)
            self.assertEqual(len({d_list, d_tuple}), 2)
            self.assertEqual(len({d_bytes, d_dict_fake}), 2)
            self.assertEqual(len({d_path, d_str_path}), 2)

            # Non-finite floats and unsupported types fail closed without cache_key_fn.
            for bad_payload in (float("nan"), float("inf"), float("-inf"), object()):
                with self.assertRaises(AuxInvocationError):
                    svc.invoke("echo", "w0", bad_payload)

    def test_lru_cache_eviction_and_version_bump(self) -> None:
        calls = 0

        def encoder_infer(payload: Any, ctx: dict[str, Any]) -> str:
            nonlocal calls
            calls += 1
            return f"{ctx['version']}:{calls}"

        with AuxiliaryModelService(default_cache_entries=2) as svc:
            svc.register_stage(
                "txt-enc", kind=AuxStageKind.TEXT_ENCODER, engine="transformers", model_id="t5-xxl", infer_fn=encoder_infer
            )
            r1 = svc.invoke("txt-enc", "w0", {"b": 2, "a": 1})
            r2 = svc.invoke("txt-enc", "w1", {"a": 1, "b": 2})
            self.assertFalse(r1.cache_hit)
            self.assertTrue(r2.cache_hit)
            self.assertEqual(r1.input_digest, r2.input_digest)
            self.assertEqual(calls, 1)

            svc.invoke("txt-enc", "w0", b"raw-bytes-payload")
            self.assertTrue(svc.invoke("txt-enc", "w0", {"a": 1, "b": 2}).cache_hit)
            svc.invoke("txt-enc", "w0", _PayloadItem("p3", 3))
            self.assertEqual(calls, 3)

            self.assertTrue(svc.invoke("txt-enc", "w0", {"a": 1, "b": 2}).cache_hit)
            self.assertFalse(svc.invoke("txt-enc", "w0", b"raw-bytes-payload").cache_hit)
            self.assertEqual(calls, 4)

            r_nocache = svc.invoke("txt-enc", "w0", {"uncached": True}, use_cache=False)
            self.assertFalse(r_nocache.cache_hit)
            self.assertFalse(svc.invoke("txt-enc", "w0", {"uncached": True}).cache_hit)
            self.assertEqual(calls, 6)

            bumped = svc.bump_version("txt-enc", "v2")
            self.assertEqual(bumped.version, "v2")
            r_v2 = svc.invoke("txt-enc", "w0", {"a": 1, "b": 2})
            self.assertFalse(r_v2.cache_hit)
            self.assertEqual(r_v2.version, "v2")
            self.assertEqual(svc.stats()["stages"]["txt-enc"]["cache_entries"], 1)


class AuxiliaryRaceAndTeardownTests(unittest.TestCase):
    def test_invoke_bump_version_new_handle_race_aba_and_cross_stage_poisoning(self) -> None:
        mem = MemoryService()
        try:
            mem.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "verl", "rm_v1", gpus=(0,), size_bytes=1024, handle_id="h1")
            mem.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "verl", "rm_v2", gpus=(1,), size_bytes=1024, handle_id="h2")

            # P1-1: bump_version(..., new_handle_id="h2") between acquire_lease("h1") and post-lease lock check.
            bounced = False
            svc_ref: list[AuxiliaryModelService] = []

            def on_acquire(hid: str, _lease: Any) -> None:
                nonlocal bounced
                if hid == "h1" and not bounced:
                    bounced = True
                    svc_ref[0].bump_version("rm", "v2", new_handle_id="h2")

            wrapped_mem = _InterceptingMemoryService(mem, on_acquire=on_acquire)
            with AuxiliaryModelService(memory_service=wrapped_mem) as svc:  # type: ignore[arg-type]
                svc_ref.append(svc)
                seen_versions: list[str] = []

                def infer_race(payload: Any, ctx: dict[str, Any]) -> str:
                    seen_versions.append(ctx["version"])
                    self.assertEqual(ctx["handle_id"], "h2")
                    self.assertEqual(mem.get_handle("h1").refcount, 0)
                    self.assertEqual(mem.get_handle("h2").refcount, 1)
                    return f"{ctx['version']}:{payload}"

                svc.register_stage("rm", kind="reward_model", engine="verl", model_id="rm", version="v1", handle_id="h1", infer_fn=infer_race)
                rec = svc.invoke("rm", "w0", "q")
                self.assertEqual(rec.version, "v2")
                self.assertEqual(seen_versions, ["v2"])
                self.assertEqual(mem.get_handle("h1").refcount, 0)
                self.assertEqual(mem.get_handle("h2").refcount, 0)

            # P1-2a: use_cache=False post-lease guard catches unregister_stage before calling infer_fn.
            unreg_called = False
            svc_unreg_ref: list[AuxiliaryModelService] = []

            def on_acquire_unreg(_hid: str, _lease: Any) -> None:
                svc_unreg_ref[0].unregister_stage("rm-unreg")

            with AuxiliaryModelService(memory_service=_InterceptingMemoryService(mem, on_acquire=on_acquire_unreg)) as svc_u:  # type: ignore[arg-type]
                svc_unreg_ref.append(svc_u)

                def never_called(_p: Any, _c: dict[str, Any]) -> Any:
                    nonlocal unreg_called
                    unreg_called = True
                    return "bad"

                svc_u.register_stage("rm-unreg", kind="reward_model", engine="verl", model_id="rm", handle_id="h1", infer_fn=never_called)
                with self.assertRaises(StageNotFoundError):
                    svc_u.invoke("rm-unreg", "w0", "q", use_cache=False)
                self.assertFalse(unreg_called)
                self.assertEqual(mem.get_handle("h1").refcount, 0)

            # P1-2b: bump_version ABA (v1 -> v2 -> v1) and unregister+reregister never poison new entry's cache.
            entered, release = threading.Event(), threading.Event()
            with AuxiliaryModelService() as svc_aba:
                def slow_v1(p: Any, ctx: dict[str, Any]) -> str:
                    if not entered.is_set():
                        entered.set()
                        release.wait(timeout=3.0)
                        return "stale-v1"
                    return f"fresh-{ctx['version']}-{p}"

                svc_aba.register_stage("judge", kind="judge", engine="miles", model_id="j", version="v1", infer_fn=slow_v1)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(svc_aba.invoke, "judge", "a0", "q1")
                    self.assertTrue(entered.wait(timeout=2.0))
                    svc_aba.bump_version("judge", "v2")
                    svc_aba.bump_version("judge", "v1")
                    release.set()
                    self.assertEqual(fut.result(timeout=5.0).output, "stale-v1")
                    self.assertEqual(svc_aba.stats()["stages"]["judge"]["cache_entries"], 0)
                    self.assertEqual(svc_aba.invoke("judge", "a0", "q1").output, "fresh-v1-q1")

                # Cross-stage unregister + re-register under same stage_id.
                entered.clear()
                release.clear()
                svc_aba.register_stage("xstage", kind="judge", engine="miles", model_id="j", infer_fn=slow_v1)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut2 = pool.submit(svc_aba.invoke, "xstage", "a0", "q2")
                    self.assertTrue(entered.wait(timeout=2.0))
                    svc_aba.unregister_stage("xstage")
                    svc_aba.register_stage("xstage", kind="judge", engine="miles", model_id="j", infer_fn=lambda p, c: "new-stage-out")
                    release.set()
                    self.assertEqual(fut2.result(timeout=5.0).output, "stale-v1")
                    st = svc_aba.stats()["stages"]["xstage"]
                    self.assertEqual(st["cache_entries"], 0)
                    self.assertEqual(st["cache_misses"], 0)
                    self.assertEqual(svc_aba.invoke("xstage", "a0", "q2").output, "new-stage-out")
        finally:
            mem.teardown()

    def test_parallel_invoke_error_cleanup_dedup_close_and_concurrent_teardown(self) -> None:
        both_inside = threading.Barrier(2, timeout=3.0)

        def parallel_infer(payload: Any, _ctx: dict[str, Any]) -> Any:
            if payload.get("barrier"):
                both_inside.wait()
            return {"val": payload["x"] * 2}

        with AuxiliaryModelService() as svc:
            svc.register_stage("ref", kind=AuxStageKind.REFERENCE_MODEL, engine="miles", model_id="ref-7b", infer_fn=parallel_infer)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(svc.invoke, "ref", f"w-{i}", {"x": i, "barrier": True}) for i in range(2)]
                self.assertEqual([f.result(timeout=5.0).output for f in futs], [{"val": 0}, {"val": 2}])
                many = [pool.submit(svc.invoke, "ref", f"w-{i}", {"x": i % 3}) for i in range(24)]
                results = [f.result(timeout=5.0) for f in many]
                for idx, res in enumerate(results):
                    self.assertEqual(res.output, {"val": (idx % 3) * 2})
                self.assertTrue(any(r.cache_hit for r in results))

        mem = MemoryService()
        shared_pipe = _PipeResource()
        pipe_race = _PipeResource()
        try:
            h = mem.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "verl", "rm", gpus=(0,), size_bytes=1024, handle_id="aux-err")

            # Duplicate close_fn across two stages is deduplicated on teardown.
            with AuxiliaryModelService(memory_service=mem) as svc_dup:
                svc_dup.register_stage("s1", kind="vae", engine="fastvideo", model_id="v", infer_fn=lambda p, c: p, close_fn=shared_pipe.close)
                svc_dup.register_stage("s2", kind="vae", engine="fastvideo", model_id="v", infer_fn=lambda p, c: p, close_fn=shared_pipe.close)
            self.assertTrue(shared_pipe.closed)
            self.assertEqual(shared_pipe.close_calls, 1)

            # Error inside infer_fn releases RO lease (even with hold_lease=True).
            with AuxiliaryModelService(memory_service=mem) as svc_err:
                svc_err.register_stage(
                    "err-stage", kind="reward_model", engine="verl", model_id="rm",
                    handle_id=h.handle_id, infer_fn=lambda _p, _c: (_ for _ in ()).throw(RuntimeError("boom")),
                )
                with self.assertRaises(AuxInvocationError):
                    svc_err.invoke("err-stage", "w0", "x", hold_lease=True)
                self.assertEqual(mem.get_handle(h.handle_id).refcount, 0)
                self.assertEqual(svc_err.stats()["error_count"], 1)

            # Concurrent teardown while infer_fn succeeds or raises raises AuxClosedError & releases lease.
            def run_teardown_race(should_raise: bool) -> None:
                entered, release = threading.Event(), threading.Event()
                svc_race = AuxiliaryModelService(memory_service=mem)

                def raced_infer(p: Any, _c: dict[str, Any]) -> Any:
                    entered.set()
                    release.wait(timeout=3.0)
                    if should_raise:
                        raise RuntimeError("in-flight fail after close")
                    return p

                svc_race.register_stage(
                    "race", kind="vae", engine="fastvideo", model_id="vae", handle_id=h.handle_id,
                    infer_fn=raced_infer, close_fn=pipe_race.close,
                )
                race_exc: list[BaseException] = []

                def run_race() -> None:
                    try:
                        svc_race.invoke("race", "w0", "x", hold_lease=True)
                    except BaseException as exc:
                        race_exc.append(exc)

                t = threading.Thread(target=run_race)
                t.start()
                self.assertTrue(entered.wait(timeout=2.0))
                td = svc_race.teardown()
                self.assertFalse(td["already_closed"])
                self.assertTrue(pipe_race.closed)
                release.set()
                t.join(timeout=3.0)
                self.assertIsInstance(race_exc[0], AuxClosedError)
                self.assertEqual(svc_race.stats()["error_count"], 0)
                self.assertEqual(mem.get_handle(h.handle_id).refcount, 0)

            run_teardown_race(False)
            run_teardown_race(True)
            gc.collect()
        finally:
            shared_pipe.close()
            pipe_race.close()
            mem.teardown()
            time.sleep(0)


if __name__ == "__main__":
    unittest.main()
