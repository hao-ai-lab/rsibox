"""Unit, fail-closed integrity, and multi-process concurrency tests for `rsibox.services.cache`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from rsibox.services import (  # noqa: E402
    CacheArtifactKind,
    CacheClosedError,
    CacheCorruptError,
    CacheEntry,
    CompileCacheError,
    CompileCacheService,
    EnvFingerprint,
    FingerprintMismatchError,
)


def _fp(**overrides: object) -> EnvFingerprint:
    base: dict[str, object] = {
        "gpu_arch": "sm_100a",
        "cuda_version": "12.8",
        "triton_version": "3.2.0",
        "torch_version": "2.6.0",
        "driver_version": "570.124",
        "engine": "vllm",
        "extra": (("smem", "228KB"), ("warp_size", "32")),
    }
    base.update(overrides)
    return EnvFingerprint(**base)  # type: ignore[arg-type]


class EnvFingerprintTests(unittest.TestCase):
    def test_roundtrip_and_canonical_digest(self) -> None:
        fp_a = _fp(extra=(("z_opt", "1"), ("a_opt", "2")))
        fp_b = _fp(extra=(("a_opt", "2"), ("z_opt", "1")))
        self.assertEqual(fp_a.digest(), fp_b.digest())
        self.assertEqual(len(fp_a.digest()), 64)

        as_dict = fp_a.to_dict()
        fp_roundtrip = EnvFingerprint.from_dict(as_dict)
        self.assertEqual(fp_roundtrip, fp_a)
        self.assertEqual(fp_roundtrip.digest(), fp_a.digest())

        as_dict_map = dict(as_dict)
        as_dict_map["extra"] = {"z_opt": "1", "a_opt": "2"}
        fp_from_map = EnvFingerprint.from_dict(as_dict_map)
        self.assertEqual(fp_from_map, fp_a)
        self.assertEqual(fp_from_map.digest(), fp_a.digest())

        with tempfile.TemporaryDirectory() as tmp:
            with CompileCacheService(tmp, fsync=False) as svc:
                svc.put("vllm", "fp_eq", b"ok", fingerprint=fp_a)
                got = svc.get("vllm", "fp_eq", expected_fingerprint=fp_a)
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got[0].fingerprint, fp_a)

    def test_rejects_invalid_types_and_empty_required_fields(self) -> None:
        for bad_dict in ("not-a-dict", ["gpu_arch"], None, 123):
            with self.assertRaises(TypeError):
                EnvFingerprint.from_dict(bad_dict)

        with self.assertRaises(ValueError):
            EnvFingerprint.from_dict({"gpu_arch": "sm_90"})
        with self.assertRaises(ValueError):
            EnvFingerprint.from_dict({"cuda_version": "12.8"})

        for bad_val in ("", "   ", "sm\x0090"):
            with self.assertRaises(ValueError):
                _fp(gpu_arch=bad_val)
            with self.assertRaises(ValueError):
                _fp(cuda_version=bad_val)
            with self.assertRaises(ValueError):
                _fp(engine=bad_val)

        with self.assertRaises(TypeError):
            EnvFingerprint.from_dict({"gpu_arch": "sm_90", "cuda_version": 12.8})
        with self.assertRaises(TypeError):
            EnvFingerprint.from_dict({"gpu_arch": "sm_90", "cuda_version": "12.8", "triton_version": None})
        with self.assertRaises(ValueError):
            _fp(torch_version="2.6\x00")
        with self.assertRaises(TypeError):
            EnvFingerprint.from_dict({"gpu_arch": "sm_90", "cuda_version": "12.8", "extra": "bad"})
        with self.assertRaises(ValueError):
            EnvFingerprint.from_dict({"gpu_arch": "sm_90", "cuda_version": "12.8", "extra": [["only_key"]]})

        for bad_bool_kw in (
            {"gpu_arch": True}, {"cuda_version": False}, {"engine": True},
            {"triton_version": True}, {"extra": True}, {"extra": [(True, "1")]}, {"extra": [("k", False)]},
        ):
            with self.assertRaises(TypeError):
                _fp(**bad_bool_kw)


class CompileCacheServiceTests(unittest.TestCase):
    def test_put_get_list_stats_and_overwrite_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                self.assertFalse(svc.closed)
                self.assertIsNone(svc.get("vllm", "missing", expected_fingerprint=fp))

                meta = {"block_m": 128, "stages": 4}
                payload = b"\x7fELF-triton-cubin-payload"
                entry = svc.put(
                    "vllm",
                    "fused_moe_sm100",
                    payload,
                    fingerprint=fp,
                    kind=CacheArtifactKind.TRITON_KERNEL,
                    metadata=meta,
                )
                meta["block_m"] = 999  # caller mutation must not leak
                self.assertIsInstance(entry, CacheEntry)
                self.assertEqual(entry.kind, CacheArtifactKind.TRITON_KERNEL)
                self.assertEqual(entry.size_bytes, len(payload))
                self.assertEqual(entry.sha256, hashlib.sha256(payload).hexdigest())
                self.assertTrue(entry.payload_path.is_file())
                self.assertEqual(entry.metadata["block_m"], 128)

                got = svc.get("vllm", "fused_moe_sm100", expected_fingerprint=fp)
                self.assertIsNotNone(got)
                assert got is not None
                got_entry, got_bytes = got
                self.assertEqual(got_bytes, payload)
                self.assertEqual(got_entry.fingerprint_digest, fp.digest())
                self.assertEqual(got_entry.metadata, {"block_m": 128, "stages": 4})

                with self.assertRaises(CompileCacheError):
                    svc.put("vllm", "fused_moe_sm100", b"other", fingerprint=fp, overwrite=False)
                # Original payload must remain intact after failed overwrite=False.
                self.assertEqual(svc.get("vllm", "fused_moe_sm100", expected_fingerprint=fp)[1], payload)  # type: ignore[index]

                svc.put("sglang", "graph_bs8", b"cuda-graph-plan", fingerprint=fp, kind=CacheArtifactKind.CUDA_GRAPH_PLAN)
                self.assertEqual(len(svc.list_entries()), 2)
                self.assertEqual([e.key for e in svc.list_entries("vllm")], ["fused_moe_sm100"])

                st = svc.stats()
                self.assertEqual(st["entries"], 2)
                self.assertEqual(st["puts"], 2)
                self.assertEqual(st["hits"], 2)
                self.assertEqual(st["misses"], 1)
                self.assertFalse(st["closed"])

            self.assertTrue(svc.closed)
            self.assertTrue(svc.stats()["closed"])
            self.assertTrue(svc.teardown()["already_closed"])
            with self.assertRaises(CacheClosedError):
                svc.get("vllm", "fused_moe_sm100", expected_fingerprint=fp)
            with self.assertRaises(CacheClosedError):
                svc.put("vllm", "k", b"1", fingerprint=fp)
            with self.assertRaises(CacheClosedError):
                svc.invalidate("vllm")
            with self.assertRaises(CacheClosedError):
                svc.list_entries()
            with self.assertRaises(CacheClosedError):
                with svc:
                    pass

    def test_strict_fingerprint_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp_orig = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                svc.put("triton", "k1", b"cubin-v1", fingerprint=fp_orig, kind=CacheArtifactKind.TRITON_KERNEL)
                drifted_cases = (
                    _fp(gpu_arch="sm_90"),
                    _fp(cuda_version="12.6"),
                    _fp(triton_version="3.1.0"),
                    _fp(torch_version="2.5.1"),
                    _fp(driver_version="565.57"),
                    _fp(engine="sglang"),
                    _fp(extra=(("smem", "228KB"), ("warp_size", "64"))),
                )
                for fp_bad in drifted_cases:
                    with self.assertRaises(FingerprintMismatchError):
                        svc.get("triton", "k1", expected_fingerprint=fp_bad, strict=True)
                    self.assertIsNone(svc.get("triton", "k1", expected_fingerprint=fp_bad, strict=False))
                # Entry remains valid for the matching fingerprint.
                hit = svc.get("triton", "k1", expected_fingerprint=fp_orig, strict=True)
                self.assertIsNotNone(hit)
                assert hit is not None
                self.assertEqual(hit[1], b"cubin-v1")

    def test_corrupt_payload_and_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                # 1. Byte flip in payload -> CacheCorruptError (strict=True), auto-evict + None (strict=False)
                entry = svc.put("inductor", "fx_graph", b"abcdef0123456789", fingerprint=fp, kind=CacheArtifactKind.TORCH_INDUCTOR)
                manifest_path = entry.payload_path.with_suffix(".json")
                raw = bytearray(entry.payload_path.read_bytes())
                raw[0] ^= 0xFF
                entry.payload_path.write_bytes(bytes(raw))

                with self.assertRaises(CacheCorruptError):
                    svc.get("inductor", "fx_graph", expected_fingerprint=fp, strict=True)
                self.assertTrue(entry.payload_path.exists())
                self.assertIsNone(svc.get("inductor", "fx_graph", expected_fingerprint=fp, strict=False))
                self.assertFalse(entry.payload_path.exists())
                self.assertFalse(manifest_path.exists())

                # 2. Truncated payload
                e2 = svc.put("inductor", "fx_trunc", b"0123456789", fingerprint=fp)
                e2.payload_path.write_bytes(b"012")
                with self.assertRaises(CacheCorruptError):
                    svc.get("inductor", "fx_trunc", expected_fingerprint=fp, strict=True)
                self.assertIsNone(svc.get("inductor", "fx_trunc", expected_fingerprint=fp, strict=False))

                # 3. Corrupt sidecar JSON
                e3 = svc.put("inductor", "fx_json", b"valid-bytes", fingerprint=fp)
                e3.payload_path.with_suffix(".json").write_text("{corrupt json", encoding="utf-8")
                with self.assertRaises(CacheCorruptError):
                    svc.get("inductor", "fx_json", expected_fingerprint=fp, strict=True)
                self.assertIsNone(svc.get("inductor", "fx_json", expected_fingerprint=fp, strict=False))
                self.assertFalse(e3.payload_path.exists())

                # 4. Orphan payload (missing manifest)
                e4 = svc.put("inductor", "fx_orphan", b"orphan-bytes", fingerprint=fp)
                e4.payload_path.with_suffix(".json").unlink()
                with self.assertRaises(CacheCorruptError):
                    svc.get("inductor", "fx_orphan", expected_fingerprint=fp, strict=True)
                self.assertIsNone(svc.get("inductor", "fx_orphan", expected_fingerprint=fp, strict=False))
                self.assertFalse(e4.payload_path.exists())

                # 5. Bool schema, non-finite created_at (NaN/Infinity), and non-hex sha256
                for idx, mutator in enumerate((
                    lambda raw_doc: raw_doc.replace('"schema":1', '"schema":true'),
                    lambda raw_doc: raw_doc.replace(f'"created_at":{json.loads(raw_doc)["created_at"]}', '"created_at":NaN'),
                    lambda raw_doc: raw_doc.replace(f'"created_at":{json.loads(raw_doc)["created_at"]}', '"created_at":Infinity'),
                    lambda raw_doc: raw_doc.replace(f'"sha256":"{json.loads(raw_doc)["sha256"]}"', f'"sha256":"{"Z" * 64}"'),
                    lambda raw_doc: raw_doc.replace(f'"sha256":"{json.loads(raw_doc)["sha256"]}"', f'"sha256":"{"A" * 64}"'),
                )):
                    e5 = svc.put("inductor", f"fx_hdr_{idx}", b"hdr-bytes", fingerprint=fp)
                    mp = e5.payload_path.with_suffix(".json")
                    mp.write_text(mutator(mp.read_text(encoding="utf-8")), encoding="utf-8")
                    with self.assertRaises(CacheCorruptError):
                        svc.get("inductor", f"fx_hdr_{idx}", expected_fingerprint=fp, strict=True)
                    self.assertIsNone(svc.get("inductor", f"fx_hdr_{idx}", expected_fingerprint=fp, strict=False))
                    self.assertFalse(e5.payload_path.exists())
                    self.assertFalse(mp.exists())

    def test_rejects_traversal_bad_keys_and_bad_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                bad_segments = ("", "   ", ".", "..", "../x", "a/b", "a\\b", "bad\x00key")
                for bad in bad_segments:
                    with self.assertRaises(ValueError):
                        svc.put(bad, "ok", b"1", fingerprint=fp)
                    with self.assertRaises(ValueError):
                        svc.put("ok", bad, b"1", fingerprint=fp)
                    with self.assertRaises(ValueError):
                        svc.get(bad, "ok", expected_fingerprint=fp)
                    with self.assertRaises(ValueError):
                        svc.get("ok", bad, expected_fingerprint=fp)
                    with self.assertRaises(ValueError):
                        svc.invalidate(bad)
                    with self.assertRaises(ValueError):
                        svc.invalidate("ok", bad)
                    with self.assertRaises(ValueError):
                        svc.list_entries(bad)

                with self.assertRaises(TypeError):
                    svc.put("ns", "k", "not-bytes", fingerprint=fp)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    svc.put("ns", "k", b"1", fingerprint="bad-fp")  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    svc.put("ns", "k", b"1", fingerprint=fp, kind="unknown_kind")
                with self.assertRaises(ValueError):
                    svc.put("ns", "k", b"1", fingerprint=fp, metadata={"bad": float("nan")})

            for bad_dir in ("", "   ", "bad\x00dir"):
                with self.assertRaises(ValueError):
                    CompileCacheService(bad_dir)
            for bad_dir_type in (123, True, None):
                with self.assertRaises(TypeError):
                    CompileCacheService(bad_dir_type)  # type: ignore[arg-type]

    def test_invalidate_key_and_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                svc.put("ns_a", "k1", b"v1", fingerprint=fp)
                svc.put("ns_a", "k2", b"v2", fingerprint=fp)
                svc.put("ns_b", "k1", b"v3", fingerprint=fp)

                self.assertEqual(svc.invalidate("ns_a", "k1"), 1)
                self.assertEqual(svc.invalidate("ns_a", "k1"), 0)
                self.assertIsNone(svc.get("ns_a", "k1", expected_fingerprint=fp))
                self.assertIsNotNone(svc.get("ns_a", "k2", expected_fingerprint=fp))

                self.assertEqual(svc.invalidate("ns_a"), 1)
                self.assertEqual(svc.invalidate("ns_a"), 0)
                self.assertEqual(len(svc.list_entries("ns_a")), 0)
                self.assertEqual(len(svc.list_entries("ns_b")), 1)

    def test_cross_container_instances_and_multithread_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as container_a, \
                 CompileCacheService(tmp, fsync=False) as container_b:
                def worker(idx: int) -> None:
                    svc = container_a if idx % 2 == 0 else container_b
                    key = f"kernel_{idx % 4}"
                    payload = f"worker-{idx}-".encode("utf-8") * 512
                    svc.put("flashinfer", key, payload, fingerprint=fp, kind=CacheArtifactKind.FLASHINFER)
                    got = svc.get("flashinfer", key, expected_fingerprint=fp, strict=True)
                    self.assertIsNotNone(got)
                    assert got is not None
                    entry, data = got
                    self.assertEqual(entry.sha256, hashlib.sha256(data).hexdigest())
                    self.assertEqual(len(data), entry.size_bytes)

                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(worker, i) for i in range(40)]
                    for fut in futures:
                        fut.result()

                self.assertEqual(len(container_b.list_entries("flashinfer")), 4)

    def test_slug_collision_resistance_overwrite_race_and_eviction_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            with CompileCacheService(tmp, fsync=False) as svc:
                # Distinct tokens that sanitize to similar slugs must never collide
                tokens = ("a_b", "a-b", "a:b", "Foo", "foo")
                for tok in tokens:
                    svc.put(tok, tok, f"payload-{tok}".encode("utf-8"), fingerprint=fp)
                self.assertEqual(len(svc.list_entries()), len(tokens))
                for tok in tokens:
                    got = svc.get(tok, tok, expected_fingerprint=fp, strict=True)
                    self.assertIsNotNone(got)
                    assert got is not None
                    self.assertEqual(got[1], f"payload-{tok}".encode("utf-8"))

                # Concurrent overwrite=False race: exactly 1 writer wins, all others get CompileCacheError
                def race_put(i: int) -> bool:
                    try:
                        svc.put("race_ns", "once", f"winner-{i}".encode("utf-8"), fingerprint=fp, overwrite=False)
                        return True
                    except CompileCacheError:
                        return False

                with ThreadPoolExecutor(max_workers=8) as pool:
                    outcomes = list(pool.map(race_put, range(16)))
                self.assertEqual(outcomes.count(True), 1)
                self.assertEqual(outcomes.count(False), 15)

                # Concurrent strict=False corruption eviction race
                corrupt_entry = svc.put("race_ns", "corrupt_me", b"good-bytes", fingerprint=fp)
                corrupt_entry.payload_path.write_bytes(b"X" * len(b"good-bytes"))

                def race_evict(_: int) -> bytes | None:
                    res = svc.get("race_ns", "corrupt_me", expected_fingerprint=fp, strict=False)
                    return None if res is None else res[1]

                with ThreadPoolExecutor(max_workers=8) as pool:
                    evict_results = list(pool.map(race_evict, range(16)))
                self.assertTrue(all(r is None for r in evict_results))
                self.assertFalse(corrupt_entry.payload_path.exists())
                self.assertFalse(corrupt_entry.payload_path.with_suffix(".json").exists())

    def test_multiprocess_fork_atomic_put_get_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = _fp()
            n_procs = 4
            n_iters = 25
            pids: list[int] = []
            for proc_id in range(n_procs):
                pid = os.fork()
                if pid == 0:
                    exit_code = 0
                    try:
                        with CompileCacheService(tmp, fsync=False) as svc:
                            for it in range(n_iters):
                                tag = f"p{proc_id}-i{it}:".encode("ascii")
                                payload = tag * 1024
                                svc.put("shared_ns", "hot_kernel", payload, fingerprint=fp)
                                res = svc.get("shared_ns", "hot_kernel", expected_fingerprint=fp, strict=True)
                                if res is None:
                                    raise AssertionError("unexpected cache miss after put")
                                entry, data = res
                                if hashlib.sha256(data).hexdigest() != entry.sha256:
                                    raise AssertionError("torn read detected")
                                if len(data) != entry.size_bytes or len(data) == 0:
                                    raise AssertionError("size mismatch detected")
                                # Verify payload came entirely from a single writer tag (no interleaved bytes)
                                prefix = data.split(b":", 1)[0] + b":"
                                if data != prefix * 1024:
                                    raise AssertionError("interleaved payload bytes")
                    except BaseException:
                        exit_code = 1
                    finally:
                        os._exit(exit_code)
                pids.append(pid)

            for pid in pids:
                _, status = os.waitpid(pid, 0)
                self.assertTrue(os.WIFEXITED(status), f"child {pid} did not exit normally: {status}")
                self.assertEqual(os.WEXITSTATUS(status), 0, f"child {pid} failed")

            with CompileCacheService(tmp, fsync=False) as verifier:
                got = verifier.get("shared_ns", "hot_kernel", expected_fingerprint=fp, strict=True)
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got[0].sha256, hashlib.sha256(got[1]).hexdigest())
            gc.collect()


if __name__ == "__main__":
    unittest.main()
