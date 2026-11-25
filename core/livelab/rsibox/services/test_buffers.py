"""Unit, fail-closed integrity, ResourceWarning, and concurrency tests for `rsibox.services.buffers`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
import warnings

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from rsibox.services import (  # noqa: E402
    BufferClosedError,
    BufferResetReceipt,
    BufferServiceError,
    CursorOutOfRangeError,
    EmptyBufferError,
    IndexCorruptError,
    IndexedPrefetchService,
    PrefetchItem,
    StaleVersionError,
    VersionedEntry,
    VersionedReplayBufferService,
)


class IndexedPrefetchServiceTests(unittest.TestCase):
    def test_build_open_get_seek_batch_and_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "dataset.jsonl"
            index_path = root / "dataset.idx"
            ckpt_path = root / "cursor.json"
            records = [b'{"id":0}', b'{"id":1,"text":"hello"}', b"", b'{"id":3}']

            count = IndexedPrefetchService.build_index(data_path, index_path, records=records)
            self.assertEqual(count, 4)

            with IndexedPrefetchService.open(data_path, index_path) as svc:
                self.assertFalse(svc.closed)
                self.assertEqual(len(svc), 4)
                self.assertEqual(svc.cursor(), 0)

                item2 = svc.get(2)
                self.assertIsInstance(item2, PrefetchItem)
                self.assertEqual(item2.index, 2)
                self.assertEqual(item2.length, 0)
                self.assertEqual(item2.payload, b"")

                self.assertEqual(svc.get(1).payload, records[1])
                self.assertEqual(svc.seek(1), 1)
                batch = svc.next_batch(2)
                self.assertEqual([it.payload for it in batch], [records[1], records[2]])
                self.assertEqual(svc.cursor(), 3)

                ckpt = svc.checkpoint_cursor(ckpt_path)
                self.assertEqual(ckpt["cursor"], 3)
                self.assertEqual(ckpt["count"], 4)

                tail = svc.next_batch(5)
                self.assertEqual([it.payload for it in tail], [records[3]])
                self.assertEqual(svc.cursor(), 4)
                self.assertEqual(svc.next_batch(2), [])

            self.assertTrue(svc.closed)
            svc.close()  # idempotent

            with IndexedPrefetchService.open(data_path, index_path) as reopened:
                self.assertEqual(reopened.restore_cursor(ckpt_path), 3)
                self.assertEqual(reopened.cursor(), 3)
                self.assertEqual([it.payload for it in reopened.next_batch(2)], [records[3]])

    def test_build_index_existing_file_and_chunk_boundary_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "existing.bin"
            index_path = root / "existing.idx"
            # Multi-byte delimiter spanning 64 KiB scan chunk boundary (65536).
            delim = b"<DELIM>"
            rec0 = b"A" * 65533  # delimiter starts at 65533, spans 65536
            rec1 = b"B" * 128
            rec2 = b"C" * 17
            data_path.write_bytes(rec0 + delim + rec1 + delim + rec2)  # no trailing delim

            count = IndexedPrefetchService.build_index(data_path, index_path, delimiter=delim)
            self.assertEqual(count, 3)

            with IndexedPrefetchService(data_path, index_path, start_cursor=1) as svc:
                self.assertEqual(svc.get(0).payload, rec0)
                self.assertEqual(svc.get(1).payload, rec1)
                self.assertEqual(svc.get(2).payload, rec2)
                self.assertEqual([it.payload for it in svc.next_batch(5)], [rec1, rec2])

    def test_empty_dataset_supported_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "empty.dat"
            index_path = root / "empty.idx"
            self.assertEqual(IndexedPrefetchService.build_index(data_path, index_path, records=[]), 0)
            self.assertEqual(data_path.stat().st_size, 0)

            with IndexedPrefetchService.open(data_path, index_path) as svc:
                self.assertEqual(len(svc), 0)
                self.assertEqual(svc.cursor(), 0)
                self.assertEqual(svc.seek(0), 0)
                self.assertEqual(svc.next_batch(4), [])
                with self.assertRaises(CursorOutOfRangeError):
                    svc.get(0)

            # Also verify scanning an existing 0-byte file
            idx2 = root / "empty2.idx"
            self.assertEqual(IndexedPrefetchService.build_index(data_path, idx2), 0)
            with IndexedPrefetchService.open(data_path, idx2) as svc2:
                self.assertEqual(len(svc2), 0)

    def test_corrupt_index_truncated_data_and_bad_checkpoint_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.bin"
            index_path = root / "data.idx"
            IndexedPrefetchService.build_index(data_path, index_path, records=[b"alpha", b"beta"])
            orig_idx = index_path.read_bytes()
            orig_data = data_path.read_bytes()

            # Short header
            index_path.write_bytes(orig_idx[:8])
            with self.assertRaises(IndexCorruptError):
                IndexedPrefetchService(data_path, index_path)

            # Bad magic
            index_path.write_bytes(b"BADMAGIC" + orig_idx[8:])
            with self.assertRaises(IndexCorruptError):
                IndexedPrefetchService(data_path, index_path)

            # Truncated index body / count mismatch
            index_path.write_bytes(orig_idx[:-4])
            with self.assertRaises(IndexCorruptError):
                IndexedPrefetchService(data_path, index_path)

            # Out-of-bounds offset in index entry
            bad_entry = struct.pack("<QQ8s", 999, 5, b"\x00" * 8)
            index_path.write_bytes(orig_idx[:24] + bad_entry + orig_idx[48:])
            with self.assertRaises(IndexCorruptError):
                IndexedPrefetchService(data_path, index_path)

            # Truncated data file
            index_path.write_bytes(orig_idx)
            data_path.write_bytes(orig_data[:-2])
            with self.assertRaises(IndexCorruptError):
                IndexedPrefetchService(data_path, index_path)

            # In-place data corruption caught by sha256 prefix check on get()
            data_path.write_bytes(b"X" + orig_data[1:])
            with IndexedPrefetchService(data_path, index_path) as svc:
                with self.assertRaises(IndexCorruptError):
                    svc.get(0)

            # Checkpoint corruption / index mismatch
            data_path.write_bytes(orig_data)
            ckpt_path = root / "ckpt.json"
            with IndexedPrefetchService(data_path, index_path) as svc:
                svc.seek(1)
                svc.checkpoint_cursor(ckpt_path)
                ckpt_doc = json.loads(ckpt_path.read_text(encoding="utf-8"))

                ckpt_path.write_text("{corrupt", encoding="utf-8")
                with self.assertRaises(IndexCorruptError):
                    svc.restore_cursor(ckpt_path)

                for bad_patch in (
                    {"schema": 99},
                    {"cursor": True},
                    {"cursor": "1"},
                    {"index_sha256": "0" * 64},
                    {"count": 99},
                ):
                    patched = {**ckpt_doc, **bad_patch}
                    ckpt_path.write_text(json.dumps(patched), encoding="utf-8")
                    with self.assertRaises(IndexCorruptError):
                        svc.restore_cursor(ckpt_path)

                ckpt_path.write_text(json.dumps({**ckpt_doc, "cursor": 99}), encoding="utf-8")
                with self.assertRaises(CursorOutOfRangeError):
                    svc.restore_cursor(ckpt_path)

    def test_rejects_bool_oob_and_post_close_with_zero_resource_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "d.bin"
            index_path = root / "d.idx"
            IndexedPrefetchService.build_index(data_path, index_path, records=[b"x", b"y"])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                with self.assertRaises(TypeError):
                    IndexedPrefetchService(data_path, index_path, start_cursor=True)  # type: ignore[arg-type]
                with self.assertRaises(CursorOutOfRangeError):
                    IndexedPrefetchService(data_path, index_path, start_cursor=-1)
                with self.assertRaises(CursorOutOfRangeError):
                    IndexedPrefetchService(data_path, index_path, start_cursor=3)

                # Trigger IndexCorruptError during __init__ after mmap is opened
                raw_idx = index_path.read_bytes()
                index_path.write_bytes(b"BADMAGIC" + raw_idx[8:])
                with self.assertRaises(IndexCorruptError):
                    IndexedPrefetchService(data_path, index_path)
                gc.collect()
            self.assertEqual([w for w in caught if issubclass(w.category, ResourceWarning)], [])

            index_path.write_bytes(raw_idx)
            with warnings.catch_warnings(record=True) as caught_normal:
                warnings.simplefilter("always", ResourceWarning)
                svc = IndexedPrefetchService(data_path, index_path)
                for bad_idx in (True, False, 1.5, "0"):
                    with self.assertRaises(TypeError):
                        svc.get(bad_idx)  # type: ignore[arg-type]
                    with self.assertRaises(TypeError):
                        svc.seek(bad_idx)  # type: ignore[arg-type]
                    with self.assertRaises(TypeError):
                        svc.next_batch(bad_idx)  # type: ignore[arg-type]
                for oob in (-1, 2):
                    with self.assertRaises(CursorOutOfRangeError):
                        svc.get(oob)
                with self.assertRaises(CursorOutOfRangeError):
                    svc.seek(-1)
                with self.assertRaises(CursorOutOfRangeError):
                    svc.seek(3)
                with self.assertRaises(ValueError):
                    svc.next_batch(0)
                svc.close()
                gc.collect()
            self.assertEqual([w for w in caught_normal if issubclass(w.category, ResourceWarning)], [])

            for fn in (
                lambda: len(svc),
                svc.cursor,
                lambda: svc.get(0),
                lambda: svc.seek(0),
                lambda: svc.next_batch(1),
                lambda: svc.checkpoint_cursor(root / "c.json"),
                lambda: svc.restore_cursor(root / "c.json"),
                svc.__enter__,
            ):
                with self.assertRaises(BufferClosedError):
                    fn()

    def test_concurrent_get_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "multi.bin"
            index_path = root / "multi.idx"
            records = [f"rec-{i}".encode("ascii") for i in range(64)]
            IndexedPrefetchService.build_index(data_path, index_path, records=records)

            with IndexedPrefetchService(data_path, index_path) as svc:
                def worker(idx: int) -> bytes:
                    return svc.get(idx % len(records)).payload

                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(worker, range(256)))
                for idx, payload in enumerate(results):
                    self.assertEqual(payload, records[idx % len(records)])


class VersionedReplayBufferServiceTests(unittest.TestCase):
    def test_push_eviction_sample_and_version_isolation(self) -> None:
        buf = VersionedReplayBufferService(capacity=3, current_code_version="v1")
        self.assertEqual(buf.current_code_version, "v1")
        self.assertEqual(len(buf), 0)

        with self.assertRaises(EmptyBufferError):
            buf.sample(1)

        meta = {"reward": 1.0}
        e1 = buf.push("r1", metadata=meta)
        meta["reward"] = 999.0  # caller mutation must not affect stored entry
        self.assertIsInstance(e1, VersionedEntry)
        self.assertEqual(e1.metadata, {"reward": 1.0})

        buf.push("r2")
        buf.push("r3")
        buf.push("r4")  # evicts r1
        self.assertEqual(len(buf), 3)

        rng = random.Random(42)
        sampled = buf.sample(10, rng=rng)  # exact=False returns all 3 eligible
        self.assertEqual({e.payload for e in sampled}, {"r2", "r3", "r4"})
        with self.assertRaises(EmptyBufferError):
            buf.sample(4, exact=True)

        # Advance with drain_stale=False so v1 entries stay in buffer
        receipt_keep = buf.advance_code_version("v2", drain_stale=False)
        self.assertEqual(receipt_keep, BufferResetReceipt(drained=0, retained=3, active_code_version="v2", epoch=1))
        with self.assertRaises(EmptyBufferError):
            buf.sample(1)  # default filters strictly to active v2

        stale = buf.sample(2, allow_stale=True, rng=random.Random(7))
        self.assertEqual(len(stale), 2)
        self.assertTrue(all(e.code_version == "v1" for e in stale))
        with self.assertRaises(StaleVersionError):
            buf.sample(1, code_version="v1", allow_stale=False)

        buf.push("r5", code_version="v2")
        v2_only = buf.sample(5)
        self.assertEqual([e.payload for e in v2_only], ["r5"])
        v1_only = buf.sample(5, code_version="v1", allow_stale=True)
        self.assertEqual({e.payload for e in v1_only}, {"r3", "r4"})

        # Advance with drain_stale=True evicts stale rollouts
        receipt_drain = buf.advance_code_version("v3", drain_stale=True)
        self.assertEqual(receipt_drain, BufferResetReceipt(drained=3, retained=0, active_code_version="v3", epoch=2))
        self.assertEqual(len(buf), 0)

        buf.push("r6")
        reset_rcpt = buf.reset_state()
        self.assertEqual(reset_rcpt, BufferResetReceipt(drained=1, retained=0, active_code_version="v3", epoch=3))
        stats = buf.stats()
        self.assertEqual(stats["evicted"], 2)
        self.assertEqual(stats["drained"], 4)
        self.assertEqual(stats["total_pushed"], 6)
        buf.close()

    def test_reject_stale_push_drain_on_mismatch_and_validation(self) -> None:
        strict_buf = VersionedReplayBufferService(capacity=4, current_code_version="v1", reject_stale_push=True)
        strict_buf.push("ok")
        with self.assertRaises(StaleVersionError):
            strict_buf.push("stale", code_version="v0")
        strict_buf.close()

        drain_buf = VersionedReplayBufferService(capacity=4, current_code_version="v1", drain_on_version_mismatch=True)
        drain_buf.advance_code_version("v2", drain_stale=False)
        drain_buf.push("old", code_version="v1")
        # Trigger mismatch drain when pushing another mismatched entry while active=v2
        drain_buf.advance_code_version("v3", drain_stale=False)
        drain_buf.push("old2", code_version="v1")
        self.assertEqual(drain_buf.stats()["drained"], 1)
        drain_buf.close()

        # Validation & bool rejections
        for bad_cap in (True, False, "4", 1.5):
            with self.assertRaises(TypeError):
                VersionedReplayBufferService(bad_cap)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VersionedReplayBufferService(0)
        for bad_ver in (True, 123, None):
            with self.assertRaises(TypeError):
                VersionedReplayBufferService(4, current_code_version=bad_ver)  # type: ignore[arg-type]
        for bad_ver_str in ("", "  ", "v\x001"):
            with self.assertRaises(ValueError):
                VersionedReplayBufferService(4, current_code_version=bad_ver_str)

        with VersionedReplayBufferService(4) as buf:
            buf.push("item")
            for bad_bool in (True, False, "1"):
                with self.assertRaises(TypeError):
                    buf.sample(bad_bool)  # type: ignore[arg-type]
            for bad_flag in (1, "yes"):
                with self.assertRaises(TypeError):
                    buf.sample(1, allow_stale=bad_flag)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    buf.sample(1, exact=bad_flag)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    buf.advance_code_version("v2", drain_stale=bad_flag)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                buf.sample(1, rng=True)  # type: ignore[arg-type]

        self.assertTrue(buf.closed)
        buf.close()
        for fn in (
            lambda: len(buf),
            lambda: buf.current_code_version,
            lambda: buf.push("x"),
            lambda: buf.sample(1),
            lambda: buf.advance_code_version("v2"),
            buf.reset_state,
            buf.__enter__,
        ):
            with self.assertRaises(BufferClosedError):
                fn()

    def test_concurrent_push_sample_and_advance_version(self) -> None:
        with VersionedReplayBufferService(capacity=32, current_code_version="v0") as buf:
            def run_worker(worker_id: int) -> None:
                rng = random.Random(worker_id)
                for step in range(40):
                    if worker_id == 0 and step % 10 == 0:
                        buf.advance_code_version(f"v{step // 10 + 1}", drain_stale=(step % 20 == 0))
                    buf.push({"w": worker_id, "s": step})
                    try:
                        batch = buf.sample(4, allow_stale=(worker_id % 2 == 1), rng=rng)
                        self.assertTrue(1 <= len(batch) <= 4)
                    except EmptyBufferError:
                        pass
                    _ = buf.stats()

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(run_worker, range(6)))

            self.assertIsInstance(buf.stats()["epoch"], int)
            self.assertTrue(issubclass(BufferClosedError, BufferServiceError))


if __name__ == "__main__":
    unittest.main()
