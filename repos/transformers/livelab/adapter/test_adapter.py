"""Comprehensive CPU golden unit tests for the HuggingFace Transformers RSIBox adapter."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[1]
CORE_LIVELAB = Path(__file__).resolve().parents[4] / "core" / "livelab"
for search_path in (str(LIVELAB_DIR), str(CORE_LIVELAB)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from adapter import TransformersQuiesceHooks  # noqa: E402
from adapter.agent import (  # noqa: E402
    _move_tensors_preserving_storage,
    _normalize_device_map,
    _normalize_max_memory,
    _quiesce_runtime,
    _resolve_input_device,
    _resume_runtime,
    _stringify_device_map,
)
from adapter.fixture.patches import real_patches  # noqa: E402
from adapter.fixture.toy_hf.modeling import ToyCausalLM  # noqa: E402
from adapter.lab import TransformersLab, _env_config  # noqa: E402
from adapter.toy_lab import ToyTransformersLab  # noqa: E402
from conformance.cases import REFUSALS  # noqa: E402
from kernel import coordinator as kernel_coordinator  # noqa: E402
from kernel import findings as kernel_findings  # noqa: E402
from kernel import locate as kernel_locate  # noqa: E402
from rsibox import harness as rsibox_harness  # noqa: E402
from rsibox.warm import (  # noqa: E402
    DefaultRespawnDriver,
    DiskStore,
    DynamoK8sSnapshotDriver,
    QuiesceHooks,
    QuiesceReceipt,
    StoreProbe,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class LazyImportAndCliTests(unittest.TestCase):
    def test_adapter_imports_do_not_eagerly_load_torch_or_transformers(self) -> None:
        probe_code = (
            "import importlib, json, sys\n"
            "forbidden = {'torch', 'transformers', 'cuda', 'cupy', 'triton', 'pynvml', 'nvidia'}\n"
            "before = set(sys.modules)\n"
            "for mod_name in ('adapter', 'adapter.lab', 'adapter.toy_lab', 'adapter.snapshot_hooks'):\n"
            "    importlib.import_module(mod_name)\n"
            "loaded = sorted({name.split('.')[0] for name in (set(sys.modules) - before) "
            "if name.split('.')[0] in forbidden})\n"
            "print(json.dumps({'loaded': loaded}))\n"
        )
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(LIVELAB_DIR), str(CORE_LIVELAB)]),
            "CUDA_VISIBLE_DEVICES": "",
        }
        proc = subprocess.run([sys.executable, "-c", probe_code], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["loaded"], [])

    def test_launch_environment_cli_contract_parses_flags_help_and_rejects_typos(self) -> None:
        env = TransformersLab.launch_environment(
            [
                "--", "--model", "Qwen/Qwen3-0.6B", "--device", "cuda:0",
                "--dtype", "bfloat16", "--max-new-tokens", "32", "--quiesce-timeout", "240",
            ]
        )
        self.assertEqual(
            env,
            {
                "RSIBOX_TRANSFORMERS_MODEL": "Qwen/Qwen3-0.6B",
                "RSIBOX_TRANSFORMERS_DEVICE": "cuda:0",
                "RSIBOX_TRANSFORMERS_DTYPE": "bfloat16",
                "RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS": "32",
                "RSIBOX_TRANSFORMERS_QUIESCE_TIMEOUT": "240.0",
            },
        )
        help_buffer = io.StringIO()
        with contextlib.redirect_stdout(help_buffer):
            with self.assertRaises(SystemExit) as help_exit:
                TransformersLab.launch_environment(["--", "--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertIn("--model", help_buffer.getvalue())

        error_buffer = io.StringIO()
        with contextlib.redirect_stderr(error_buffer):
            with self.assertRaises(SystemExit) as typo_exit:
                TransformersLab.launch_environment(["--model", "Qwen/Qwen3-0.6B", "--unknown-flag"])
        self.assertEqual(typo_exit.exception.code, 2)

    def test_device_map_and_max_memory_cli_env_and_validation(self) -> None:
        env = TransformersLab.launch_environment(
            [
                "--model", "Qwen/Qwen3-8B",
                "--device-map", "auto",
                "--max-memory", '{"0": "40GiB", "1": "40GiB"}',
            ]
        )
        self.assertEqual(env["RSIBOX_TRANSFORMERS_DEVICE_MAP"], "auto")
        self.assertEqual(env["RSIBOX_TRANSFORMERS_MAX_MEMORY"], '{"0": "40GiB", "1": "40GiB"}')
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = _env_config()
        self.assertEqual(cfg["device_map"], "auto")
        self.assertEqual(cfg["max_memory"], {"0": "40GiB", "1": "40GiB"})

        self.assertEqual(_normalize_device_map("auto"), "auto")
        self.assertEqual(_normalize_device_map('{"embed": 0, "lm_head": "cuda:1"}'), {"embed": 0, "lm_head": "cuda:1"})
        self.assertEqual(_normalize_max_memory({"0": "40GiB", "cpu": "64GiB"}), {0: "40GiB", "cpu": "64GiB"})
        self.assertEqual(_stringify_device_map({"embed": "0", "head": 1}), {"embed": "cuda:0", "head": "cuda:1"})
        # Whitespace-only env vars do not crash _env_config().
        with mock.patch.dict(
            os.environ,
            {
                "RSIBOX_TRANSFORMERS_MODEL": "Qwen/Qwen3-8B",
                "RSIBOX_TRANSFORMERS_DEVICE_MAP": "   ",
                "RSIBOX_TRANSFORMERS_MAX_MEMORY": "   ",
            },
            clear=False,
        ):
            ws_cfg = _env_config()
        self.assertNotIn("device_map", ws_cfg)
        self.assertNotIn("max_memory", ws_cfg)
        for bad_dm in ({}, {"": 0}, {"embed": True}, {"embed": -1}, "{bad-json", 123):
            with self.assertRaises(ValueError):
                _normalize_device_map(bad_dm)
        for bad_mm in ({}, {True: "10GiB"}, {-1: "10GiB"}, {"0": 0}, {"0": -1}, {"0": True}, {"²": "10GiB"}, "{bad-json", 123):
            with self.assertRaises(ValueError):
                _normalize_max_memory(bad_mm)


class ToyTransformersLiveLabTests(unittest.TestCase):
    def test_live_swap_witness_rollback_and_fresh_parity(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)

        base_status = lab.status()
        self.assertEqual(base_status["state"], "base")
        base_check = lab.check()
        self.assertTrue(base_check["ok"], base_check)
        base_output = base_check["output"]

        fresh_base = lab.fresh_check(None)
        self.assertTrue(fresh_base["ok"], fresh_base)
        self.assertEqual(fresh_base["output"], base_output)

        visible_patch = lab.patches()["visible"]
        install_reply = lab.install(visible_patch)
        self.assertTrue(install_reply["ok"], install_reply)
        self.assertEqual(install_reply["outcome"], "committed")
        self.assertEqual(install_reply["receipt"]["engine"]["version"], "toy-1")

        edited_check = lab.check()
        self.assertTrue(edited_check["ok"], edited_check)
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], base_output)
        worker_witness = edited_check["witness"].get("worker0", {})
        self.assertTrue(worker_witness and min(worker_witness.values()) > 0, edited_check["witness"])

        fresh_edited = lab.fresh_check(visible_patch)
        self.assertTrue(fresh_edited["ok"], fresh_edited)
        self.assertEqual(fresh_edited["output"], edited_check["output"])

        uninstall_reply = lab.uninstall()
        self.assertTrue(uninstall_reply["ok"], uninstall_reply)
        self.assertEqual(uninstall_reply["state"], "base")
        self.assertEqual(lab.status()["fingerprint"], base_status["fingerprint"])
        after_rollback = lab.check()
        self.assertTrue(after_rollback["ok"])
        self.assertEqual(after_rollback["output"], base_output)
        self.assertIsNone(after_rollback["proven"])
        self.assertEqual(after_rollback["witness"], {})

    def test_fault_injection_crash_hang_lost_tamper_and_respawn_recovery(self) -> None:
        lab = ToyTransformersLab(recovery_driver=DefaultRespawnDriver(), weight_store=DiskStore())
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        visible_patch = lab.patches()["visible"]

        for fault_name in ("crash", "hang", "lost", "tamper"):
            with self.subTest(fault=fault_name):
                injected = lab.inject(fault_name)
                self.assertTrue(injected["ok"], injected)
                lab.check()
                poisoned_status = lab.status()
                self.assertEqual(poisoned_status["state"], "poisoned", poisoned_status)
                refused = lab.install(visible_patch)
                self.assertFalse(refused["ok"])
                self.assertEqual(refused["codes"], ["session_poisoned"])
                respawn_reply = lab.reset("respawn")
                self.assertTrue(respawn_reply["ok"], respawn_reply)
                self.assertEqual(respawn_reply["driver"], "default_respawn")
                self.assertEqual(lab.status()["state"], "base")
                self.assertEqual(lab.check()["output"], base_output)

    def test_stale_worker_generation_error_cannot_dirty_freshly_respawned_worker(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        stale_generation = lab._worker_generation
        stale_sock = lab.sock_path
        respawn_reply = lab.reset("respawn")
        self.assertTrue(respawn_reply["ok"], respawn_reply)
        self.assertGreater(lab._worker_generation, stale_generation)
        self.assertFalse(lab.dirty)

        # Simulate a stale call from the prior generation hitting a dead socket.
        original_gen = lab._worker_generation
        lab._worker_generation = stale_generation
        with mock.patch.object(lab, "sock_path", stale_sock):
            # Advance generation right before the socket error unwinds, as respawn does.
            def fail_after_respawn(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                lab._worker_generation = original_gen
                raise TimeoutError("stale worker timed out")

            with mock.patch("adapter.lab.ipc.unix_json_call", side_effect=fail_after_respawn):
                with self.assertRaises(ConnectionError):
                    lab.worker_call("ping")
        self.assertFalse(lab.dirty)
        self.assertTrue(lab.check()["ok"])

    def test_unhealthy_weight_store_fails_closed_at_startup(self) -> None:
        class _BrokenWeightStore:
            name = "broken_store"

            def probe(self, **kwargs: Any) -> StoreProbe:
                return StoreProbe(ok=False, backend="broken_store", error="store offline")

        with self.assertRaisesRegex(RuntimeError, "store offline"):
            ToyTransformersLab(weight_store=_BrokenWeightStore())

    def test_unassisted_sigkill_fast_fails_without_waiting_for_rpc_timeout(self) -> None:
        """Golden case: SIGKILLing the live worker mid-RPC fails in <1 s instead of waiting 30 s."""
        lab = ToyTransformersLab(rpc_timeout=30.0, check_timeout=30.0)
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        self.assertIsNotNone(lab._worker_proc)
        worker_pid = lab._worker_proc.pid

        # 1. Pre-existing SIGKILL fails fast on next worker_call.
        os.kill(worker_pid, signal.SIGKILL)
        lab._worker_proc.wait(timeout=5.0)
        t0 = time.monotonic()
        with self.assertRaises(ConnectionError):
            lab.worker_call("ping", timeout=30.0)
        elapsed_dead = time.monotonic() - t0
        self.assertLess(elapsed_dead, 1.0, f"dead worker call took {elapsed_dead:.3f}s")
        self.assertTrue(lab.dirty)

        # Respawn cleanly, then kill mid-flight during a 25 s hang RPC!
        respawn = lab.reset("respawn")
        self.assertTrue(respawn["ok"], respawn)
        self.assertEqual(lab.check()["output"], base_output)
        self.assertTrue(lab.inject("hang", hang_s=25.0)["ok"])
        inflight_pid = lab._worker_proc.pid

        def kill_after_start() -> None:
            time.sleep(0.15)
            with contextlib.suppress(ProcessLookupError):
                os.kill(inflight_pid, signal.SIGKILL)

        killer = threading.Thread(target=kill_after_start, daemon=True)
        killer.start()
        t1 = time.monotonic()
        with self.assertRaises(ConnectionError):
            lab.run_generate()
        elapsed_inflight = time.monotonic() - t1
        killer.join(timeout=2.0)
        self.assertLess(elapsed_inflight, 2.0, f"mid-flight SIGKILL took {elapsed_inflight:.3f}s (expected <2s vs 30s)")
        self.assertTrue(lab.dirty)

        # Clean respawn restores full parity.
        self.assertTrue(lab.reset("respawn")["ok"])
        self.assertEqual(lab.check()["output"], base_output)

    def test_multigpu_device_map_quiesce_resume_live_swap_and_tp_probe(self) -> None:
        """Golden case: device_map sharding preserves shard map across quiesce/resume and live swaps."""
        probed: dict[str, Any] = {}

        class _RecordingStore:
            name = "recording_store"

            def probe(self, **kwargs: Any) -> StoreProbe:
                probed.update(kwargs)
                return StoreProbe(ok=True, backend="recording_store")

        shard_map = {"embed": "gpu:0", "layers.0": "gpu:0", "layers.1": "gpu:1", "lm_head": "gpu:1"}
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, clear=False):
            lab = ToyTransformersLab(device_map=shard_map, weight_store=_RecordingStore())
        self.addCleanup(lab.close)
        # Subset dict device_map probes ONLY the 2 targeted GPUs ("4", "5"), NOT all 4 visible GPUs!
        self.assertEqual(probed.get("tp"), 2)
        self.assertEqual(probed.get("gpus"), ("4", "5"))
        self.assertEqual(lab.tp, 2)
        self.assertEqual(lab.hf_device_map, shard_map)
        self.assertEqual(lab.devices, ["gpu:0", "gpu:1"])

        # Auto device_map with unset CUDA_VISIBLE_DEVICES passes tp=None so GMS probes all visible GPUs.
        lab_auto = ToyTransformersLab(device_map="auto", auto_spawn=False)
        self.assertEqual(lab_auto._weight_store_probe_kwargs(None), {"tp": None, "visible_devices": None})
        self.assertEqual(lab_auto._weight_store_probe_kwargs("4,5"), {"tp": 2, "visible_devices": "4,5"})
        lab_cpu = ToyTransformersLab(device_map={"embed": "cpu"}, auto_spawn=False)
        self.assertEqual(lab_cpu._weight_store_probe_kwargs("4,5"), {"tp": 1, "visible_devices": "4,5"})
        lab_ws = ToyTransformersLab(device_map="   ", auto_spawn=False)
        self.assertEqual(lab_ws._weight_store_probe_kwargs("4,5"), {"tp": 1, "visible_devices": "4,5"})
        # Empty CUDA_VISIBLE_DEVICES="" or out-of-range GPU index fails closed (never bypasses via raw gpus=("0",)!).
        lab_oob = ToyTransformersLab(device_map={"embed": 0, "lm_head": 2}, auto_spawn=False)
        self.assertEqual(lab_oob._weight_store_probe_kwargs(""), {"tp": 3, "visible_devices": ""})
        self.assertEqual(lab_oob._weight_store_probe_kwargs("4,5"), {"tp": 3, "visible_devices": "4,5"})
        self.assertEqual(lab_oob._weight_store_probe_kwargs(None), {"gpus": ("0", "2"), "tp": 2, "visible_devices": None})

        base_output = lab.check()["output"]
        hooks = lab.quiesce_hooks(offload_to_cpu=True)
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok, receipt)
        # Double quiesce fails closed without overwriting saved shard placement.
        self.assertFalse(lab.quiesce_worker(offload_to_cpu=True)["ok"])
        resume_reply = hooks.resume_after_restore(receipt)
        self.assertTrue(resume_reply["ok"], resume_reply)
        self.assertEqual(resume_reply["hf_device_map"], shard_map)
        self.assertEqual(resume_reply["devices"], ["gpu:0", "gpu:1"])

        # Live edit + fresh_check parity work identically under multi-GPU device_map.
        visible = lab.patches()["visible"]
        self.assertTrue(lab.install(visible)["ok"])
        edited = lab.check()
        self.assertTrue(edited["ok"] and edited["proven"])
        fresh = lab.fresh_check(visible)
        self.assertTrue(fresh["ok"])
        self.assertEqual(fresh["output"], edited["output"])
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual(lab.check()["output"], base_output)


class TransformersQuiesceHooksTests(unittest.TestCase):
    def test_toy_model_raises_when_offloaded_and_worker_blocks_calls_while_quiesced(self) -> None:
        toy_model = ToyCausalLM()
        toy_model.offload_to_cpu()
        with self.assertRaisesRegex(RuntimeError, "offloaded"):
            toy_model.forward_logits([1, 2, 3])
        toy_model.restore_to_device("cpu")
        self.assertEqual(len(toy_model.forward_logits([1, 2, 3])), 64)

        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        hooks = lab.quiesce_hooks(offload_to_cpu=True)
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok, receipt)

        # Worker refuses generation and live-edit prepare/install while quiesced.
        with self.assertRaisesRegex(RuntimeError, "quiesced"):
            lab.run_generate()
        refused_install = lab.install(lab.patches()["visible"])
        self.assertFalse(refused_install["ok"])

        # Resuming without "weights" leaves ToyCausalLM offloaded (verifying selective restore).
        partial_resume = lab.resume_worker(released=("kv_cache",))
        self.assertTrue(partial_resume["ok"], partial_resume)
        self.assertEqual(partial_resume["resumed"], ["kv_cache"])
        with self.assertRaisesRegex(RuntimeError, "offloaded"):
            lab.run_generate()

        # Full resume restores weights and returns exact generation parity.
        full_resume = hooks.resume_after_restore(receipt)
        self.assertTrue(full_resume["ok"], full_resume)
        self.assertEqual(full_resume["resumed"], ["kv_cache", "weights"])
        self.assertEqual(lab.check()["output"], base_output)

    def test_real_tamper_blocks_warmup_and_quiesce_without_writing_ready_sentinel(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks()
        self.assertTrue(lab.inject("tamper")["ok"])

        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            hooks.warmup_for_snapshot()
        receipt = hooks.quiesce_for_snapshot()
        self.assertFalse(receipt.ok)
        self.assertIn("poisoned", receipt.error or "")

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="transformers", control_dir=control_dir, wait_timeout_s=0.5, poll_s=0.01
            )
            lifecycle = driver.checkpoint_and_wait(hooks, skip_warmup=True)
            self.assertFalse(lifecycle.ok)
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertFalse((Path(control_dir) / "transformers-restore-ready").exists())

    def test_failed_quiesce_receipt_is_rejected_immediately_by_resume(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks()
        failed_receipt = QuiesceReceipt(ok=False, error="quiesce aborted")
        resume_reply = hooks.resume_after_restore(failed_receipt)
        self.assertFalse(resume_reply["ok"])
        self.assertIn("quiesce aborted", resume_reply["error"])
        self.assertFalse(resume_reply["participants_ok"])

    def test_full_snapshot_driver_lifecycle_with_real_worker_and_post_resume_live_edit(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        base_fingerprint = lab.status()["fingerprint"]
        hooks = lab.quiesce_hooks()
        self.assertIsInstance(hooks, TransformersQuiesceHooks)
        self.assertIsInstance(hooks, QuiesceHooks)

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
            control_path = Path(control_dir)
            ready_sentinel = control_path / "ready-for-snapshot"
            restore_complete_sentinel = control_path / "restore-complete"
            restore_ready_sentinel = control_path / "transformers-restore-ready"
            for stale_file in (ready_sentinel, restore_complete_sentinel, restore_ready_sentinel):
                stale_file.write_text("stale", encoding="utf-8")

            observed: dict[str, Any] = {}

            def simulate_snapshot_agent() -> None:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if ready_sentinel.exists() and not restore_complete_sentinel.exists():
                        observed["saw_clean_ready"] = True
                        restore_complete_sentinel.write_text("ok\n", encoding="utf-8")
                        return
                    time.sleep(0.01)

            agent_thread = threading.Thread(target=simulate_snapshot_agent, daemon=True)
            agent_thread.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="transformers", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            lifecycle = driver.checkpoint_and_wait(hooks)
            agent_thread.join(timeout=2.0)

            self.assertTrue(lifecycle.ok, lifecycle.to_dict())
            self.assertTrue(observed.get("saw_clean_ready"))
            self.assertTrue(restore_ready_sentinel.exists())
            self.assertEqual(lifecycle.quiesce.released, ("kv_cache", "weights"))
            self.assertEqual(lifecycle.quiesce.token, "transformers:cpu_offload:kv_cache,weights")
            self.assertTrue(lifecycle.resume_details["participants_ok"])
            self.assertEqual(lifecycle.resume_details["fingerprint"], base_fingerprint)

        self.assertEqual(lab.check()["output"], base_output)
        install_reply = lab.install(lab.patches()["visible"])
        self.assertTrue(install_reply["ok"], install_reply)
        edited_check = lab.check()
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], base_output)
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual(lab.check()["output"], base_output)

    def test_cache_clear_only_mode_skips_weight_offload(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks(offload_to_cpu=False, pin_memory=False)
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok, receipt)
        self.assertEqual(receipt.released, ("kv_cache",))
        self.assertEqual(receipt.token, "transformers:cache_clear:kv_cache")
        resume_reply = hooks.resume_after_restore(receipt)
        self.assertTrue(resume_reply["ok"], resume_reply)
        self.assertEqual(resume_reply["resumed"], ["kv_cache"])

    def test_stopped_dirty_and_quiesce_probe_failures_fail_closed_with_rollback(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks()
        sample_receipt = QuiesceReceipt(ok=True, released=("kv_cache", "weights"))

        # Post-quiesce probe failure triggers rollback resume and blocks ready-for-snapshot.
        with mock.patch.object(lab, "_probe_participants", return_value=(False, "worker0 unreachable")):
            failed_probe_receipt = hooks.quiesce_for_snapshot()
        self.assertFalse(failed_probe_receipt.ok)
        self.assertIn("worker0 unreachable", failed_probe_receipt.error or "")
        # Rollback restored weights cleanly, so generation still works!
        self.assertTrue(lab.check()["ok"])

        # Dirty channel fails closed across all three hooks.
        lab.dirty = True
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            hooks.warmup_for_snapshot()
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(sample_receipt)["ok"])

        # Stopped worker fails closed across all three hooks.
        lab.dirty = False
        lab.close()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            hooks.warmup_for_snapshot()
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(sample_receipt)["ok"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TorchRuntimeQuiesceTests(unittest.TestCase):
    def test_tied_weights_buffer_restoration_and_partial_quiesce_rollback_on_cpu(self) -> None:
        import torch

        class _TiedTinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embed.weight
                self.register_buffer("scale_buf", torch.ones(8))

        model = _TiedTinyModel()
        self.assertIs(model.lm_head.weight, model.embed.weight)
        runtime = {
            "mode": "real", "model": model, "tokenizer": None,
            "device": "cpu", "seed": 0, "model_module": __name__, "quiesced": False,
        }

        quiesce_reply = _quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)
        self.assertTrue(quiesce_reply["ok"], quiesce_reply)
        self.assertTrue(runtime["quiesced"])
        self.assertEqual(quiesce_reply["released"], ["kv_cache", "weights"])
        self.assertIs(model.lm_head.weight, model.embed.weight)

        resume_reply = _resume_runtime(runtime, quiesce_reply["released"])
        self.assertTrue(resume_reply["ok"], resume_reply)
        self.assertFalse(runtime["quiesced"])
        self.assertEqual(resume_reply["resumed"], ["kv_cache", "weights"])
        self.assertIs(model.lm_head.weight, model.embed.weight)
        self.assertEqual(model.scale_buf.device, torch.device("cpu"))

        # Cache-clear-only resume does NOT report "weights" when tensors are already on target device.
        cache_quiesce = _quiesce_runtime(runtime, offload_to_cpu=False, pin_memory=False)
        self.assertEqual(cache_quiesce["released"], ["kv_cache"])
        cache_resume = _resume_runtime(runtime, cache_quiesce["released"])
        self.assertEqual(cache_resume["resumed"], ["kv_cache"])

        # Partial-quiesce failure after model.to("cpu") rolls back cleanly and clears quiesced flag.
        with mock.patch("adapter.agent.gc.collect", side_effect=RuntimeError("gc boom")):
            failed_quiesce = _quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)
        self.assertFalse(failed_quiesce["ok"])
        self.assertIn("gc boom", failed_quiesce["error"])
        self.assertFalse(runtime["quiesced"])
        self.assertIs(model.lm_head.weight, model.embed.weight)

    def test_sharded_accelerate_model_quiesce_never_calls_model_to_and_preserves_tied_weights(self) -> None:
        """Golden case: hf_device_map sharded model offloads/resumes per-tensor without model.to(),
        handles cache-clear-only resume, blocks double-quiesce, and preserves shared storage views."""
        import torch

        class _AccelerateShardedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embed.weight
                # Distinct Parameter + non-contiguous slice sharing one underlying storage!
                base = torch.arange(32, dtype=torch.float32).reshape(4, 8)
                self.alias_a = torch.nn.Parameter(base)
                self.alias_b = torch.nn.Parameter(base.view(8, 4))
                self.register_buffer("slice_buf", base[1:3, ::2])
                self.hf_device_map = {"embed": "cpu", "lm_head": "cpu"}

            def get_input_embeddings(self) -> torch.nn.Module:
                return self.embed

            def to(self, *args: Any, **kwargs: Any) -> Any:
                raise ValueError("You shouldn't move a model that is dispatched using accelerate hooks.")

        model = _AccelerateShardedModel()
        self.assertEqual(_resolve_input_device(model, "cpu", torch), torch.device("cpu"))
        # _resolve_input_device fallback when get_input_embeddings() is None and hf_device_map uses "0" / int / disk.
        class _MetaFallbackModel(torch.nn.Module):
            def __init__(self, hf_map: dict[str, Any]) -> None:
                super().__init__()
                self.hf_device_map = hf_map
                self.p = torch.nn.Parameter(torch.empty(2, device="meta"))

            def get_input_embeddings(self) -> Any:
                return None

        self.assertEqual(_resolve_input_device(_MetaFallbackModel({"a": "disk", "b": "0"}), "cpu", torch), torch.device("cuda:0"))
        self.assertEqual(_resolve_input_device(_MetaFallbackModel({"a": "meta", "b": 2}), "cpu", torch), torch.device("cuda:2"))

        runtime = {
            "mode": "real", "model": model, "tokenizer": None,
            "device": "cpu", "devices": ["cpu"], "hf_device_map": dict(model.hf_device_map),
            "seed": 0, "model_module": __name__, "quiesced": False,
        }

        # 1. P0-1: cache-clear-only quiesce(offload_to_cpu=False) + resume NEVER calls model.to()!
        cc_q = _quiesce_runtime(runtime, offload_to_cpu=False, pin_memory=False)
        self.assertTrue(cc_q["ok"], cc_q)
        self.assertEqual(cc_q["released"], ["kv_cache"])
        self.assertNotIn("saved_devices", runtime)
        # Double quiesce while already quiesced fails closed!
        self.assertFalse(_quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)["ok"])
        cc_r = _resume_runtime(runtime, cc_q["released"])
        self.assertTrue(cc_r["ok"], cc_r)
        self.assertEqual(cc_r["resumed"], ["kv_cache"])
        self.assertFalse(runtime["quiesced"])

        # 2. Full sharded offload + double-quiesce refusal + selective resume + full restore!
        q_reply = _quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)
        self.assertTrue(q_reply["ok"], q_reply)
        self.assertIn("weights", q_reply["released"])
        self.assertIn("saved_devices", runtime)
        self.assertFalse(_quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)["ok"])
        self.assertIs(model.lm_head.weight, model.embed.weight)
        self.assertEqual(
            model.alias_a.untyped_storage().data_ptr(),
            model.alias_b.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            model.alias_a.untyped_storage().data_ptr(),
            model.slice_buf.untyped_storage().data_ptr(),
        )

        # Selective resume without "weights" keeps saved_devices intact until full restore.
        sel_r = _resume_runtime(runtime, ["kv_cache"])
        self.assertTrue(sel_r["ok"], sel_r)
        self.assertEqual(sel_r["resumed"], ["kv_cache"])
        self.assertIn("saved_devices", runtime)

        r_reply = _resume_runtime(runtime, q_reply["released"])
        self.assertTrue(r_reply["ok"], r_reply)
        self.assertIn("weights", r_reply["resumed"])
        self.assertNotIn("saved_devices", runtime)
        self.assertIs(model.lm_head.weight, model.embed.weight)
        self.assertEqual(
            model.alias_a.untyped_storage().data_ptr(),
            model.alias_b.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            model.alias_a.untyped_storage().data_ptr(),
            model.slice_buf.untyped_storage().data_ptr(),
        )

        # Verify _move_tensors_preserving_storage actually allocates fresh storage and preserves aliasing/views.
        old_ptr = model.alias_a.untyped_storage().data_ptr()
        orig_slice = model.slice_buf.clone()

        class _FakeTargetDevice:
            type = "cpu"

            def __str__(self) -> str:
                return "fake:0"

        fake_dev = _FakeTargetDevice()
        orig_empty = torch.empty
        orig_to = torch.Tensor.to

        def _fake_empty(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("device") is fake_dev:
                kwargs = {**kwargs, "device": "cpu"}
            return orig_empty(*args, **kwargs)

        def _fake_to(self_t: Any, *args: Any, **kwargs: Any) -> Any:
            if args and args[0] is fake_dev:
                return self_t.clone()
            return orig_to(self_t, *args, **kwargs)

        with mock.patch.object(torch, "empty", side_effect=_fake_empty), mock.patch.object(
            torch.Tensor, "to", new=_fake_to
        ):
            moved = _move_tensors_preserving_storage(
                [model.alias_a, model.alias_b, model.slice_buf],
                {id(model.alias_a): fake_dev, id(model.alias_b): fake_dev, id(model.slice_buf): fake_dev},
                torch,
            )
        self.assertEqual(moved, [fake_dev])
        new_ptr = model.alias_a.untyped_storage().data_ptr()
        self.assertNotEqual(new_ptr, old_ptr)
        self.assertEqual(model.alias_b.untyped_storage().data_ptr(), new_ptr)
        self.assertEqual(model.slice_buf.untyped_storage().data_ptr(), new_ptr)
        self.assertIsInstance(model.alias_a, torch.nn.Parameter)
        self.assertIsInstance(model.alias_b, torch.nn.Parameter)
        self.assertTrue(torch.equal(model.slice_buf, orig_slice))
        # Mutation through alias_a propagates into slice_buf, proving true shared storage!
        with torch.no_grad():
            model.alias_a[1, 0] = 999.0
        self.assertEqual(float(model.slice_buf[0, 0]), 999.0)

        # 3. Rollback on mid-quiesce error also never calls model.to() and cleans saved_devices.
        with mock.patch("adapter.agent.gc.collect", side_effect=RuntimeError("sharded gc boom")):
            fail_reply = _quiesce_runtime(runtime, offload_to_cpu=True, pin_memory=False)
        self.assertFalse(fail_reply["ok"])
        self.assertIn("sharded gc boom", fail_reply["error"])
        self.assertNotIn("saved_devices", runtime)
        self.assertFalse(runtime["quiesced"])
        self.assertIs(model.lm_head.weight, model.embed.weight)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers not installed")
class RealTransformersPatchSuiteTests(unittest.TestCase):
    def test_real_patches_anchors_and_kernel_staging_match_conformance_contracts(self) -> None:
        suite = real_patches(rsibox_harness.read_module)
        self.assertEqual(set(suite["refused"]), set(REFUSALS))

        def prepare_files(txn_id: str, files_map: dict[str, str]) -> dict[str, Any]:
            participant = kernel_coordinator.LocalParticipant("worker0")
            modules_spec = {
                module_name: {
                    "source": patched_source,
                    "base_hash": kernel_locate.text_hash(rsibox_harness.read_module(module_name)),
                }
                for module_name, patched_source in files_map.items()
            }
            return participant.prepare({"id": txn_id, "modules": modules_spec})

        # 1. Verify all 4 REFUSALS classes block with the exact required conformance code.
        for refusal_class, expected_code in REFUSALS.items():
            with self.subTest(refusal=refusal_class):
                reply = prepare_files(f"ref-{refusal_class}", suite["refused"][refusal_class]["files"])
                blocking = kernel_findings.decide(reply.get("findings") or ())["blocking"]
                self.assertIn(expected_code, blocking, reply)
                if refusal_class == "empty":
                    self.assertEqual(blocking, ["empty_change"])

        # 2. Verify "multi" stages >= 3 distinct functions across >= 2 distinct modules cleanly.
        multi_reply = prepare_files("multi-check", suite["multi"]["files"])
        self.assertTrue(multi_reply["ok"], multi_reply)
        self.assertEqual(kernel_findings.decide(multi_reply.get("findings") or ())["blocking"], [])
        staged_keys = list((multi_reply.get("staged") or {}).keys())
        self.assertGreaterEqual(len(staged_keys), 3, staged_keys)
        self.assertGreaterEqual(len({key.split(":")[0] for key in staged_keys}), 2, staged_keys)

        # 3. Verify "unreached" stages load-time-only PreTrainedModel.tie_weights cleanly.
        unreached_reply = prepare_files("unreached-check", suite["unreached"]["files"])
        self.assertTrue(unreached_reply["ok"], unreached_reply)
        self.assertEqual(kernel_findings.decide(unreached_reply.get("findings") or ())["blocking"], [])
        unreached_keys = list((unreached_reply.get("staged") or {}).keys())
        self.assertTrue(
            any(key.startswith("transformers.modeling_utils:PreTrainedModel.tie_weights:") for key in unreached_keys),
            unreached_keys,
        )


if __name__ == "__main__":
    unittest.main()
