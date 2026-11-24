"""Comprehensive CPU golden unit tests for the HuggingFace Transformers RSIBox adapter."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
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
from adapter.agent import _quiesce_runtime, _resume_runtime  # noqa: E402
from adapter.fixture.patches import real_patches  # noqa: E402
from adapter.fixture.toy_hf.modeling import ToyCausalLM  # noqa: E402
from adapter.lab import TransformersLab  # noqa: E402
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
