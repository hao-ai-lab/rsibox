"""Golden CPU tests for the SGLang live-lab adapter, decode-graph recapture, and TP rollback."""
import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

from adapter.conformance_lab import SglangConformanceLab
from adapter.fixture.harness import edit_once, sglang_env_config
from adapter.toy_lab import LAYERNORM, ToySglangLab


class SglangAdapterTests(unittest.TestCase):
    def test_graph_edit_requires_recapture_and_recapture_matches_fresh(self):
        """A layer edit without recapture leaves stale decode graphs; recapturing matches fresh."""
        lab = ToySglangLab(tp_size=2)
        try:
            base_out = lab.check()["output"]
            norm_src = lab._base(LAYERNORM)
            edited_norm = edit_once(norm_src, "(h * 7 + 1) % MODULUS", "(h * 11 + 1) % MODULUS")
            fresh_out = lab.fresh_check({"files": {LAYERNORM: edited_norm}})["output"]
            self.assertNotEqual(fresh_out, base_out)

            # Without recapture: prefill runs new Python (proven True), decode replays stale graph.
            stale_install = lab.install({"files": {LAYERNORM: edited_norm}, "recapture": False})
            self.assertTrue(stale_install["ok"])
            self.assertEqual(stale_install["mechanisms"], ["R0 swap"])
            stale_check = lab.check()
            self.assertTrue(stale_check["proven"])
            self.assertNotEqual(stale_check["output"], fresh_out)
            self.assertTrue(lab.uninstall()["ok"])

            # Default GRAPH install: recaptures decode graphs in the commit gap; capture calls
            # stay out of live witness counts until a real workload step runs.
            recap_install = lab.install({"files": {LAYERNORM: edited_norm}})
            self.assertTrue(recap_install["ok"])
            self.assertIn("R1 recapture", recap_install["mechanisms"])
            txn_id = recap_install["id"]
            for rank in range(lab.ranks):
                pre_proof = lab.rank_call(rank, "proof", txn_id)
                self.assertEqual(list(pre_proof["witness"].values()), [0])
                self.assertGreater(min(pre_proof["witness_capture"].values()), 0)

            live_check = lab.check()
            self.assertTrue(live_check["proven"])
            self.assertEqual(live_check["output"], fresh_out)
            self.assertTrue(lab.uninstall()["ok"])
            self.assertEqual(lab.check()["output"], base_out)
        finally:
            lab.close()

    def test_single_rank_recapture_failure_rolls_back_all_tp_ranks(self):
        """If rank1 fails decode-graph recapture after rank0 succeeds, every rank rolls back."""
        lab = ToySglangLab(tp_size=2)
        try:
            base_status = lab.status()
            base_out = lab.check()["output"]
            norm_src = lab._base(LAYERNORM)
            edited_norm = edit_once(norm_src, "(h * 7 + 1) % MODULUS", "(h * 13 + 1) % MODULUS")

            self.assertTrue(lab.inject("fail_recapture", participant="rank1")["ok"])
            failed = lab.install({"files": {LAYERNORM: edited_norm}})
            self.assertFalse(failed["ok"])
            self.assertEqual(failed["outcome"], "rolled_back")
            self.assertIn("rebuild_failed", failed["codes"])

            after_status = lab.status()
            self.assertEqual(after_status["state"], "base")
            self.assertEqual(after_status["fingerprint"], base_status["fingerprint"])
            for rank in range(lab.ranks):
                graph_info = lab.rank_call(rank, "graphs")
                self.assertEqual(graph_info["capture_bs"], [1, 2, 4])
            self.assertEqual(lab.check()["output"], base_out)
        finally:
            lab.close()

    def test_selective_sizes_and_recapture_extend_drop_on_rank_failure(self):
        """Partial `sizes=[4]` leaves smaller batches eager; a failed `extend` drops on all ranks."""
        lab = ToySglangLab(tp_size=2)
        try:
            norm_src = lab._base(LAYERNORM)
            edited_norm = edit_once(norm_src, "(h * 7 + 1) % MODULUS", "(h * 11 + 1) % MODULUS")
            installed = lab.install({"files": {LAYERNORM: edited_norm}, "sizes": [4]})
            self.assertTrue(installed["ok"])
            txn_id = installed["id"]
            for rank in range(lab.ranks):
                self.assertEqual(lab.rank_call(rank, "graphs")["capture_bs"], [4])

            # Single-request batch (bs=1) falls back to eager on the new code and matches fresh.
            single_workload = {"input_ids": [[5, 9, 2, 7, 1]], "max_new_tokens": 6, "temperature": 0.0}
            fresh_single = lab.fresh_check({"files": {LAYERNORM: edited_norm}}, single_workload)["output"]
            self.assertEqual(lab.check(single_workload)["output"], fresh_single)

            # Fault on rank1 during extend: rank0's extend is dropped in the same gap.
            self.assertTrue(lab.inject("fail_recapture", participant="rank1")["ok"])
            failed_ext = lab.recapture_extend(txn_id, sizes=[1, 2])
            self.assertFalse(failed_ext["ok"])
            for rank in range(lab.ranks):
                self.assertEqual(lab.rank_call(rank, "graphs")["capture_bs"], [4])

            # Clean extend brings both ranks to [1, 2, 4]; uninstall restores base graphs.
            ok_ext = lab.recapture_extend(txn_id, sizes=[1, 2])
            self.assertTrue(ok_ext["ok"])
            for rank in range(lab.ranks):
                self.assertEqual(lab.rank_call(rank, "graphs")["capture_bs"], [1, 2, 4])
            self.assertTrue(lab.uninstall()["ok"])
        finally:
            lab.close()

    def test_overlap_scheduler_and_normal_loop_agree_across_state_reset(self):
        """Normal and overlap loops give identical outputs before and after deterministic reset."""
        normal_lab = ToySglangLab(tp_size=2, disable_overlap_schedule=True)
        overlap_lab = ToySglangLab(tp_size=2, disable_overlap_schedule=False)
        try:
            out_normal = normal_lab.check()["output"]
            out_overlap_1 = overlap_lab.check()["output"]
            out_overlap_cached = overlap_lab.check()["output"]
            reset_reply = overlap_lab.reset("state")
            self.assertTrue(reset_reply["ok"])
            self.assertEqual(reset_reply["reset"], ["radix_cache", "kv_pool", "mamba_pool", "rng"])
            out_overlap_after = overlap_lab.check()["output"]
            self.assertEqual(out_normal, out_overlap_1)
            self.assertEqual(out_normal, out_overlap_cached)
            self.assertEqual(out_normal, out_overlap_after)
        finally:
            normal_lab.close()
            overlap_lab.close()

    def test_conformance_lab_launch_environment_and_lazy_imports(self):
        """`SglangConformanceLab.launch_environment` maps CLI flags cleanly without GPU imports."""
        self.assertFalse(any(m.split(".")[0] in ("torch", "sglang") for m in sys.modules))
        env = SglangConformanceLab.launch_environment([
            "--model", "Qwen/Qwen3-8B",
            "--tp", "2",
            "--mem-fraction", "0.4",
            "--max-bs", "8",
            "--port", "31000",
            "--work", "rel-sgl-work",
            "--disable-cuda-graph",
            "--extra", '{"context_length": 2048}',
            "--check-timeout", "30",
            "--rpc-timeout", "240",
        ])
        self.assertEqual(env["FASTRSI_SGLANG_MODEL"], "Qwen/Qwen3-8B")
        self.assertEqual(env["FASTRSI_SGLANG_TP"], "2")
        self.assertEqual(env["FASTRSI_SGLANG_MEM_FRACTION"], "0.4")
        self.assertEqual(env["FASTRSI_SGLANG_MAX_BS"], "8")
        self.assertEqual(env["TMPDIR"], os.path.abspath("rel-sgl-work"))
        self.assertEqual(env["FASTRSI_SGLANG_CHECK_TIMEOUT"], "30.0")
        self.assertEqual(env["FASTRSI_SGLANG_RPC_TIMEOUT"], "240.0")
        self.assertEqual(
            json.loads(env["FASTRSI_SGLANG_EXTRA_ARGS"]),
            {"context_length": 2048, "port": 31000, "disable_cuda_graph": True},
        )
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = sglang_env_config()
        self.assertEqual(cfg["model_path"], "Qwen/Qwen3-8B")
        self.assertEqual(cfg["tp_size"], 2)
        self.assertEqual(cfg["mem_fraction_static"], 0.4)
        self.assertEqual(cfg["port"], 31000)
        self.assertTrue(cfg["disable_cuda_graph"])
        self.assertEqual(cfg["context_length"], 2048)

        minimal = SglangConformanceLab.launch_environment(["--", "--model", "/models/qwen"])
        self.assertEqual(minimal, {"FASTRSI_SGLANG_MODEL": "/models/qwen"})
        self.assertNotIn("FASTRSI_SGLANG_EXTRA_ARGS", minimal)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            SglangConformanceLab.launch_environment(["--extra", "[1, 2]"])
        self.assertEqual(cm.exception.code, 2)

    def test_agent_rank_fallback_resolves_runtime_context(self):
        """`agent._rank` falls back cleanly across modern SGLang runtime_context getters."""
        from types import SimpleNamespace
        from adapter import agent

        self.assertEqual(agent._rank(SimpleNamespace(tp_rank=3)), 3)
        self.assertEqual(agent._rank(SimpleNamespace(tp_worker=SimpleNamespace(tp_rank=2))), 2)
        sched_none = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace()))
        fake_ctx = SimpleNamespace(get_parallel=lambda: SimpleNamespace(tp_rank=1))
        with mock.patch.dict(sys.modules, {"sglang.srt.runtime_context": fake_ctx}):
            self.assertEqual(agent._rank(sched_none), 1)
        with mock.patch.dict(sys.modules, {"sglang.srt.runtime_context": None, "sglang.srt.distributed": None}):
            self.assertEqual(agent._rank(sched_none), 0)


if __name__ == "__main__":
    unittest.main()
