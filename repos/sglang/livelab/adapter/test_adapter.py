"""Golden CPU tests for the SGLang live-lab adapter, decode-graph recapture, and TP rollback."""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
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
        self.assertEqual(env["RSIBOX_SGLANG_MODEL"], "Qwen/Qwen3-8B")
        self.assertEqual(env["RSIBOX_SGLANG_TP"], "2")
        self.assertEqual(env["RSIBOX_SGLANG_MEM_FRACTION"], "0.4")
        self.assertEqual(env["RSIBOX_SGLANG_MAX_BS"], "8")
        self.assertEqual(env["TMPDIR"], os.path.abspath("rel-sgl-work"))
        self.assertEqual(env["RSIBOX_SGLANG_CHECK_TIMEOUT"], "30.0")
        self.assertEqual(env["RSIBOX_SGLANG_RPC_TIMEOUT"], "240.0")
        self.assertEqual(
            json.loads(env["RSIBOX_SGLANG_EXTRA_ARGS"]),
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
        self.assertEqual(minimal, {"RSIBOX_SGLANG_MODEL": "/models/qwen"})
        self.assertNotIn("RSIBOX_SGLANG_EXTRA_ARGS", minimal)

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

    def test_faults_inject_and_module_state_fallback_without_legacy_attrs(self):
        """`faults.inject` uses participant name without `.tp_rank`; `module_state` patch falls back cleanly."""
        from types import SimpleNamespace
        from adapter.fixture import faults, harness

        class DummyRunner:
            def sample(self):
                return "ok"

        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=DummyRunner()))
        lp = SimpleNamespace(name="rank1")
        state = {"faults": {}}
        reply = faults.inject(sched, lp, state, "crash")
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["participant"], "rank1")
        with self.assertRaisesRegex(RuntimeError, "rank1: injected crash in sample"):
            sched.tp_worker.model_runner.sample()
        self.assertEqual(sched.tp_worker.model_runner.sample(), "ok")

        sampler_src = (
            'SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")\n'
            'def _trace_e2e_sampler(stage, **fields):\n'
            '    details = " ".join(f"{key}={value}" for key, value in fields.items())\n'
            '    return details\n'
            'class Sampler:\n'
            '    def forward(self, logits):\n'
            '        batch_next_token_ids = torch.argmax(logits, -1)\n'
            '        return logits\n'
        )
        worker_src = (
            '    def set_hicache_consumer(self, consumer_index: int):\n        pass\n'
            '    def _init_dllm_algorithm(self):\n        pass\n'
        )
        patches = harness.sglang_patches(
            lambda m: sampler_src if "sampler" in m else worker_src,
            "sglang.srt.layers.sampler",
            "sglang.srt.managers.tp_worker",
        )
        ms = next(iter(patches["refused"]["module_state"]["files"].values()))
        self.assertIn("bool(get_bool_env_var(", ms)
        unreached_src = next(iter(patches["unreached"]["files"].values()))
        self.assertNotIn("param.data.neg_()", unreached_src)
        self.assertIn("param.data = param.data.neg().contiguous()", unreached_src)

    def test_fork_subprocs_cli_flag_and_conformance_env(self):
        """`--fork-subprocs` maps to `RSIBOX_SGLANG_FORK_SUBPROCS=1` and is omitted by default."""
        from adapter import conformance_lab as clab_mod
        from adapter import lab as lab_mod

        default_env = SglangConformanceLab.launch_environment(["--model", "Qwen/Qwen3-8B"])
        self.assertNotIn("RSIBOX_SGLANG_FORK_SUBPROCS", default_env)

        fork_env = SglangConformanceLab.launch_environment(["--model", "Qwen/Qwen3-8B", "--fork-subprocs"])
        self.assertEqual(fork_env["RSIBOX_SGLANG_FORK_SUBPROCS"], "1")

        seen_flags = []

        def fake_super_init(self_lab, config, modules, receipts=None, rpc_timeout=30.0, check_timeout=120.0,
                            fork_subprocs=False, weight_cache_dir=None):
            seen_flags.append(fork_subprocs)
            self_lab.fork_subprocs = bool(fork_subprocs)
            self_lab.weight_cache_dir = weight_cache_dir
            self_lab.dirty = True

        prev_shared = SglangConformanceLab._shared
        prev_loops = len(clab_mod._LOOPS)

        def cleanup():
            SglangConformanceLab._shared = prev_shared
            while len(clab_mod._LOOPS) > prev_loops:
                loop = clab_mod._LOOPS.pop()
                with contextlib.suppress(Exception):
                    loop.close()

        self.addCleanup(cleanup)
        SglangConformanceLab._shared = None
        with mock.patch.object(lab_mod.SglangLab, "__init__", fake_super_init):
            with mock.patch.dict(os.environ, default_env, clear=False):
                os.environ.pop("RSIBOX_SGLANG_FORK_SUBPROCS", None)
                os.environ.pop("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", None)
                SglangConformanceLab()
            SglangConformanceLab._shared = None
            with mock.patch.dict(os.environ, fork_env, clear=False):
                os.environ.pop("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", None)
                SglangConformanceLab()
        self.assertEqual(seen_flags, [False, True])

    def _make_stub_lab_class(self, observed):
        import multiprocessing as mp
        from multiprocessing.connection import Client
        from adapter.lab import SglangLab

        class _StubEngine:
            run_scheduler_process_func = staticmethod(lambda *a, **k: None)

            def __init__(self, **cfg):
                observed.append({
                    "default_method": mp.get_start_method(allow_none=True),
                    "spawn_ctx": mp.get_context("spawn").get_start_method(),
                    "numa_v2": os.environ.get("SGLANG_NUMA_BIND_V2"),
                })
                self._conn = Client(cfg["_sock_path"], "AF_UNIX")
                self._conn.send({"rank": 0})

            def shutdown(self):
                with contextlib.suppress(OSError):
                    self._conn.close()

        class _StubLab(SglangLab):
            def __init__(self, **kwargs):
                super().__init__({"tp_size": 1}, (), rpc_timeout=2.0, check_timeout=1.0, **kwargs)

            def engine_class(self):
                self.config["_sock_path"] = self.sock_path
                return _StubEngine

        return _StubLab

    def test_fork_subprocs_clean_parent_preloads_once_and_enables_fork(self):
        """Clean parent preloads `SUBPROC_PRELOAD` once per process and enables fork before engine init."""
        import multiprocessing as mp
        import forksafe
        from adapter import lab as lab_mod
        from services import zygote as sgl_zygote

        orig_state = dict(lab_mod._FORK_SUBPROCS_STATE)
        orig_method = mp.get_start_method(allow_none=True)
        orig_get_ctx = mp.get_context
        orig_set_method = mp.set_start_method
        orig_numa = os.environ.get("SGLANG_NUMA_BIND_V2")

        def restore_globals():
            lab_mod._FORK_SUBPROCS_STATE.clear()
            lab_mod._FORK_SUBPROCS_STATE.update(orig_state)
            mp.get_context = orig_get_ctx
            mp.set_start_method = orig_set_method
            mp.set_start_method(orig_method, force=True)
            if orig_numa is None:
                os.environ.pop("SGLANG_NUMA_BIND_V2", None)
            else:
                os.environ["SGLANG_NUMA_BIND_V2"] = orig_numa

        self.addCleanup(restore_globals)
        lab_mod._FORK_SUBPROCS_STATE.update({"preloaded": False, "clear_nvml_env": False})
        os.environ.pop("SGLANG_NUMA_BIND_V2", None)

        observed = []
        stub_cls = self._make_stub_lab_class(observed)
        with (
            mock.patch.object(forksafe, "poison_reasons", return_value=[]),
            mock.patch.object(forksafe, "cuda_initialized", return_value=False),
            mock.patch.object(sgl_zygote, "install_guards", return_value=({"nvml_env_set": False}, lambda: None)) as m_guards,
            mock.patch.object(sgl_zygote, "preload", return_value={}) as m_preload,
            mock.patch.object(sgl_zygote, "enable_fork_subprocs", wraps=sgl_zygote.enable_fork_subprocs) as m_enable,
        ):
            lab1 = stub_cls(fork_subprocs=True)
            try:
                self.assertTrue(lab1.fork_subprocs)
                self.assertEqual(lab1.fork_subprocs_fallback, [])
                self.assertEqual(observed[-1], {"default_method": "fork", "spawn_ctx": "fork", "numa_v2": "0"})
            finally:
                lab1.close()

            # Closing restores multiprocessing and SGLANG_NUMA_BIND_V2 cleanly.
            self.assertIsNone(os.environ.get("SGLANG_NUMA_BIND_V2"))
            self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")

            # A second engine start in the same process reuses the one-time preload.
            lab2 = stub_cls(fork_subprocs=True)
            try:
                self.assertEqual(lab2.fork_subprocs_fallback, [])
                self.assertEqual(observed[-1], {"default_method": "fork", "spawn_ctx": "fork", "numa_v2": "0"})
            finally:
                lab2.close()

        m_guards.assert_called_once()
        m_preload.assert_called_once_with(sgl_zygote.SUBPROC_PRELOAD)
        self.assertEqual(m_enable.call_count, 2)

    def test_fork_subprocs_poisoned_parent_falls_back_cleanly(self):
        """Non-empty `poison_reasons()` or preload CUDA init records fallback and stays on spawn."""
        import forksafe
        from adapter import lab as lab_mod
        from services import zygote as sgl_zygote

        orig_state = dict(lab_mod._FORK_SUBPROCS_STATE)
        orig_numa = os.environ.get("SGLANG_NUMA_BIND_V2")

        def restore_globals():
            lab_mod._FORK_SUBPROCS_STATE.clear()
            lab_mod._FORK_SUBPROCS_STATE.update(orig_state)
            if orig_numa is None:
                os.environ.pop("SGLANG_NUMA_BIND_V2", None)
            else:
                os.environ["SGLANG_NUMA_BIND_V2"] = orig_numa

        self.addCleanup(restore_globals)
        lab_mod._FORK_SUBPROCS_STATE.update({"preloaded": False, "clear_nvml_env": False})
        os.environ.pop("SGLANG_NUMA_BIND_V2", None)

        observed = []
        stub_cls = self._make_stub_lab_class(observed)
        reasons = ["CUDA driver initialized in the zygote (cuInit)"]
        with (
            mock.patch.object(forksafe, "poison_reasons", return_value=reasons),
            mock.patch.object(sgl_zygote, "preload") as m_preload,
            mock.patch.object(sgl_zygote, "enable_fork_subprocs") as m_enable,
        ):
            lab = stub_cls(fork_subprocs=True)
            try:
                self.assertTrue(lab.fork_subprocs)
                self.assertEqual(lab.fork_subprocs_fallback, reasons)
                self.assertEqual(observed[-1]["spawn_ctx"], "spawn")
                self.assertIsNone(observed[-1]["numa_v2"])
            finally:
                lab.close()
        m_preload.assert_not_called()
        m_enable.assert_not_called()

        # Preload raising SystemExit (CUDA init mid-preload) also falls back cleanly without raising.
        with (
            mock.patch.object(forksafe, "poison_reasons", side_effect=[[], ["torch CUDA runtime initialized"]]),
            mock.patch.object(sgl_zygote, "install_guards", return_value=({"nvml_env_set": False}, lambda: None)),
            mock.patch.object(sgl_zygote, "preload", side_effect=SystemExit("initialized CUDA")),
            mock.patch.object(sgl_zygote, "enable_fork_subprocs") as m_enable2,
        ):
            lab_pre = stub_cls(fork_subprocs=True)
            try:
                self.assertEqual(lab_pre.fork_subprocs_fallback, ["torch CUDA runtime initialized"])
                self.assertEqual(observed[-1]["spawn_ctx"], "spawn")
                self.assertIsNone(observed[-1]["numa_v2"])
            finally:
                lab_pre.close()
        m_enable2.assert_not_called()

    def test_sglang_env_config_avoids_importing_sglang_via_ast_probe(self):
        """`sglang_env_config()` reads `ServerArgs` fields via AST without importing `sglang`/`torch`."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "sglang", "srt")
            os.makedirs(pkg, exist_ok=True)
            with open(os.path.join(tmp, "sglang", "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write("raise AssertionError('sglang package must not be imported')\n")
            with open(os.path.join(pkg, "server_args.py"), "w", encoding="utf-8") as fh:
                fh.write(
                    "class ServerArgs:\n"
                    "    cuda_graph_max_bs_decode: int = -1\n"
                )
            with (
                mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m", "RSIBOX_SGLANG_MAX_BS": "4"}, clear=False),
                mock.patch.dict(sys.modules),
                mock.patch.object(sys, "path", [tmp, *sys.path]),
            ):
                for key in list(sys.modules):
                    if key == "sglang" or key.startswith("sglang."):
                        sys.modules.pop(key, None)
                cfg = sglang_env_config()
                self.assertEqual(cfg["cuda_graph_max_bs_decode"], 4)
                self.assertTrue(cfg["disable_prefill_cuda_graph"])
                self.assertNotIn("cuda_graph_max_bs", cfg)
                self.assertNotIn("sglang", sys.modules)
                self.assertNotIn("sglang.srt.server_args", sys.modules)

            # Without `cuda_graph_max_bs_decode`, falls back to legacy `cuda_graph_max_bs`.
            with open(os.path.join(pkg, "server_args.py"), "w", encoding="utf-8") as fh:
                fh.write("class ServerArgs:\n    tp_size: int = 1\n")
            with (
                mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m", "RSIBOX_SGLANG_MAX_BS": "8"}, clear=False),
                mock.patch.dict(sys.modules),
                mock.patch.object(sys, "path", [tmp, *sys.path]),
            ):
                for key in list(sys.modules):
                    if key == "sglang" or key.startswith("sglang."):
                        sys.modules.pop(key, None)
                cfg_legacy = sglang_env_config()
                self.assertEqual(cfg_legacy["cuda_graph_max_bs"], 8)
                self.assertNotIn("cuda_graph_max_bs_decode", cfg_legacy)

        # Already-loaded `sys.modules['sglang.srt.server_args']` is respected directly.
        fake_mod = types.ModuleType("sglang.srt.server_args")
        fake_mod.ServerArgs = type("ServerArgs", (), {"cuda_graph_max_bs_decode": 4})
        with (
            mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m"}, clear=False),
            mock.patch.dict(sys.modules, {"sglang.srt.server_args": fake_mod}),
        ):
            cfg_loaded = sglang_env_config()
            self.assertEqual(cfg_loaded["cuda_graph_max_bs_decode"], 4)
            self.assertTrue(cfg_loaded["disable_prefill_cuda_graph"])

    def test_conformance_lab_preloads_fork_subprocs_before_env_config(self):
        """`SglangConformanceLab(..., fork_subprocs=True)` installs preload guards before `sglang_env_config()`."""
        import adapter.conformance_lab as clab_mod
        from adapter import lab as lab_mod

        prev_shared, prev_loops, order = SglangConformanceLab._shared, len(clab_mod._LOOPS), []

        def cleanup():
            SglangConformanceLab._shared = prev_shared
            while len(clab_mod._LOOPS) > prev_loops:
                loop = clab_mod._LOOPS.pop()
                with contextlib.suppress(Exception):
                    loop.close()

        self.addCleanup(cleanup)
        SglangConformanceLab._shared = None
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(lab_mod.SglangLab, "_preload_fork_subprocs", side_effect=lambda: (order.append("preload") or (True, []))),
            mock.patch.object(clab_mod, "sglang_env_config", side_effect=lambda: (order.append("env_config") or {"tp_size": 1})),
            mock.patch.object(lab_mod.SglangLab, "__init__", side_effect=lambda self, *a, **kw: order.append("lab_init")),
        ):
            os.environ.pop("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", None)
            SglangConformanceLab(fork_subprocs=True)
        self.assertEqual(order, ["preload", "env_config", "lab_init"])


    def test_weight_cache_dir_cli_flag_and_conformance_env(self):
        """`--weight-cache-dir` sets `RSIBOX_SGLANG_WEIGHT_CACHE_DIR` and `weight_cache_mode='client'`."""
        from adapter import conformance_lab as clab_mod
        from adapter import lab as lab_mod

        default_env = SglangConformanceLab.launch_environment(["--model", "Qwen/Qwen3-8B"])
        self.assertNotIn("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", default_env)

        wc_env = SglangConformanceLab.launch_environment([
            "--model", "Qwen/Qwen3-8B", "--weight-cache-dir", "/tmp/sgl-wc",
        ])
        self.assertEqual(wc_env["RSIBOX_SGLANG_WEIGHT_CACHE_DIR"], "/tmp/sgl-wc")

        calls = []

        def fake_super_init(self_lab, config, modules, receipts=None, rpc_timeout=30.0, check_timeout=120.0,
                            fork_subprocs=False, weight_cache_dir=None):
            calls.append((dict(config), fork_subprocs, weight_cache_dir))
            self_lab.fork_subprocs = bool(fork_subprocs)
            self_lab.weight_cache_dir = weight_cache_dir
            self_lab.dirty = False
            self_lab._state = "base"

        prev_shared = SglangConformanceLab._shared
        prev_loops = len(clab_mod._LOOPS)

        def cleanup():
            SglangConformanceLab._shared = prev_shared
            while len(clab_mod._LOOPS) > prev_loops:
                loop = clab_mod._LOOPS.pop()
                with contextlib.suppress(Exception):
                    loop.close()

        self.addCleanup(cleanup)
        SglangConformanceLab._shared = None
        with (
            mock.patch.object(lab_mod.SglangLab, "__init__", fake_super_init),
            mock.patch.object(lab_mod.SglangLab, "status", return_value={"state": "base"}),
            mock.patch.dict(os.environ, wc_env, clear=False),
        ):
            SglangConformanceLab()
            # Mismatched weight_cache_dir must miss the cached `_shared` handle.
            SglangConformanceLab._shared = types.SimpleNamespace(
                engine=object(),
                dirty=False,
                status=lambda: {"state": "base"},
                fork_subprocs=False,
                weight_cache_dir="/tmp/other-wc",
                _shutdown_engine=lambda: None,
            )
            SglangConformanceLab(weight_cache_dir="/tmp/sgl-wc-2/")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].get("weight_cache_mode"), "client")
        self.assertEqual(calls[0][2], os.path.abspath("/tmp/sgl-wc"))
        self.assertEqual(calls[1][0].get("weight_cache_mode"), "client")
        self.assertEqual(calls[1][2], os.path.abspath("/tmp/sgl-wc-2"))

        rel_env = SglangConformanceLab.launch_environment([
            "--model", "Qwen/Qwen3-8B", "--weight-cache-dir", "rel-wc/",
        ])
        self.assertEqual(rel_env["RSIBOX_SGLANG_WEIGHT_CACHE_DIR"], os.path.abspath("rel-wc"))

    def test_sglang_lab_weight_cache_fails_closed_when_daemon_missing(self):
        from adapter.lab import SglangLab
        from services import weights as wc

        started = []

        class _StubLab(SglangLab):
            def __init__(self):
                super().__init__({"tp_size": 1}, (), rpc_timeout=1.0, check_timeout=1.0,
                                 fork_subprocs=True, weight_cache_dir="/tmp/missing-wc")

            def _prepare_fork_subprocs(self):
                started.append("fork_subprocs")

            def engine_class(self):
                started.append("engine_class")
                raise AssertionError("engine_class must not run when weight cache is missing")

        fake_probe = {"ok": False, "sock_dir": "/tmp/missing-wc", "missing_ready": ["/tmp/missing-wc/r"]}
        with mock.patch.object(wc, "probe", return_value=fake_probe):
            with self.assertRaisesRegex(RuntimeError, "weight-cache daemon not ready"):
                _StubLab()
        self.assertEqual(started, [])

    def test_sglang_lab_weight_cache_sets_env_before_fork_subprocs_and_restores_on_close(self):
        from multiprocessing.connection import Client
        from adapter.lab import SglangLab
        from services import weights as wc

        events = []

        class _StubEngine:
            run_scheduler_process_func = staticmethod(lambda *a, **k: None)

            def __init__(self, **cfg):
                events.append(("engine", cfg.get("weight_cache_mode"), os.environ.get(wc.SOCKET_ENV)))
                self._conn = Client(cfg["_sock_path"], "AF_UNIX")
                self._conn.send({"rank": 0})

            def shutdown(self):
                with contextlib.suppress(OSError):
                    self._conn.close()

        class _StubLab(SglangLab):
            def __init__(self):
                super().__init__({"tp_size": 1}, (), rpc_timeout=2.0, check_timeout=1.0,
                                 fork_subprocs=True, weight_cache_dir="/tmp/sgl-live-wc")

            def _prepare_fork_subprocs(self):
                events.append(("fork_subprocs", os.environ.get(wc.SOCKET_ENV), os.environ.get(wc.READY_ENV)))
                self.fork_subprocs_fallback = []

            def engine_class(self):
                self.config["_sock_path"] = self.sock_path
                return _StubEngine

        prev_sock = os.environ.pop(wc.SOCKET_ENV, None)
        prev_ready = os.environ.pop(wc.READY_ENV, None)
        self.addCleanup(lambda: prev_sock is not None and os.environ.__setitem__(wc.SOCKET_ENV, prev_sock))
        self.addCleanup(lambda: prev_ready is not None and os.environ.__setitem__(wc.READY_ENV, prev_ready))

        with mock.patch.object(wc, "probe", return_value={"ok": True, "matched_gpus": True}):
            lab = _StubLab()
        try:
            want_env = wc.client_env("/tmp/sgl-live-wc")
            self.assertEqual(
                events[0],
                ("fork_subprocs", want_env[wc.SOCKET_ENV], want_env[wc.READY_ENV]),
            )
            self.assertEqual(events[1], ("engine", "client", want_env[wc.SOCKET_ENV]))
            self.assertNotIn("weight_cache_mode", lab.config)
        finally:
            lab.close()
        self.assertIsNone(os.environ.get(wc.SOCKET_ENV))
        self.assertIsNone(os.environ.get(wc.READY_ENV))

    def test_sglang_lab_prepare_weight_cache_passes_tp_size(self):
        from multiprocessing.connection import Client
        from adapter.lab import SglangLab
        from services import weights as wc

        class _StubEngine:
            run_scheduler_process_func = staticmethod(lambda *a, **k: None)

            def __init__(self, **cfg):
                self._conns = []
                for rank in range(cfg["tp_size"]):
                    c = Client(cfg["_sock_path"], "AF_UNIX")
                    c.send({"rank": rank})
                    self._conns.append(c)

            def shutdown(self):
                for c in self._conns:
                    with contextlib.suppress(OSError):
                        c.close()

        class _StubLab(SglangLab):
            def __init__(self):
                super().__init__({"tp_size": 2}, (), rpc_timeout=2.0, check_timeout=1.0,
                                 weight_cache_dir="/tmp/sgl-tp2-wc/")

            def engine_class(self):
                self.config["_sock_path"] = self.sock_path
                return _StubEngine

        with mock.patch.object(wc, "probe", return_value={"ok": True, "matched_gpus": True}) as m_probe:
            lab = _StubLab()
        try:
            m_probe.assert_called_once_with(os.path.abspath("/tmp/sgl-tp2-wc"), tp_size=2)
        finally:
            lab.close()

    def test_start_engine_failure_cleans_up_fork_env_listener_and_start_method(self):
        import multiprocessing as mp
        import forksafe
        from adapter import lab as lab_mod
        from services import weights as wc
        from services import zygote as sgl_zygote

        orig_state = dict(lab_mod._FORK_SUBPROCS_STATE)
        orig_method = mp.get_start_method(allow_none=True)
        orig_numa = os.environ.get("SGLANG_NUMA_BIND_V2")
        prev_sock = os.environ.pop(wc.SOCKET_ENV, None)
        prev_ready = os.environ.pop(wc.READY_ENV, None)

        def restore_globals():
            lab_mod._FORK_SUBPROCS_STATE.clear()
            lab_mod._FORK_SUBPROCS_STATE.update(orig_state)
            mp.set_start_method(orig_method, force=True)
            if orig_numa is None:
                os.environ.pop("SGLANG_NUMA_BIND_V2", None)
            else:
                os.environ["SGLANG_NUMA_BIND_V2"] = orig_numa
            if prev_sock is None:
                os.environ.pop(wc.SOCKET_ENV, None)
            else:
                os.environ[wc.SOCKET_ENV] = prev_sock
            if prev_ready is None:
                os.environ.pop(wc.READY_ENV, None)
            else:
                os.environ[wc.READY_ENV] = prev_ready

        self.addCleanup(restore_globals)
        lab_mod._FORK_SUBPROCS_STATE.update({"preloaded": False, "clear_nvml_env": False})
        os.environ.pop("SGLANG_NUMA_BIND_V2", None)
        mp.set_start_method(None, force=True)

        created_socks = []

        class _BoomEngine:
            run_scheduler_process_func = staticmethod(lambda *a, **k: None)

            def __init__(self, **_cfg):
                raise RuntimeError("engine boom")

        class _FailLab(lab_mod.SglangLab):
            def __init__(self):
                super().__init__({"tp_size": 1}, (), rpc_timeout=1.0, check_timeout=1.0,
                                 fork_subprocs=True, weight_cache_dir="/tmp/sgl-fail-wc")

            def engine_class(self):
                created_socks.append(self.sock_path)
                return _BoomEngine

        with (
            mock.patch.object(wc, "probe", return_value={"ok": True, "matched_gpus": True}),
            mock.patch.object(forksafe, "poison_reasons", return_value=[]),
            mock.patch.object(forksafe, "cuda_initialized", return_value=False),
            mock.patch.object(sgl_zygote, "install_guards", return_value=({"nvml_env_set": False}, lambda: None)),
            mock.patch.object(sgl_zygote, "preload", return_value={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "engine boom"):
                _FailLab()

        # Start method restored back to None, mp unpatched, env vars restored, socket unlinked.
        self.assertIsNone(mp.get_start_method(allow_none=True))
        self.assertNotIn("get_context", mp.context._default_context.__dict__)
        self.assertIsNone(os.environ.get("SGLANG_NUMA_BIND_V2"))
        self.assertIsNone(os.environ.get(wc.SOCKET_ENV))
        self.assertIsNone(os.environ.get(wc.READY_ENV))
        self.assertEqual(len(created_socks), 1)
        self.assertFalse(os.path.exists(created_socks[0]))

        # `_prepare_fork_subprocs()` routes via `set_start_method("spawn", force=True)` so early CUDA
        # after `_prepare_fork_subprocs()` falls back to `spawn` on `_default_context.get_context()`.
        cuda_flag = {"init": False}
        fallback_buf = io.StringIO()
        lab_dummy = lab_mod.SglangLab.__new__(lab_mod.SglangLab)
        lab_dummy._fork_subprocs_restore = None
        lab_dummy.fork_subprocs_fallback = []
        with (
            mock.patch.object(forksafe, "poison_reasons", return_value=[]),
            mock.patch.object(forksafe, "cuda_initialized", side_effect=lambda: cuda_flag["init"]),
            contextlib.redirect_stdout(fallback_buf),
        ):
            self.assertTrue(lab_dummy._prepare_fork_subprocs())
            try:
                self.assertEqual(mp.get_start_method(), "fork")
                cuda_flag["init"] = True
                self.assertEqual(mp.context._default_context.get_context().get_start_method(), "spawn")
            finally:
                lab_dummy._fork_subprocs_restore()
                lab_dummy._fork_subprocs_restore = None
        self.assertIsNone(mp.get_start_method(allow_none=True))
        self.assertIn("fork_subprocs_fallback", fallback_buf.getvalue())

        # `_preload_fork_subprocs()` unpatches child + clears NVML env on SystemExit fallback.
        lab_mod._FORK_SUBPROCS_STATE.update({"preloaded": False, "clear_nvml_env": False})
        with (
            mock.patch.object(forksafe, "poison_reasons", side_effect=[[], ["cuda poisoned"]]),
            mock.patch.object(sgl_zygote, "install_guards", return_value=({"nvml_env_set": True}, lambda: None)),
            mock.patch.object(sgl_zygote, "preload", side_effect=SystemExit("initialized CUDA")),
            mock.patch.object(forksafe, "unpatch_child") as m_unpatch,
        ):
            ok, reasons = lab_mod.SglangLab._preload_fork_subprocs()
        self.assertFalse(ok)
        self.assertEqual(reasons, ["cuda poisoned"])
        m_unpatch.assert_called_once_with(None, clear_env=True)

    def test_conformance_lab_clears_shared_on_replacement_failure_and_requires_live_engine(self):
        from adapter import conformance_lab as clab_mod
        from adapter import lab as lab_mod

        prev_shared = SglangConformanceLab._shared
        prev_loops = len(clab_mod._LOOPS)

        def cleanup():
            SglangConformanceLab._shared = prev_shared
            while len(clab_mod._LOOPS) > prev_loops:
                loop = clab_mod._LOOPS.pop()
                with contextlib.suppress(Exception):
                    loop.close()

        self.addCleanup(cleanup)

        # Replacement init failure must leave `_shared = None`, never pointing to a shut-down lab.
        shutdown_calls = []
        SglangConformanceLab._shared = types.SimpleNamespace(
            engine=object(),
            dirty=False,
            status=lambda: {"state": "base"},
            fork_subprocs=False,
            weight_cache_dir=None,
            _shutdown_engine=lambda: shutdown_calls.append("shutdown"),
        )
        with (
            mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m"}, clear=False),
            mock.patch.object(lab_mod.SglangLab, "_preload_fork_subprocs", return_value=(True, [])),
            mock.patch.object(lab_mod.SglangLab, "__init__", side_effect=RuntimeError("daemon missing")),
        ):
            os.environ.pop("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", None)
            with self.assertRaisesRegex(RuntimeError, "daemon missing"):
                SglangConformanceLab(fork_subprocs=True)
        self.assertEqual(shutdown_calls, ["shutdown"])
        self.assertIsNone(SglangConformanceLab._shared)

        # Dead `_shared` with `engine=None` must not be reused even if flags match.
        reinit_calls = []

        def fake_init(self_lab, config, modules, receipts=None, rpc_timeout=30.0, check_timeout=120.0,
                      fork_subprocs=False, weight_cache_dir=None):
            reinit_calls.append(True)
            self_lab.engine = object()
            self_lab.dirty = False
            self_lab.fork_subprocs = fork_subprocs
            self_lab.weight_cache_dir = weight_cache_dir

        SglangConformanceLab._shared = types.SimpleNamespace(
            engine=None,
            dirty=False,
            status=lambda: {"state": "base"},
            fork_subprocs=False,
            weight_cache_dir=None,
            _shutdown_engine=lambda: shutdown_calls.append("dead_shutdown"),
        )
        with (
            mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m"}, clear=False),
            mock.patch.object(lab_mod.SglangLab, "__init__", fake_init),
        ):
            os.environ.pop("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", None)
            os.environ.pop("RSIBOX_SGLANG_FORK_SUBPROCS", None)
            lab = SglangConformanceLab()
        self.assertEqual(reinit_calls, [True])
        self.assertIs(SglangConformanceLab._shared, lab)

    def test_sglang_env_config_handles_pep660_editable_install_via_find_spec(self):
        import importlib.abc
        import importlib.machinery

        with tempfile.TemporaryDirectory() as tmp:
            pkg_root = os.path.join(tmp, "editable_src", "sglang")
            os.makedirs(os.path.join(pkg_root, "srt"), exist_ok=True)
            with open(os.path.join(pkg_root, "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write("raise AssertionError('PEP 660 sglang package must not be imported')\n")
            with open(os.path.join(pkg_root, "srt", "server_args.py"), "w", encoding="utf-8") as fh:
                fh.write("class ServerArgs:\n    cuda_graph_max_bs_decode: int = -1\n")

            class _EditableFinder(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "sglang":
                        spec = importlib.machinery.ModuleSpec("sglang", None, is_package=True)
                        spec.submodule_search_locations = [pkg_root]
                        return spec
                    return None

            finder = _EditableFinder()
            with (
                mock.patch.dict(os.environ, {"RSIBOX_SGLANG_MODEL": "/m", "RSIBOX_SGLANG_MAX_BS": "4"}, clear=False),
                mock.patch.dict(sys.modules),
                mock.patch.object(sys, "meta_path", [finder, *sys.meta_path]),
            ):
                for key in list(sys.modules):
                    if key == "sglang" or key.startswith("sglang."):
                        sys.modules.pop(key, None)
                cfg = sglang_env_config()
                self.assertEqual(cfg["cuda_graph_max_bs_decode"], 4)
                self.assertTrue(cfg["disable_prefill_cuda_graph"])
                self.assertNotIn("sglang", sys.modules)


if __name__ == "__main__":
    unittest.main()
