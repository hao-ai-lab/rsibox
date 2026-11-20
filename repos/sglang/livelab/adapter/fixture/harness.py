"""Conformance-only harness helpers (`fresh_check`, `inject`, and named patch suites)."""
import json
import os

from fastrsi import harness as H
from fastrsi.harness import edit_once

GREEDY = "batch_next_token_ids = torch.argmax(logits, -1)"


class ConformanceHarness:
    """Provides `fresh_check` and `inject` for `core/livelab/conformance` (`LabUnderTest`)."""

    package = "sglang"
    fresh_module = "adapter.fixture.fresh"
    fresh_mode = "real"

    def fresh_check(self, patch, workload=None):
        livelab_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            files = self.files(patch, against="disk") if hasattr(self, "files") else (patch or {}).get("files")
        except ValueError as err:
            return {"ok": False, "output": None, "error": str(err)}
        spec = {
            "mode": self.fresh_mode,
            "config": self.config,
            "workload": workload or self.default_workload(),
        }
        with H.PackageOverlay(self.package, files) as overlay:
            return H.run_fresh_subprocess(
                self.fresh_module,
                spec,
                overlay.root,
                extra_pythonpath=(livelab_dir,),
                timeout=self.check_timeout + 300,
            )

    def inject(self, fault, **kwargs):
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 5.0)
        rank = kwargs.pop("participant", self.ranks - 1)
        rank = int(rank.replace("rank", "")) if isinstance(rank, str) else rank
        with self.lock:
            try:
                return self.rank_call(rank, "inject", {"fault": fault, **kwargs})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def toy_patches(base_fn, sampler_mod, logits_mod, norm_mod, runner_mod):
    sampler_src, logits_src = base_fn(sampler_mod), base_fn(logits_mod)
    norm_src, runner_src = base_fn(norm_mod), base_fn(runner_mod)
    norm_v1 = edit_once(norm_src, "(h * 7 + 1) % MODULUS", "(h * 11 + 1) % MODULUS")
    norm_v2 = edit_once(norm_v1, "(h * 11 + 1) % MODULUS", "(h * 13 + 1) % MODULUS")
    sampler_v3 = edit_once(sampler_src, "    return best\n", "    return (best + 1) % len(row)\n")
    multi_sampler = edit_once(sampler_v3, "if temperature < TEMPERATURE_EPS:",
                              "if float(temperature) < TEMPERATURE_EPS:")
    multi_logits = edit_once(logits_src, "(v + 1 + vocab_offset)", "(v + 2 + vocab_offset)")
    wrap = lambda m, text: {"files": {m: text}}
    return {
        "noop": wrap(sampler_mod, edit_once(sampler_src, "    best = 0\n", "    best = int(0)\n")),
        "visible": wrap(norm_mod, norm_v1),
        "chain": [wrap(norm_mod, norm_v1), wrap(norm_mod, norm_v2), wrap(sampler_mod, sampler_v3)],
        "multi": {"files": {sampler_mod: multi_sampler, logits_mod: multi_logits}},
        "unreached": wrap(runner_mod, edit_once(runner_src, "self.vocab_offset = 0", "self.vocab_offset = 3")),
        "refused": {
            "syntax": wrap(sampler_mod, edit_once(sampler_src, "    return best\n", "    return (\n")),
            "closure": wrap(sampler_mod, edit_once(sampler_src, "return index < k", "return index < k + limit")),
            "module_state": wrap(sampler_mod, edit_once(sampler_src, "TEMPERATURE_EPS = 1e-6", "TEMPERATURE_EPS = 1e-5")),
            "empty": wrap(sampler_mod, edit_once(sampler_src, "def argmax(row):", "def argmax(row):  # greedy")),
        },
    }


def sglang_env_config():
    model_path = os.environ.get("FASTRSI_SGLANG_MODEL")
    if not model_path:
        raise RuntimeError("set FASTRSI_SGLANG_MODEL to the model path or repo id")
    max_bs = int(os.environ.get("FASTRSI_SGLANG_MAX_BS", "4"))
    graph_kw = {"cuda_graph_max_bs": max_bs}
    try:
        from sglang.srt.server_args import ServerArgs
        fields = set(getattr(ServerArgs, "__dataclass_fields__", ())) | set(getattr(ServerArgs, "__struct_fields__", ()))
        if "cuda_graph_max_bs_decode" in fields or hasattr(ServerArgs, "cuda_graph_max_bs_decode"):
            graph_kw = {"cuda_graph_max_bs_decode": max_bs, "disable_prefill_cuda_graph": True}
    except Exception:
        pass
    return {
        "model_path": model_path,
        "tp_size": int(os.environ.get("FASTRSI_SGLANG_TP", "1")),
        "mem_fraction_static": float(os.environ.get("FASTRSI_SGLANG_MEM_FRACTION", "0.25")),
        **graph_kw,
        "random_seed": 0,
        "log_level": "error",
        **json.loads(os.environ.get("FASTRSI_SGLANG_EXTRA_ARGS") or "{}"),
    }


def toy_config(tp_size, kwargs):
    return {
        "tp_size": tp_size, "vocab_size": 64, "random_seed": 0, "max_running_requests": 8,
        "cuda_graph_max_bs": 4, "disable_cuda_graph": False,
        "disable_overlap_schedule": kwargs.pop("disable_overlap_schedule", False),
        "disable_radix_cache": kwargs.pop("disable_radix_cache", False),
    }


def sglang_patches(base_fn, sampler_mod, worker_mod):
    sampler_src, worker_src = base_fn(sampler_mod), base_fn(worker_mod)
    vis_1 = edit_once(sampler_src, GREEDY,
                      "batch_next_token_ids = (torch.argmax(logits, -1) + 1) % logits.shape[-1]")
    vis_2 = edit_once(sampler_src, GREEDY,
                      "batch_next_token_ids = (torch.argmax(logits, -1) + 2) % logits.shape[-1]")
    vis_3 = edit_once(vis_2, "        return logits\n", "        return logits.flip(-1)\n")
    multi_sampler = edit_once(vis_1, "        return logits\n", "        return logits.clone()\n")
    multi_worker = edit_once(
        worker_src, "    def set_hicache_consumer(self, consumer_index: int):\n",
        "    def set_hicache_consumer(self, consumer_index: int):\n        consumer_index = int(consumer_index)\n",
    )
    unreached = edit_once(
        worker_src, "    def _init_dllm_algorithm(self):\n",
        "    def _init_dllm_algorithm(self):\n"
        "        for param in self.model_runner.model.parameters():\n"
        "            param.data.neg_()\n            break\n",
    )
    wrap = lambda m, text: {"files": {m: text}}
    return {
        "noop": wrap(sampler_mod, edit_once(sampler_src, GREEDY, GREEDY + ".clone()")),
        "visible": wrap(sampler_mod, vis_1),
        "chain": [wrap(sampler_mod, vis_1), wrap(sampler_mod, vis_2), wrap(sampler_mod, vis_3)],
        "multi": {"files": {sampler_mod: multi_sampler, worker_mod: multi_worker}},
        "unreached": wrap(worker_mod, unreached),
        "refused": {
            "syntax": wrap(sampler_mod, edit_once(sampler_src, GREEDY, "batch_next_token_ids = (")),
            "closure": wrap(sampler_mod, edit_once(
                sampler_src, "if any(x > 0 for x in top_logprobs_nums):",
                "if any(x > 0 and sampling_info is not None for x in top_logprobs_nums):")
                if "if any(x > 0 for x in top_logprobs_nums):" in sampler_src else edit_once(
                sampler_src, 'details = " ".join(f"{key}={value}" for key, value in fields.items())',
                'details = " ".join(f"{stage}:{key}={value}" for key, value in fields.items())')),
            "module_state": wrap(sampler_mod, edit_once(
                sampler_src, '_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch",',
                '_BUILT_IN_SAMPLING_BACKENDS = {"pytorch",')
                if '_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch",' in sampler_src else edit_once(
                sampler_src, 'SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")',
                'SGLANG_RETURN_ORIGINAL_LOGPROB = bool(get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB"))')),
            "empty": wrap(sampler_mod, edit_once(sampler_src, GREEDY, GREEDY + "  # greedy")),
        },
    }
