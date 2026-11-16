"""CPU unit tests for `adapter.conformance_lab` (no vLLM, no GPU).

Runs the full 11-case conformance suite (`core/livelab/conformance`) against `VllmLab` backed
by a real `adapter.host.Host` over in-process `Side` endpoints (one worker rank and one engine
core, each with its own synthetic copies of the 4 vLLM modules `VllmLab` edits), plus unit
tests for farm construction and fault/timeout handling.
"""
import importlib.util
import itertools
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from conformance import cases
from . import conformance_lab as CL
from . import host as H
from .side import Side

FAKE_SAMPLER = '''\
class _States:
    def max_num_logprobs(self, idx_mapping_np):
        return 0

    def get_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np, vocab):
        return None, None, False


class Sampler:
    def __init__(self):
        self.sampling_states = _States()

    def get_logprobs_dims(self, idx_mapping_np):
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        return max_num_logprobs

    def sample(self, logits, idx_mapping_np, return_logprobs=False):
        top_k, top_p, has_topp_only_rows = self.sampling_states.get_top_k_top_p(
            None, idx_mapping_np, len(logits)
        )
        best = max(range(len(logits)), key=lambda i: logits[i])
        return best, logits

    def __call__(self, logits, idx_mapping_np):
        _ = self.get_logprobs_dims(idx_mapping_np)
        sampled, processed_logits = self.sample(
            logits, idx_mapping_np, return_logprobs=False
        )
        # These are GPU tensors.
        sampler_output = sampled
        return sampler_output
'''

FAKE_SCHED_UTILS = '''\
def check_stop(request, max_model_len):
    last_token_id = request.output_token_ids[-1]
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        return True
    return False
'''

FAKE_SCHEDULER = '''\
def init_logger(name):
    return name


class Scheduler:
    def __init__(self):
        self.deferred_waiting = set()

    def schedule(self, queue, token_budget=8):
        if queue is not None:
            def skip_request(from_queue):
                request_to_skip = from_queue.pop()
                self.deferred_waiting.add(request_to_skip)
        return token_budget
'''

FAKE_LOGITS_PROC = '''\
def init_logger(name):
    return name


logger = init_logger(__name__)


class _Tensor(list):
    def __neg__(self):
        return _Tensor([-x for x in self])

    def roll(self, shift, dims=-1):
        shift = shift % len(self)
        return _Tensor(self[-shift:] + self[:-shift])


class LogitsProcessor:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, base_logits):
        return _Tensor([x * self.scale for x in base_logits])
'''

FAKE_SOURCES = {
    CL.SAMPLER: FAKE_SAMPLER,
    CL.SCHED_UTILS: FAKE_SCHED_UTILS,
    CL.SCHEDULER: FAKE_SCHEDULER,
    CL.LOGITS_PROC: FAKE_LOGITS_PROC,
}
_counter = itertools.count(1)


def _load_fake_tree(prefix, overrides=None, only=None):
    """Load isolated copies of `only` (default all 4) modules under `<prefix>.<module>`."""
    sources = {k: v for k, v in {**FAKE_SOURCES, **(overrides or {})}.items() if only is None or k in only}
    root = tempfile.mkdtemp(prefix=f"fastrsi-fake-{prefix}-")
    loaded = []
    for mod_name, text in sources.items():
        aliased = f"{prefix}.{mod_name.split('.', 1)[1]}"
        parts = aliased.split(".")
        for idx in range(1, len(parts)):
            pkg = ".".join(parts[:idx])
            if pkg not in sys.modules:
                pkg_mod = types.ModuleType(pkg)
                pkg_mod.__path__ = [os.path.join(root, *parts[:idx])]
                sys.modules[pkg] = pkg_mod
                loaded.append(pkg)
        file_path = os.path.join(root, *parts) + ".py"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as fh:
            fh.write(text)
        spec = importlib.util.spec_from_file_location(aliased, file_path)
        module_obj = importlib.util.module_from_spec(spec)
        sys.modules[aliased] = module_obj
        spec.loader.exec_module(module_obj)
        loaded.append(aliased)
    return root, loaded


def _run_fake_step(worker_prefix, core_prefix, sampler_obj, lproc_obj, prompts, max_tokens):
    sched_utils = sys.modules[f"{core_prefix}.v1.core.sched.utils"]
    vocab = [" paris", " two", " 100C", " alpha", " beta", " gamma"]
    outputs = []
    for p_idx, prompt in enumerate(prompts):
        req = NS(output_token_ids=[], num_tokens=len(prompt.split()), num_output_tokens=0, max_tokens=max_tokens)
        pieces = []
        for step in range(max_tokens):
            raw = [float((idx - (step + p_idx) % 4) ** 2) + idx * 0.1 for idx in range(len(vocab))]
            logits = lproc_obj(raw)
            tok = sampler_obj(logits, [0])
            req.output_token_ids.append(tok)
            req.num_tokens += 1
            req.num_output_tokens += 1
            pieces.append(vocab[tok % len(vocab)])
            if sched_utils.check_stop(req, 2048):
                break
        outputs.append("".join(pieces))
    return outputs


class FakeExecutor:
    def __init__(self, endpoints, lost_flag):
        self.endpoints = endpoints
        self.lost_flag = lost_flag

    def collective_rpc(self, _name, timeout=None, args=()):
        if self.lost_flag[0]:
            raise ConnectionError("worker process exited")
        method, payload = args
        return [ep.dispatch(method, payload) for ep in self.endpoints]


class FakeBackend:
    """In-process stand-in for the vLLM server + control channel using the real `Host`."""

    def __init__(self):
        self._roots, self._loaded = [], []
        self.respawn()

    def respawn(self):
        self._cleanup_current()
        tag = next(_counter)
        self.w_prefix, self.c_prefix = f"fvw{tag}", f"fvc{tag}"
        w_root, w_mods = _load_fake_tree(self.w_prefix, only=CL.WORKER_MODULES)
        c_root, c_mods = _load_fake_tree(self.c_prefix, only=CL.CORE_MODULES)
        self.host_dir = tempfile.mkdtemp(prefix="fastrsi-fake-host-")
        self._roots.extend([w_root, c_root, self.host_dir])
        self._loaded.extend(w_mods + c_mods)

        self.lost = [False]
        samp_cls = sys.modules[f"{self.w_prefix}.v1.worker.gpu.sample.sampler"].Sampler
        lp_cls = sys.modules[f"{self.w_prefix}.model_executor.layers.logits_processor"].LogitsProcessor
        self.sampler = samp_cls()
        self.lproc = lp_cls(scale=1.0)
        self.cache_hits = 0

        worker_obj = NS(execute_model=lambda *a, **k: True)
        self.worker_obj = worker_obj
        self.worker_side = Side("worker", scope={"worker": worker_obj}, rank=0, alias={"vllm": self.w_prefix})
        core_obj = NS(
            model_executor=FakeExecutor([self.worker_side], self.lost),
            reset_prefix_cache=self._reset_cache,
        )
        with mock.patch.dict(sys.modules, {"vllm": types.SimpleNamespace(__version__="0.33.0-fake")}):
            self.host = H.Host(core_obj, self.host_dir)
            self.host.core_side = Side("core", scope={"core": core_obj}, alias={"vllm": self.c_prefix})
            from kernel import coordinator as C
            self.host.co.participants[1] = C.CallParticipant("core", self.host.core_side.dispatch, bounded=True)

    def _reset_cache(self):
        self.cache_hits = 0
        return True

    def reset_state(self):
        self._reset_cache()

    def ctl(self, cmd, timeout=5.0):
        if cmd.get("op") == "txn":
            for rep in (cmd.get("plan") or {}).get("repairs") or ():
                if "os._exit" in rep.get("cell", ""):
                    self.lost[0] = True
                    return {"ok": True, "outcome": "committed", "txn": cmd["plan"]["id"]}
        if self.lost[0] and cmd.get("op") in ("ping", "fingerprint", "prove", "txn"):
            return {"ok": False, "lost": True, "error": "worker lost"}
        return self.host.handle(cmd)

    def complete(self, prompts, max_tokens, timeout=5.0):
        box = {}

        def run():
            try:
                self.worker_obj.execute_model()
                box["out"] = _run_fake_step(
                    self.w_prefix, self.c_prefix, self.sampler, self.lproc, prompts, max_tokens
                )
            except BaseException as err:
                box["err"] = err

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise TimeoutError(f"timed out after {timeout} s")
        if "err" in box:
            raise box["err"]
        return {"worker0": box["out"]}

    def fresh_check(self, patch, workload=None):
        spec = workload or CL.WORKLOAD
        prompts = spec.get("prompts", CL.PROMPTS)
        max_tokens = int(spec.get("max_tokens", CL.WORKLOAD["max_tokens"]))
        tag = next(_counter)
        fw, fc = f"fvfreshw{tag}", f"fvfreshc{tag}"
        files = (patch or {}).get("files") or {}
        w_root, w_mods = _load_fake_tree(fw, files, only=CL.WORKER_MODULES)
        c_root, c_mods = _load_fake_tree(fc, files, only=CL.CORE_MODULES)
        try:
            samp = sys.modules[f"{fw}.v1.worker.gpu.sample.sampler"].Sampler()
            lproc = sys.modules[f"{fw}.model_executor.layers.logits_processor"].LogitsProcessor(scale=1.0)
            out = _run_fake_step(fw, fc, samp, lproc, prompts, max_tokens)
            return {"ok": True, "output": {"worker0": out}}
        except Exception as err:
            return {"ok": False, "error": str(err), "output": None}
        finally:
            for m_name in w_mods + c_mods:
                sys.modules.pop(m_name, None)
            shutil.rmtree(w_root, ignore_errors=True)
            shutil.rmtree(c_root, ignore_errors=True)

    def _cleanup_current(self):
        for m_name in self._loaded:
            sys.modules.pop(m_name, None)
        self._loaded.clear()
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)
        self._roots.clear()

    def close(self):
        self._cleanup_current()


def make_cpu_lab(check_timeout=0.4):
    return CL.VllmLab(check_timeout=check_timeout, backend=FakeBackend(), base_texts=FAKE_SOURCES)


class ConformanceOnFakeVllmTest(unittest.TestCase):
    """Every conformance case (c0-c10) passes on `VllmLab` over `Host` + `Side`."""

    def test_all_11_conformance_cases_pass(self):
        for name in cases.CASES:
            with self.subTest(case=name):
                verdict = cases.run(name, make_cpu_lab)
                self.assertEqual(verdict["result"], "pass", f"{name}: {verdict}")


class FarmAndHelpersTest(unittest.TestCase):

    def test_build_farm_copies_py_and_symlinks_other_files(self):
        src = tempfile.mkdtemp()
        dst = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, src, True)
        self.addCleanup(shutil.rmtree, dst, True)
        os.makedirs(os.path.join(src, "sub", "__pycache__"))
        with open(os.path.join(src, "mod_a.py"), "w") as fh:
            fh.write("X = 1\n")
        with open(os.path.join(src, "sub", "native.so"), "w") as fh:
            fh.write("bin")
        with open(os.path.join(src, "sub", "__pycache__", "mod_a.cpython-312.pyc"), "w") as fh:
            fh.write("pyc")

        CL.build_farm(src, dst)
        self.assertFalse(os.path.islink(os.path.join(dst, "mod_a.py")))
        self.assertTrue(os.path.islink(os.path.join(dst, "sub", "native.so")))
        self.assertFalse(os.path.exists(os.path.join(dst, "sub", "__pycache__")))

    def test_unknown_fault_returns_not_ok_without_raising(self):
        lab = make_cpu_lab()
        self.addCleanup(lab.close)
        reply = lab.inject("nonexistent_fault")
        self.assertFalse(reply["ok"])
        self.assertIn("unknown fault", reply["error"])

    def test_replay_stack_preserves_accept_list(self):
        lab = make_cpu_lab()
        self.addCleanup(lab.close)
        unreached = lab.patches()["unreached"]
        self.assertTrue(lab.install(unreached)["ok"])
        saved = [dict(x) for x in lab._stack_files]
        self.assertEqual(saved[0]["accept"], ["constructor_edit"])
        self.assertTrue(lab.reset("respawn")["ok"])
        for entry in saved:
            self.assertTrue(lab.install(entry)["ok"])
        self.assertTrue(lab.uninstall()["ok"])


if __name__ == "__main__":
    unittest.main()
