"""CPU unit tests for `adapter.conformance_lab` (no vLLM, no GPU).

Runs the full 11-case conformance suite (`core/livelab/conformance`) against `VllmLab` backed
by a real `adapter.host.Host` over in-process `Side` endpoints (one worker rank and one engine
core, each with its own synthetic copies of the 4 vLLM modules `VllmLab` edits), plus unit
tests for farm construction and fault/timeout handling.
"""
import difflib
import importlib.util
import itertools
import os
import shutil
import signal
import subprocess
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
    root = tempfile.mkdtemp(prefix=f"rsibox-fake-{prefix}-")
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
        self.host_dir = tempfile.mkdtemp(prefix="rsibox-fake-host-")
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
            vllm_config=NS(parallel_config=NS(world_size=1)),
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

    def test_single_worker_host_cannot_satisfy_tp2(self):
        lab = make_cpu_lab()
        self.addCleanup(lab.close)
        ping = lab._ctl({"op": "ping"})
        CL._check_topology(ping, 1)
        with self.assertRaisesRegex(RuntimeError, "requested TP=2"):
            CL._check_topology(ping, 2)

    @unittest.skipUnless(os.path.isdir("/proc"), "requires Linux process identity")
    def test_stop_reclaims_marked_worker_after_server_exit(self):
        with tempfile.TemporaryDirectory() as marker:
            with subprocess.Popen(
                [sys.executable, "-c", "import os, subprocess, sys; "
                 "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
                 "stdout=subprocess.DEVNULL); print(p.pid, flush=True); input(); os._exit(0)"],
                env={**os.environ, "RSIBOX_LIVE_RECONFIG_DIR": marker},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True,
            ) as leader:
                worker = int(leader.stdout.readline())
                try:
                    srv = CL._ServerHandle(leader.pid, 0, marker, 0, "", marker=marker)
                    leader.communicate("\n", timeout=5)
                    self.assertFalse(srv.alive())
                    self.assertTrue(CL._alive(worker))
                    srv.stop()
                    self.assertFalse(CL._alive(worker))
                finally:
                    if CL._alive(worker):
                        os.kill(worker, signal.SIGKILL)
                    if leader.poll() is None:
                        leader.kill()
                        leader.communicate()

    def test_original_edit_id_can_undo_after_fresh_check_pauses_live_server(self):
        lab = make_cpu_lab()
        backend = lab._backend
        self.addCleanup(backend.close)
        edit = lab.patches()["visible"]
        original = lab.install(edit)["id"]
        # Keep the real session, replacement host and participants. Only process startup
        # and HTTP are replaced with the fixture's independently loaded modules.
        srv = NS(alive=lambda: True, stop=lambda: None, port=8111)
        def start_server(port, tag, use_farm):
            if tag == "live":
                backend.respawn()
            return srv
        lab.server, lab.port, lab.share_gpu = srv, srv.port, True
        lab.shared, lab.server_key, lab._owns_server = True, ("fake-key", 8111), False
        with (mock.patch.object(lab, "_backend", None),
              mock.patch.object(lab, "_ctl", backend.ctl),
              mock.patch.object(lab, "_start_server", start_server),
              mock.patch.object(lab, "_write_farm"),
              mock.patch.object(lab, "_run_http", return_value=backend.fresh_check(edit)["output"]),
              mock.patch.dict(CL.VllmLab._shared_servers, {lab.server_key: srv}, clear=True)):
            self.assertTrue(lab.fresh_check(edit)["ok"])
            self.assertEqual(lab.status()["txns"], [original])
            self.assertTrue(lab.uninstall(original)["ok"])
            self.assertEqual(lab.status()["state"], "base")

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

    def test_launch_environment_maps_flags_and_trailing_serve_args(self):
        env = CL.VllmLab.launch_environment([
            "--model", "Qwen/Qwen3-8B",
            "--venv", "rel-venv",
            "--work", "rel-work",
            "--port", "19000",
            "--tp", "2",
            "--check-timeout", "45.5",
            "--no-share-gpu",
            "--extra", "--seed 0 --enforce-eager",
            "--", "--max-model-len", "1024",
        ])
        self.assertEqual(env, {
            "RSIBOX_VLLM_MODEL": "Qwen/Qwen3-8B",
            "RSIBOX_VLLM_VENV": os.path.abspath("rel-venv"),
            "RSIBOX_VLLM_WORK": os.path.abspath("rel-work"),
            "RSIBOX_VLLM_PORT": "19000",
            "RSIBOX_VLLM_TP": "2",
            "RSIBOX_VLLM_CHECK_TIMEOUT": "45.5",
            "RSIBOX_VLLM_SHARE_GPU": "0",
            "RSIBOX_VLLM_EXTRA": "--seed 0 --enforce-eager --max-model-len 1024",
        })
        minimal = CL.VllmLab.launch_environment(["--model", "/models/qwen"])
        self.assertEqual(minimal, {"RSIBOX_VLLM_MODEL": "/models/qwen"})
        self.assertNotIn("RSIBOX_VLLM_TP", minimal)
        self.assertNotIn("RSIBOX_VLLM_EXTRA", minimal)
        only_tail = CL.VllmLab.launch_environment(["--model", "/models/qwen", "--", "--enforce-eager"])
        self.assertEqual(only_tail["RSIBOX_VLLM_EXTRA"], "--enforce-eager")

        # `rsibox conformance vllm --real -- --model ...` forwards a leading "--".
        leading = CL.VllmLab.launch_environment(["--", "--model", "/models/qwen", "--tp", "2"])
        self.assertEqual(leading, {"RSIBOX_VLLM_MODEL": "/models/qwen", "RSIBOX_VLLM_TP": "2"})

        leading_with_tail = CL.VllmLab.launch_environment(
            ["--", "--model=/models/qwen", "--", "--enforce-eager", "--max-model-len", "1024"]
        )
        self.assertEqual(leading_with_tail, {
            "RSIBOX_VLLM_MODEL": "/models/qwen",
            "RSIBOX_VLLM_EXTRA": "--enforce-eager --max-model-len 1024",
        })

        # Preset RSIBOX_VLLM_MODEL + raw serve flags behind a single leading "--".
        raw_tail_only = CL.VllmLab.launch_environment(["--", "--enforce-eager", "--max-model-len", "1024"])
        self.assertEqual(raw_tail_only, {"RSIBOX_VLLM_EXTRA": "--enforce-eager --max-model-len 1024"})

    def test_install_and_fresh_check_accept_stacked_unified_diffs_and_refuse_stale_diff(self):
        lab = make_cpu_lab()
        self.addCleanup(lab.close)
        base_out = lab.check()["output"]
        sampler_path = CL.SAMPLER.replace(".", "/") + ".py"
        anchor = "        sampled, processed_logits = self.sample(\n"
        v1 = CL._edit(FAKE_SAMPLER, anchor, "        logits = -logits\n" + anchor)
        v2 = CL._edit(v1, "        logits = -logits\n", "        logits = logits.roll(1, dims=-1)\n")

        def make_diff(old_src, new_src):
            return "".join(difflib.unified_diff(
                old_src.splitlines(True), new_src.splitlines(True),
                fromfile=f"a/{sampler_path}", tofile=f"b/{sampler_path}",
            ))

        diff1 = make_diff(FAKE_SAMPLER, v1)
        diff2 = make_diff(v1, v2)
        combined_diff = make_diff(FAKE_SAMPLER, v2)

        r1 = lab.install({"diff": diff1})
        self.assertTrue(r1["ok"], r1)
        out1 = lab.check()["output"]
        self.assertNotEqual(out1, base_out)

        # Stacking diff2 applies against live v1 text; re-applying diff1 is refused cleanly.
        stale = lab.install({"diff": diff1})
        self.assertEqual((stale["ok"], stale["outcome"], stale["codes"]), (False, "refused", ["patch_invalid"]))
        r2 = lab.install({"diff": diff2})
        self.assertTrue(r2["ok"], r2)
        out2 = lab.check()["output"]
        self.assertNotEqual(out2, out1)

        # fresh_check resolves unified diff against base/disk text and matches live stacked output.
        fresh = lab.fresh_check({"diff": combined_diff})
        self.assertTrue(fresh["ok"], fresh)
        self.assertEqual(fresh["output"], out2)
        bad_fresh = lab.fresh_check({"diff": diff2})
        self.assertFalse(bad_fresh["ok"])

        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual((lab.check()["output"], lab.status()["state"]), (out1, "edited"))
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual((lab.check()["output"], lab.status()["state"]), (base_out, "base"))


class VllmLabMultiInstanceAndCloseTest(unittest.TestCase):
    """Verify key-indexed `_shared_servers`, `shared=False` ownership, and `close()` teardown."""

    def setUp(self):
        with CL.VllmLab._shared_lock:
            self._saved_servers = dict(CL.VllmLab._shared_servers)
            CL.VllmLab._shared_servers.clear()

    def tearDown(self):
        CL._cleanup_shared()
        with CL.VllmLab._shared_lock:
            CL.VllmLab._shared_servers.update(self._saved_servers)

    @staticmethod
    def _make_fake_site(root):
        site = os.path.join(root, "site")
        for mod_name, src_text in FAKE_SOURCES.items():
            path = os.path.join(site, mod_name.replace(".", os.sep) + ".py")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(src_text)
        return site

    @staticmethod
    def _make_stub_start(started_handles):
        def _start(lab_self, port, tag, use_farm):
            state = {"alive": True, "stops": 0}

            def _stop():
                state["stops"] += 1
                state["alive"] = False

            handle = NS(
                port=port,
                key=lab_self.server_key,
                state=state,
                alive=lambda: state["alive"],
                stop=_stop,
                start_s=0.01,
            )
            started_handles.append(handle)
            return handle

        return _start

    def test_distinct_server_keys_coexist_without_stopping_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True)):
                with mock.patch.dict(os.environ, {
                    "RSIBOX_VLLM_MODEL": "m",
                    "RSIBOX_VLLM_PORT": "18600",
                    "RSIBOX_VLLM_WORK": os.path.join(tmp, "w0"),
                    "CUDA_VISIBLE_DEVICES": "2",
                }):
                    lab0 = CL.VllmLab(shared=True)
                with mock.patch.dict(os.environ, {
                    "RSIBOX_VLLM_MODEL": "m",
                    "RSIBOX_VLLM_PORT": "18601",
                    "RSIBOX_VLLM_WORK": os.path.join(tmp, "w1"),
                    "CUDA_VISIBLE_DEVICES": "3",
                }):
                    lab1 = CL.VllmLab(shared=True)

            self.assertEqual(len(started), 2)
            self.assertNotEqual(lab0.server_key, lab1.server_key)
            self.assertTrue(lab0.server.alive())
            self.assertTrue(lab1.server.alive())
            self.assertEqual(lab0.server.state["stops"], 0)
            self.assertEqual(lab1.server.state["stops"], 0)
            self.assertEqual(set(CL.VllmLab._shared_servers.keys()), {lab0.server_key, lab1.server_key})

            # Explicit shared teardown for lab1 leaves distinct sibling lab0 untouched.
            lab1.close(stop_shared=True)
            self.assertTrue(lab0.server.alive())
            self.assertFalse(lab1.server.alive())
            self.assertEqual(lab0.server.state["stops"], 0)
            self.assertEqual(lab1.server.state["stops"], 1)

            self.assertEqual(CL.VllmLab.stop_shared(lab0.server_key), 1)
            self.assertFalse(lab0.server.alive())
            self.assertEqual(lab0.server.state["stops"], 1)
            self.assertEqual(CL.VllmLab._shared_servers, {})

    def test_unshared_close_stops_server_immediately_when_backend_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.dict(os.environ, {
                      "RSIBOX_VLLM_MODEL": "m",
                      "RSIBOX_VLLM_PORT": "18610",
                      "RSIBOX_VLLM_WORK": os.path.join(tmp, "w"),
                  })):
                lab_a = CL.VllmLab(shared=False)
                lab_b = CL.VllmLab(shared=False)

            self.assertIsNone(lab_a._backend)
            self.assertTrue(lab_a._owns_server)
            self.assertTrue(lab_b._owns_server)
            self.assertIsNot(lab_a.server, lab_b.server)
            self.assertEqual(CL.VllmLab._shared_servers, {})

            lab_a.close()
            self.assertFalse(lab_a.server.alive())
            self.assertEqual(lab_a.server.state["stops"], 1)
            self.assertTrue(lab_b.server.alive())
            self.assertEqual(lab_b.server.state["stops"], 0)

            # Idempotent second close must not stop again.
            lab_a.close()
            lab_a.close(stop_shared=True)
            self.assertEqual(lab_a.server.state["stops"], 1)

            lab_b.close()
            self.assertFalse(lab_b.server.alive())
            self.assertEqual(lab_b.server.state["stops"], 1)

    def test_shared_close_preserves_warm_server_for_sequential_reuse_and_stop_shared_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            env = {
                "RSIBOX_VLLM_MODEL": "m",
                "RSIBOX_VLLM_PORT": "18620",
                "RSIBOX_VLLM_WORK": os.path.join(tmp, "shared_w"),
                "CUDA_VISIBLE_DEVICES": "2",
            }
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True),
                  mock.patch.dict(os.environ, env)):
                lab1 = CL.VllmLab(shared=True)
                lab1.close()
                # Sequential conformance c0..c10 expects warm reuse after close().
                self.assertTrue(lab1.server.alive())
                self.assertEqual(lab1.server.state["stops"], 0)
                self.assertIn(lab1.server_key, CL.VllmLab._shared_servers)

                lab2 = CL.VllmLab(shared=True)
                self.assertEqual(len(started), 1)
                self.assertIs(lab1.server, lab2.server)
                lab2.close()
                self.assertTrue(lab2.server.alive())
                self.assertEqual(lab2.server.state["stops"], 0)

            # Explicit stop_shared=True (or atexit _cleanup_shared) stops and removes it once.
            lab2.close(stop_shared=True)
            lab2.close(stop_shared=True)
            self.assertFalse(lab2.server.alive())
            self.assertEqual(lab2.server.state["stops"], 1)
            self.assertEqual(CL.VllmLab._shared_servers, {})

            # Two shared=True owners sharing one server: both calling close(stop_shared=True) stops once.
            started.clear()
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True),
                  mock.patch.dict(os.environ, env)):
                owner_a = CL.VllmLab(shared=True)
                owner_b = CL.VllmLab(shared=True)
            self.assertIs(owner_a.server, owner_b.server)
            owner_a.close(stop_shared=True)
            owner_b.close(stop_shared=True)
            self.assertEqual(owner_a.server.state["stops"], 1)
            self.assertEqual(CL.VllmLab._shared_servers, {})

            # Calling close(stop_shared=True) after VllmLab.stop_shared() does not stop again.
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True),
                  mock.patch.dict(os.environ, env)):
                owner_c = CL.VllmLab(shared=True)
            self.assertEqual(CL.VllmLab.stop_shared(), 1)
            owner_c.close(stop_shared=True)
            self.assertEqual(owner_c.server.state["stops"], 1)
            self.assertEqual(CL.VllmLab._shared_servers, {})

    def test_unshared_init_cleans_up_server_if_session_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.SS, "Session", side_effect=RuntimeError("session boom")),
                  mock.patch.dict(os.environ, {
                      "RSIBOX_VLLM_MODEL": "m",
                      "RSIBOX_VLLM_PORT": "18625",
                      "RSIBOX_VLLM_WORK": os.path.join(tmp, "fail_init"),
                  })):
                with self.assertRaisesRegex(RuntimeError, "session boom"):
                    CL.VllmLab(shared=False)

            self.assertEqual(len(started), 1)
            self.assertFalse(started[0].alive())
            self.assertEqual(started[0].state["stops"], 1)
            self.assertEqual(CL.VllmLab._shared_servers, {})

    def test_acquire_shared_server_reaps_raced_dead_candidate_without_touching_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True)):
                with mock.patch.dict(os.environ, {
                    "RSIBOX_VLLM_MODEL": "m",
                    "RSIBOX_VLLM_PORT": "18626",
                    "RSIBOX_VLLM_WORK": os.path.join(tmp, "sibling"),
                    "CUDA_VISIBLE_DEVICES": "2",
                }):
                    sibling = CL.VllmLab(shared=True)

            # Register a candidate that passes initial alive() + _server_at_base()
            # and then exits right before re-acquiring _shared_lock.
            raced_state = {"alive_calls": 0, "stops": 0}

            def raced_alive():
                raced_state["alive_calls"] += 1
                return raced_state["alive_calls"] == 1

            def raced_stop():
                raced_state["stops"] += 1

            raced_srv = NS(port=18627, alive=raced_alive, stop=raced_stop, start_s=0.01)
            target_env = {
                "RSIBOX_VLLM_MODEL": "m",
                "RSIBOX_VLLM_PORT": "18627",
                "RSIBOX_VLLM_WORK": os.path.join(tmp, "target"),
                "CUDA_VISIBLE_DEVICES": "3",
            }
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=True),
                  mock.patch.dict(os.environ, target_env)):
                # Pre-populate target key using the exact server_key that VllmLab computes.
                probe_lab = CL.VllmLab(shared=False)
                probe_lab.close()
                CL.VllmLab._shared_servers[probe_lab.server_key] = raced_srv

                recovered = CL.VllmLab(shared=True)

            self.assertEqual(raced_state["stops"], 1)
            self.assertIsNot(recovered.server, raced_srv)
            self.assertIs(CL.VllmLab._shared_servers[recovered.server_key], recovered.server)
            self.assertTrue(sibling.server.alive())
            self.assertEqual(sibling.server.state["stops"], 0)

            # Intentional replacement when same-key server is alive but not at base
            # must also leave distinct siblings untouched.
            old_recovered_srv = recovered.server
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_server_at_base", return_value=False),
                  mock.patch.dict(os.environ, target_env)):
                replaced = CL.VllmLab(shared=True)

            self.assertFalse(old_recovered_srv.alive())
            self.assertEqual(old_recovered_srv.state["stops"], 1)
            self.assertIs(CL.VllmLab._shared_servers[replaced.server_key], replaced.server)
            self.assertTrue(sibling.server.alive())
            self.assertEqual(sibling.server.state["stops"], 0)

    def test_shadow_promotion_syncs_server_key_port_and_shared_registry(self):
        from . import snapshot_hooks as SH

        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.dict(os.environ, {
                      "RSIBOX_VLLM_MODEL": "m",
                      "RSIBOX_VLLM_PORT": "18640",
                      "RSIBOX_VLLM_WORK": os.path.join(tmp, "promo"),
                      "CUDA_VISIBLE_DEVICES": "2",
                  })):
                lab = CL.VllmLab(shared=True)

            old_key = lab.server_key
            self.assertEqual(old_key[2], 18640)
            self.assertEqual(old_key[6], "2")
            shadow = NS(port=18645, pid=99999, kernel_dir=os.path.join(tmp, "sk"), alive=lambda: True)
            promoter = SH.VllmShadowPromoter(shadow_server=shadow, timeout=5.0)
            with (mock.patch.object(SH, "_wake_resume_and_verify", return_value={"ok": True}),
                  mock.patch.object(lab.session, "recovered", return_value={"ok": True}),
                  mock.patch.object(lab.session, "poisoned", return_value=[])):
                res = promoter.promote(lab)

            self.assertTrue(res["ok"], res)
            self.assertEqual(lab.port, 18645)
            self.assertEqual(lab.server_key, old_key[:2] + (18645,) + old_key[3:])
            self.assertEqual(lab.server_key[6], "2")
            self.assertNotIn(old_key, CL.VllmLab._shared_servers)
            self.assertIs(CL.VllmLab._shared_servers[lab.server_key], shadow)

    def test_sandbox_pool_drain_and_close_stop_unshared_vllm_labs(self):
        from rsibox.pool import PoolSlot, SandboxPool

        with tempfile.TemporaryDirectory() as tmp:
            site = self._make_fake_site(tmp)
            started = []
            with (mock.patch.object(CL, "_site_packages", return_value=site),
                  mock.patch.object(CL.VllmLab, "_start_server", self._make_stub_start(started)),
                  mock.patch.object(CL.VllmLab, "_ctl", return_value={"ok": True}),
                  mock.patch.object(CL.VllmLab, "status", return_value={"state": "base"})):
                with mock.patch.dict(os.environ, {
                    "RSIBOX_VLLM_MODEL": "m",
                    "RSIBOX_VLLM_PORT": "18630",
                    "RSIBOX_VLLM_WORK": os.path.join(tmp, "p0"),
                    "CUDA_VISIBLE_DEVICES": "2",
                }):
                    lab0 = CL.VllmLab(shared=False)
                with mock.patch.dict(os.environ, {
                    "RSIBOX_VLLM_MODEL": "m",
                    "RSIBOX_VLLM_PORT": "18631",
                    "RSIBOX_VLLM_WORK": os.path.join(tmp, "p1"),
                    "CUDA_VISIBLE_DEVICES": "3",
                }):
                    lab1 = CL.VllmLab(shared=False)
                pool = SandboxPool([PoolSlot("s0", lab0), PoolSlot("s1", lab1)], close_labs=True)
                drained = pool.drain_slot("s1", close_lab=True)
                self.assertTrue(drained["ok"])
                self.assertTrue(lab0.server.alive())
                self.assertFalse(lab1.server.alive())
                self.assertEqual(lab1.server.state["stops"], 1)
                pool.close()
                self.assertFalse(lab0.server.alive())
                self.assertEqual(lab0.server.state["stops"], 1)


if __name__ == "__main__":
    unittest.main()
