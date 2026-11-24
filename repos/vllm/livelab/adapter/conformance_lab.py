"""VllmLab: a real vLLM server behind the conformance harness (LabUnderTest).

Starts `vllm serve` through `services/engine.py start` with `RSIBOX_KERNEL_DIR`, a
`sitecustomize.py` importing `adapter.host`, and `adapter.worker_ext.KernelWorkerExtension`
(`--distributed-executor-backend mp`). Control uses `adapter.client.send`; `check` runs greedy
`/v1/completions` (`temperature=0`, `seed=0`) on fixed prompts and returns generated texts.

Environment variables (no hostnames or paths in code):
- `RSIBOX_VLLM_MODEL` (required): model path or name served as `m`.
- `RSIBOX_VLLM_VENV`: venv root (`bin/python`, `bin/vllm`, site-packages; default `sys.prefix`).
- `RSIBOX_VLLM_WORK`: scratch dir for logs, caches, channels and source farm (default tempdir).
- `RSIBOX_VLLM_PORT`: live port (default `18500`; `fresh_check` uses `port + 1`).
- `RSIBOX_VLLM_TP`: tensor-parallel worker count (default 1).
- `RSIBOX_VLLM_EXTRA`: extra `vllm serve` flags (default `--seed 0 --gpu-memory-utilization 0.35
  --enforce-eager --max-model-len 2048 --disable-flashinfer-autotune`).
- `RSIBOX_VLLM_CHECK_TIMEOUT`: HTTP / control timeout in seconds (default `30`).
"""
import argparse
import atexit
import contextlib
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from rsibox import harness
from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L
from kernel import session as SS

from services import engine

from . import client

SAMPLER, LOGITS_PROC = "vllm.v1.worker.gpu.sample.sampler", "vllm.model_executor.layers.logits_processor"
SCHED_UTILS, SCHEDULER = "vllm.v1.core.sched.utils", "vllm.v1.core.sched.scheduler"
WORKER_MODULES, CORE_MODULES = (SAMPLER, LOGITS_PROC), (SCHED_UTILS, SCHEDULER)
MODULES = WORKER_MODULES + CORE_MODULES

PROMPTS = ("The capital of France is", "List the first five prime numbers:", "Water boils at a temperature of")
WORKLOAD = {"prompts": list(PROMPTS), "max_tokens": 12}
DEFAULT_EXTRA = ("--seed 0 --gpu-memory-utilization 0.35 --enforce-eager --max-model-len 2048 "
                 "--disable-flashinfer-autotune")
HOOK_SITE = "import os\nif os.environ.get('RSIBOX_KERNEL_DIR'):\n    import adapter.host\n"
_seq = itertools.count(1)


_edit = harness.edit_once
_module_relpath = harness.module_relpath


def _alive(pid):
    return bool(pid) and not engine.zygote._exited(pid)


def stop_tree(pid, grace_s=8.0):
    """Stop a live service process and its children using the launcher's lifecycle rules."""
    if pid:
        engine.stop_engine(pid, grace=grace_s)


def build_farm(src_pkg, dst_pkg):
    """Copy `.py` files of `src_pkg` into `dst_pkg` and symlink everything else."""
    for root, dirs, files in os.walk(src_pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        out_dir = os.path.join(dst_pkg, os.path.relpath(root, src_pkg))
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            src_path, dst_path = os.path.join(root, fname), os.path.join(out_dir, fname)
            if not os.path.lexists(dst_path):
                (shutil.copy2 if fname.endswith(".py") else os.symlink)(src_path, dst_path)


def _site_packages(venv):
    lib = os.path.join(venv, "lib")
    for entry in sorted(os.listdir(lib)) if os.path.isdir(lib) else ():
        cand = os.path.join(lib, entry, "site-packages")
        if os.path.isdir(os.path.join(cand, "vllm")):
            return cand
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    if spec and spec.submodule_search_locations:
        return os.path.dirname(list(spec.submodule_search_locations)[0])
    raise RuntimeError(f"vllm package not found under {venv}")


def _check_topology(ping, tp):
    if not ping.get("ok"):
        raise RuntimeError(f"adapter host did not answer ping: {ping}")
    parallel = ping.get("parallel") or {}
    if tp is not None and (parallel.get("world") != tp
                           or parallel.get("expected") != list(range(tp))):
        raise RuntimeError(f"requested TP={tp}, adapter host reports {parallel}")


def _http_post(port, path, body, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
        {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class HostBridge:
    """Session-compatible coordinator adapter over `adapter.client.send`."""

    def __init__(self, call_ctl):
        self._ctl, self.poison = call_ctl, []
        self.participants = [
            C.CallParticipant("workers", lambda m, p: self._part("workers", m, p)),
            C.CallParticipant("core", lambda m, p: self._part("core", m, p)),
        ]

    def _part(self, name, method, payload):
        if method != "fingerprint":
            return F.not_ok(f"bridge supports fingerprint only, not {method}")
        role_set = CORE_MODULES if name == "core" else WORKER_MODULES
        mods = [m for m in (payload or ()) if m in role_set] or list(role_set)
        reply = self._ctl({"op": "fingerprint", "modules": mods})
        side = (reply.get(name) or {}) if reply.get("ok") else {}
        return F.ok(code=side.get("code") or {}) if side.get("ok") else F.not_ok(
            reply.get("error") or side.get("error") or f"{name} fingerprint failed")

    def run(self, plan, **kw):
        cmd = {"op": "txn", "plan": plan, **({"edit_time": kw["edit_time"]} if kw.get("edit_time") else {})}
        reply = self._ctl(cmd)
        self._sync_poison(reply)
        if "outcome" not in reply:
            return F.not_ok(reply.get("error") or "host did not answer txn",
                            [F.finding("participant_failed", str(reply.get("error")), "host answers")],
                            outcome="aborted", txn=plan.get("id"))
        return reply

    def prove(self, txn, golden=None):
        return self._ctl({"op": "prove", "txn": txn, "golden": golden or {}})

    def undo(self, txn):
        reply = self._ctl({"op": "undo", "txn": txn})
        self._sync_poison(reply)
        return reply

    def status(self, txn=None):
        reply = self._ctl({"op": "status", **({"txn": txn} if txn is not None else {})})
        self._sync_poison(reply)
        return reply

    def recovered(self):
        self.poison.clear()
        return F.ok()

    def _sync_poison(self, reply):
        ping = self._ctl({"op": "ping"})
        if ping.get("ok") and ping.get("poison"):
            self.poison = list(ping["poison"])
        elif not reply.get("ok") and reply.get("lost"):
            self.poison.append(F.finding("participant_failed", str(reply.get("error")), "host answers"))


class _ServerHandle:
    def __init__(self, pid, port, kernel_dir, start_s, log_path, marker=None):
        self.pid, self.port, self.kernel_dir = pid, port, kernel_dir
        self.start_s, self.log_path = start_s, log_path
        self.marker, self.start_ticks = marker, engine._start_time(pid)

    def alive(self):
        return _alive(self.pid) and engine._same_process(self.pid, self.start_ticks)

    def stop(self):
        if self.alive() or not _alive(self.pid):
            engine.stop_engine(self.pid, grace=8, marker=self.marker)


class VllmLab:
    """`LabUnderTest` driving a live vLLM server through `adapter.host`."""
    name = "vllm"
    _shared = None

    @staticmethod
    def launch_environment(argv):
        """Translate `rsibox conformance vllm --real ...` flags into `RSIBOX_VLLM_*` env vars.

        Accepts both `rsibox conformance vllm --real --model M [-- <serve flags>]` and
        `rsibox conformance vllm --real -- --model M [-- <serve flags>]`, as well as
        `rsibox conformance vllm --real -- <serve flags>` when `RSIBOX_VLLM_MODEL` is preset.
        """
        argv = list(argv)
        known_opts = {"--model", "--venv", "--work", "--port", "--tp", "--extra",
                      "--check-timeout", "--no-share-gpu",
                      "--weights-backend", "--gms-socket-dir", "--gms-pythonpath",
                      "--gms-read-only", "--gms-scratch-kv", "-h", "--help"}
        if argv[:1] == ["--"] and len(argv) > 1:
            first_opt = argv[1].split("=", 1)[0]
            if first_opt in known_opts or "--" in argv[1:]:
                argv = argv[1:]
        if "--" in argv:
            idx = argv.index("--")
            own, tail = argv[:idx], argv[idx + 1:]
        else:
            own, tail = argv, []
        norm, i = [], 0
        while i < len(own):
            if own[i] == "--extra" and i + 1 < len(own) and own[i + 1] not in known_opts:
                norm.append(f"--extra={own[i + 1]}")
                i += 2
            else:
                norm.append(own[i])
                i += 1
        ap = argparse.ArgumentParser(prog="rsibox conformance vllm --real", allow_abbrev=False)
        for flag, kw in (
            ("--model", dict(default=None, help="model path or HuggingFace repo id")),
            ("--venv", dict(default=None, help="vLLM virtualenv root")),
            ("--work", dict(default=None, help="scratch directory for logs, caches and farm")),
            ("--port", dict(type=int, default=None, help="live server port")),
            ("--tp", dict(type=int, default=None, help="tensor-parallel worker count")),
            ("--extra", dict(default=None, help="extra `vllm serve` flags string")),
            ("--check-timeout", dict(type=float, default=None, help="HTTP / control timeout in seconds")),
            ("--no-share-gpu", dict(action="store_true", help="keep live server running during fresh_check")),
            ("--weights-backend", dict(choices=["none", "ipc_cache", "dynamo_gms"], default=None, help="pluggable resident weights backend")),
            ("--gms-socket-dir", dict(default=None, help="Dynamo GMS socket directory")),
            ("--gms-pythonpath", dict(default=None, help="extra PYTHONPATH containing gpu_memory_service")),
            ("--gms-read-only", dict(action="store_true", help="attach Dynamo GMS weights read-only")),
            ("--gms-scratch-kv", dict(action="store_true", help="enable Dynamo GMS Scratch-KV mode")),
        ):
            ap.add_argument(flag, **kw)
        ns = ap.parse_args(norm)
        env = {}
        for attr, key, xform in (
            ("model", "RSIBOX_VLLM_MODEL", str),
            ("venv", "RSIBOX_VLLM_VENV", os.path.abspath),
            ("work", "RSIBOX_VLLM_WORK", os.path.abspath),
            ("port", "RSIBOX_VLLM_PORT", str),
            ("tp", "RSIBOX_VLLM_TP", str),
            ("check_timeout", "RSIBOX_VLLM_CHECK_TIMEOUT", str),
            ("weights_backend", "RSIBOX_VLLM_WEIGHTS_BACKEND", str),
            ("gms_socket_dir", "RSIBOX_VLLM_GMS_SOCKET_DIR", os.path.abspath),
            ("gms_pythonpath", "RSIBOX_VLLM_GMS_PYTHONPATH", str),
        ):
            val = getattr(ns, attr)
            if val is not None and (attr not in ("model", "venv", "work", "gms_socket_dir") or val):
                env[key] = xform(val)
        if ns.no_share_gpu:
            env["RSIBOX_VLLM_SHARE_GPU"] = "0"
        if ns.gms_read_only:
            env["RSIBOX_VLLM_GMS_READ_ONLY"] = "1"
        if ns.gms_scratch_kv:
            env["RSIBOX_VLLM_GMS_SCRATCH_KV"] = "1"
        if ns.extra is not None or tail:
            env["RSIBOX_VLLM_EXTRA"] = shlex.join((shlex.split(ns.extra) if ns.extra is not None else []) + tail)
        return env

    def __init__(self, receipts=None, check_timeout=None, backend=None, base_texts=None):
        self.receipts = receipts
        self.check_timeout = float(check_timeout or os.environ.get("RSIBOX_VLLM_CHECK_TIMEOUT", "30"))
        self._backend, self._fault_txns, self._pending_inject, self._stack_files = backend, [], None, []
        if backend is not None:
            self.base_text = dict(base_texts)
            self.bridge = HostBridge(self._ctl)
            self.session = SS.Session(self.bridge, MODULES)
            return
        self.model = os.environ["RSIBOX_VLLM_MODEL"]
        self.venv = os.environ.get("RSIBOX_VLLM_VENV") or sys.prefix
        self.python_bin, self.vllm_bin = os.path.join(self.venv, "bin", "python"), os.path.join(self.venv, "bin", "vllm")
        self.site = _site_packages(self.venv)
        self.port = int(os.environ.get("RSIBOX_VLLM_PORT", "18500"))
        self.tp = int(os.environ.get("RSIBOX_VLLM_TP", "1"))
        self.extra = shlex.split(os.environ.get("RSIBOX_VLLM_EXTRA", DEFAULT_EXTRA))
        self.share_gpu = os.environ.get("RSIBOX_VLLM_SHARE_GPU", "1") != "0"
        self.weights_backend = os.environ.get("RSIBOX_VLLM_WEIGHTS_BACKEND", "").strip()
        self.gms_socket_dir = os.environ.get("RSIBOX_VLLM_GMS_SOCKET_DIR", "").strip()
        self.gms_pythonpath = os.environ.get("RSIBOX_VLLM_GMS_PYTHONPATH", "").strip()
        self.gms_read_only = os.environ.get("RSIBOX_VLLM_GMS_READ_ONLY", "0").strip().lower() in ("1", "true", "yes")
        self.gms_scratch_kv = os.environ.get("RSIBOX_VLLM_GMS_SCRATCH_KV", "0").strip().lower() in ("1", "true", "yes")
        work_env = os.environ.get("RSIBOX_VLLM_WORK")
        self.work_dir = os.path.abspath(work_env) if work_env else tempfile.mkdtemp(prefix="rsibox-vllm-")
        self.livelab_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine_py = os.path.join(self.livelab_root, "services", "engine.py")
        self.hooks_dir = os.path.join(self.work_dir, "hooks")
        self.cache_root = os.path.join(self.work_dir, "cache")
        self.farm_dir = os.path.join(self.work_dir, "farm")
        for path in (self.work_dir, self.hooks_dir, self.cache_root):
            os.makedirs(path, exist_ok=True)
        Path(self.hooks_dir, "sitecustomize.py").write_text(HOOK_SITE)
        if not os.path.isdir(os.path.join(self.farm_dir, "vllm")):
            build_farm(os.path.join(self.site, "vllm"), os.path.join(self.farm_dir, "vllm"))
        self._farm_dirty = set()
        self.base_text = {m: Path(self.site, _module_relpath(self.site, m)).read_text() for m in MODULES}
        self.server_key = (self.model, self.venv, self.port, self.work_dir, self.tp,
                           tuple(self.extra), os.environ.get("CUDA_VISIBLE_DEVICES"),
                           os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
                           os.environ.get("VLLM_BATCH_INVARIANT"),
                           self.weights_backend,
                           self.gms_socket_dir or os.environ.get("GMS_SOCKET_DIR"),
                           self.gms_pythonpath, self.gms_read_only,
                           self.gms_scratch_kv or os.environ.get("DYN_GMS_SCRATCH_KV_ENABLED"))
        shared = VllmLab._shared
        if (shared is not None and shared.key == self.server_key and shared.alive()
                and self._server_at_base(shared)):
            self.server = shared
        else:
            if shared is not None:
                shared.stop()
                VllmLab._shared = None
            self.server = VllmLab._shared = self._start_server(self.port, "live", use_farm=False)
        self.bridge = HostBridge(self._ctl)
        self.session = SS.Session(self.bridge, MODULES)

    def _server_at_base(self, srv):
        try:
            ping = client.send(srv.kernel_dir, {"op": "ping"}, timeout=10.0)
            return bool(ping.get("ok") and not ping.get("poison")
                        and client.send(srv.kernel_dir, {"op": "at_base"}, timeout=10.0).get("ok"))
        except Exception:
            return False

    def _start_server(self, port, tag, use_farm):
        run_id = next(_seq)
        kernel_dir = os.path.join(self.work_dir, f"kdir-{tag}-{run_id}")
        shutil.rmtree(kernel_dir, ignore_errors=True)
        os.makedirs(kernel_dir, exist_ok=True)
        log_path = os.path.join(self.work_dir, f"engine-{tag}-{run_id}.log")
        py_parts = [self.hooks_dir, self.livelab_root] + ([self.farm_dir] if use_farm else [])
        py_parts += [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
        env = {**os.environ, "RSIBOX_KERNEL_DIR": kernel_dir, "PYTHONPATH": os.pathsep.join(py_parts),
               "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
               "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT", "1")}
        cmd = [
            self.python_bin, self.engine_py, "start", "--cache-root", self.cache_root,
            "--log", log_path, "--vllm", self.vllm_bin,
        ]
        # Ground-truth farm runs (`use_farm=True`) always load fresh from disk so
        # loader/quantization patches are never bypassed by committed resident weights.
        active_backend = "" if use_farm else self.weights_backend
        if active_backend == "dynamo_gms":
            cmd += ["--weights-backend", "dynamo_gms"]
            if self.gms_socket_dir:
                cmd += ["--gms-socket-dir", self.gms_socket_dir]
            if self.gms_pythonpath:
                cmd += ["--gms-pythonpath", self.gms_pythonpath]
            if self.gms_read_only:
                cmd += ["--gms-read-only"]
            if self.gms_scratch_kv:
                cmd += ["--gms-scratch-kv"]
        else:
            cmd += [f"--kvplan={os.path.join(self.work_dir, 'kvplan')}"]
            if active_backend == "ipc_cache":
                cmd += ["--weights-backend", "ipc_cache"]
        cmd += [
            "--timeout", "900", "--",
            self.model, "--port", str(port), "--served-model-name", "m", "-tp", str(self.tp),
            "--distributed-executor-backend", "mp",
            "--worker-extension-cls", "adapter.worker_ext.KernelWorkerExtension", *self.extra,
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=960)
        lines = proc.stdout.strip().splitlines()
        report = json.loads(lines[-1]) if lines else {}
        if not report.get("healthy_after_s"):
            if report.get("pid"):
                stop_tree(report["pid"])
            tail = Path(log_path).read_text(errors="replace")[-2000:] if os.path.isfile(log_path) else ""
            raise RuntimeError(f"vllm start ({tag}) failed rc={proc.returncode}: "
                               f"{report.get('findings') or report.get('error')} {tail}")
        pid = int(report["pid"])
        state = engine.load_engine_state(self.cache_root, pid) or {}
        srv = _ServerHandle(pid, port, kernel_dir, round(time.perf_counter() - t0, 3),
                            log_path, state.get("live_dir"))
        srv.key = self.server_key
        try:
            _http_post(port, "/v1/completions",
                       {"model": "m", "prompt": PROMPTS[0], "max_tokens": 2, "temperature": 0.0, "seed": 0},
                       timeout=120.0)
            ping = client.send(kernel_dir, {"op": "ping"}, timeout=30.0)
            # Older callers can still select TP through arbitrary extra serve flags.
            _check_topology(ping, self.tp if "RSIBOX_VLLM_TP" in os.environ else None)
        except BaseException:
            srv.stop()
            raise
        return srv

    def _ctl(self, cmd, timeout=None):
        wait_s = timeout or self.check_timeout
        if self._backend is not None:
            return self._backend.ctl(cmd, timeout=wait_s)
        if not self.server.alive():
            return {"ok": False, "lost": True, "error": "engine process exited"}
        reply = client.send(self.server.kernel_dir, cmd, timeout=wait_s)
        if not reply.get("ok") and str(reply.get("error", "")).startswith("no reply in "):
            reply["lost"] = True
        return reply

    def _run_http(self, port, workload, timeout):
        spec = workload or WORKLOAD
        prompts, max_tokens = spec.get("prompts", PROMPTS), int(spec.get("max_tokens", WORKLOAD["max_tokens"]))
        if self._backend is not None:
            return self._backend.complete(prompts, max_tokens, timeout=timeout)
        return {"worker0": [_http_post(
            port, "/v1/completions",
            {"model": "m", "prompt": pr, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0},
            timeout=timeout)["choices"][0]["text"] for pr in prompts]}

    def _append_receipt(self, record):
        if record:
            harness.append_jsonl(self.receipts, record)

    def _disk_source(self, module_name):
        if module_name not in self.base_text and getattr(self, "site", None):
            with contextlib.suppress(FileNotFoundError, OSError):
                self.base_text[module_name] = Path(self.site, _module_relpath(self.site, module_name)).read_text()
        return self.base_text.get(module_name)

    def _sources(self):
        return harness.SourceStack(self._disk_source, self._stack_files, files_key="files", min_len=0)

    def _live_source(self, module_name):
        src = self._sources().live_source(module_name)
        if src is None:
            raise FileNotFoundError(f"module source not found: {module_name}")
        return src

    def files(self, patch, against="live"):
        """Resolve `{module: new_text}` from `patch["files"]` or unified `patch["diff"]`."""
        return self._sources().resolve(patch, against=against)

    def _role_of(self, module_name):
        return "core" if (module_name in CORE_MODULES or module_name.startswith("vllm.v1.core.")) else "worker"

    def _clear_fault_txns(self):
        while self._fault_txns:
            self._ctl({"op": "undo", "txn": self._fault_txns.pop()}, timeout=15.0)
        self.session.expected[0] = self.session._fingerprint()

    def install(self, patch):
        plan_id = f"txn-{next(_seq):04d}-{int(time.time() * 1000) % 1000000}"
        return self._install(patch, plan_id)

    def _install(self, patch, plan_id):
        t0 = time.perf_counter()
        try:
            if not isinstance(patch, dict) or ("diff" not in patch and "files" not in patch):
                raise ValueError("patch must define 'files' or 'diff'")
            resolved_files = self.files(patch, against="live")
            modules = {
                m: {"source": src, "base_hash": L.text_hash(self._live_source(m)),
                    "targets": "auto", "on": [self._role_of(m)]}
                for m, src in resolved_files.items()
            }
        except Exception as err:
            finding = F.finding("patch_invalid", f"{type(err).__name__}: {err}", "valid module")
            return harness.install_refusal(["patch_invalid"], [finding], t0=t0)
        plan = {"id": plan_id, "modules": modules, "accept": list(patch.get("accept", [])),
                "undo_resets": list(patch.get("undo_resets", [])), "recipes": ["R0 swap"]}
        pending = self._pending_inject
        if pending and pending["fault"] == "fail_commit":
            self._pending_inject = None
            plan["faults"] = {pending.get("role", "worker"): "commit"}
        res = self.session.install(plan, edit_time=patch.get("edit_time"))
        if pending and pending["fault"] == "fail_apply":
            self._pending_inject = None
            self._clear_fault_txns()
        outcome, rec = res.get("outcome") or "refused", res.get("receipt")
        self._append_receipt(rec)
        if outcome == "committed":
            self._sources().push({"id": res["txn"], "files": dict(resolved_files),
                                  "accept": list(patch.get("accept", [])),
                                  "undo_resets": list(patch.get("undo_resets", []))})
        codes = rec["decisions"]["blocking"] if rec else sorted({f["check"] for f in res.get("findings", [])})
        return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome,
                "codes": [] if outcome == "committed" else codes,
                "mechanisms": rec["actual_recipes"] if rec else [],
                "findings": rec["findings"] if rec else res.get("findings", []),
                "receipt": rec, "cost_s": round(time.perf_counter() - t0, 6)}

    def _check_failed(self, msg, expected, t0):
        self.session.fault("participant_failed", msg, expected)
        return harness.check_failure(msg, list(self.session.poisoned()), t0=t0)

    def check(self, workload=None):
        t0 = time.perf_counter()
        ping = self._ctl({"op": "ping"}, timeout=min(self.check_timeout, 10.0))
        if not ping.get("ok"):
            return self._check_failed(f"ping: {ping.get('error')}", "the engine host answers", t0)
        fp_probe = self.bridge._part("workers", "fingerprint", list(MODULES))
        if not fp_probe.get("ok"):
            return self._check_failed(f"workers: {fp_probe.get('error')}", "every worker answers", t0)
        try:
            out = self._run_http(0 if self._backend is not None else self.server.port,
                                 workload, timeout=self.check_timeout)
        except Exception as err:
            return self._check_failed(f"check: {type(err).__name__}: {err}",
                                      f"the workload finishes within {self.check_timeout} s", t0)
        golden = {"output_hash": L.text_hash(json.dumps(out, sort_keys=True))}
        proofs = []
        for tid in self.session.stack:
            pr = self.bridge.prove(tid, golden=golden)
            self._append_receipt(pr.get("receipt"))
            proofs.append(pr)
        witness = {}
        for pr in proofs:
            for part, counts in ((pr.get("receipt") or {}).get("witness") or {}).items():
                if counts:
                    witness.setdefault(part, {}).update(counts)
        return {"ok": True, "output": out,
                "proven": all(p.get("ok") for p in proofs) if proofs else None,
                "witness": witness,
                "findings": [f for p in proofs for f in (p.get("receipt") or {}).get("findings", [])],
                "cost_s": round(time.perf_counter() - t0, 6)}

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        if kind == "respawn":
            self._pending_inject, self._fault_txns, self._stack_files = None, [], []
            if self._backend is not None:
                self._backend.respawn()
                start_info = {}
            else:
                self.server.stop()
                VllmLab._shared = None
                self.server = VllmLab._shared = self._start_server(self.port, "live", use_farm=False)
                start_info = {"start_s": self.server.start_s}
            self.bridge = HostBridge(self._ctl)
            return {"ok": self.session.recovered(self.bridge)["ok"], "reset": ["worker group", "session"],
                    **start_info, "cost_s": round(time.perf_counter() - t0, 6)}
        if self._backend is not None:
            self._backend.reset_state()
            return {"ok": True, "reset": ["prefix_cache"], "cost_s": round(time.perf_counter() - t0, 6)}
        tid = f"reset-{next(_seq)}"
        cell = "core.reset_prefix_cache() if hasattr(core, 'reset_prefix_cache') else None\n"
        res = self._ctl({"op": "txn", "plan": {
            "id": tid, "modules": {}, "repairs": [{"cell": cell, "declares": ["core"], "on": ["core"]}]}},
            timeout=self.check_timeout)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "state reset failed", "reset": [],
                    "cost_s": round(time.perf_counter() - t0, 6)}
        self._ctl({"op": "undo", "txn": tid}, timeout=self.check_timeout)
        return {"ok": True, "reset": ["prefix_cache"], "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        t0 = time.perf_counter()
        res = self.session.uninstall(id)
        for item in res.get("results", []):
            self._append_receipt(item.get("receipt"))
        undone = res.get("undone", [])
        if undone:
            self._sources().pop(len(undone))
        return {"ok": res["ok"], "undone": undone, "state": self.session.state(),
                "findings": res.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        return self.session.status()

    def close(self):
        if self._backend is not None:
            self._backend.close()

    def patches(self):
        samp, sutils = self.base_text[SAMPLER], self.base_text[SCHED_UTILS]
        sched, lproc = self.base_text[SCHEDULER], self.base_text[LOGITS_PROC]
        s_anchor = "        sampled, processed_logits = self.sample(\n"
        s_noop = _edit(samp, s_anchor, "        logits = logits\n" + s_anchor)
        s_vis1 = _edit(samp, s_anchor, "        logits = -logits\n" + s_anchor)
        s_vis2 = _edit(s_vis1, "        logits = -logits\n", "        logits = logits.roll(1, dims=-1)\n")
        u_anchor = "    last_token_id = request.output_token_ids[-1]\n"
        u_noop = _edit(sutils, u_anchor, "    _ = max_model_len\n" + u_anchor)
        u_vis = _edit(sutils, "        or request.num_output_tokens >= request.max_tokens\n",
                      "        or request.num_output_tokens >= max(1, request.max_tokens - 4)\n")
        s_m1 = _edit(s_vis1, "        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)\n",
                     "        idx_mapping_np = idx_mapping_np\n"
                     "        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)\n")
        s_multi = _edit(s_m1, "        top_k, top_p, has_topp_only_rows = self.sampling_states.get_top_k_top_p(\n",
                        "        return_logprobs = bool(return_logprobs)\n"
                        "        top_k, top_p, has_topp_only_rows = self.sampling_states.get_top_k_top_p(\n")
        lp_unreached = _edit(lproc, "        self.scale = scale\n", "        self.scale = -1.0\n")
        sched_closure = _edit(sched, "                self.deferred_waiting.add(request_to_skip)\n",
                              "                _ = token_budget\n"
                              "                self.deferred_waiting.add(request_to_skip)\n")
        lproc_state = _edit(lproc, "logger = init_logger(__name__)\n",
                            "logger = init_logger(__name__)\n_RSIBOX_MOD_STATE = 1\n")
        p = lambda **files: {"files": dict(files)}  # noqa: E731
        return {
            "noop": p(**{SAMPLER: s_noop, SCHED_UTILS: u_noop}),
            "visible": p(**{SAMPLER: s_vis1, SCHED_UTILS: u_noop}),
            "chain": [p(**{SAMPLER: s_vis1, SCHED_UTILS: u_noop}), p(**{SAMPLER: s_vis2}), p(**{SCHED_UTILS: u_vis})],
            "multi": p(**{SAMPLER: s_multi, SCHED_UTILS: u_vis}),
            "unreached": {**p(**{LOGITS_PROC: lp_unreached}), "accept": ["constructor_edit"]},
            "refused": {
                "syntax": p(**{SAMPLER: _edit(samp, "return sampler_output\n", "return ,,\n")}),
                "closure": p(**{SCHEDULER: sched_closure}),
                "module_state": p(**{LOGITS_PROC: lproc_state}),
                "empty": p(**{SAMPLER: _edit(samp, "        # These are GPU tensors.\n",
                                             "        # These are GPU tensors (unchanged).\n")}),
            },
        }

    def _write_farm(self, files):
        for name in list(self._farm_dirty):
            rel = _module_relpath(self.site, name)
            shutil.copy2(os.path.join(self.site, rel), os.path.join(self.farm_dir, rel))
        self._farm_dirty.clear()
        for name, text in (files or {}).items():
            dst = os.path.join(self.farm_dir, _module_relpath(self.site, name))
            if os.path.islink(dst):
                os.unlink(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Path(dst).write_text(text)
            self._farm_dirty.add(name)

    def fresh_check(self, patch, workload=None):
        try:
            resolved_files = self.files(patch, against="disk")
        except Exception as err:
            return {"ok": False, "error": f"{type(err).__name__}: {err}", "output": None}
        if self._backend is not None:
            return self._backend.fresh_check({**(patch or {}), "files": resolved_files}, workload)
        paused_live, saved_stack = False, [dict(x) for x in self._stack_files]
        if self.share_gpu and self.server is not None and self.server.alive():
            self.server.stop()
            VllmLab._shared, paused_live = None, True
        fresh_srv = None
        try:
            self._write_farm(resolved_files)
            fresh_srv = self._start_server(self.port + 1, "fresh", use_farm=True)
            return {"ok": True, "output": self._run_http(fresh_srv.port, workload, max(self.check_timeout, 60.0))}
        except Exception as err:
            return {"ok": False, "error": f"{type(err).__name__}: {err}", "output": None}
        finally:
            if fresh_srv is not None:
                fresh_srv.stop()
            self._write_farm({})
            if paused_live:
                self.server = VllmLab._shared = self._start_server(self.port, "live", use_farm=False)
                self.bridge = HostBridge(self._ctl)
                self.session = SS.Session(self.bridge, MODULES)
                self._stack_files.clear()
                for saved_patch in saved_stack:
                    if not self._install(saved_patch, saved_patch["id"])["ok"]:
                        self.session.fault("participant_failed", "incomplete edit replay",
                                           "restore the live stack after fresh_check")
                        raise RuntimeError("failed to restore live stack after fresh_check")

    def inject(self, fault, **kw):
        if fault == "fail_commit":
            self._pending_inject = {"fault": "fail_commit", "role": kw.get("participant", "worker")}
            return {"ok": True, "fault": fault}
        if fault == "fail_apply":
            after = int(kw.get("after", 1))
            cell = (f"_orig, _n = journal.set_code, [0]\n"
                    f"def _set_code(*a, **k):\n"
                    f"    _n[0] += 1\n"
                    f"    if _n[0] > {after}:\n"
                    f"        journal.set_code = _orig\n"
                    f"        raise RuntimeError('injected fail_apply after {after} swap(s)')\n"
                    f"    return _orig(*a, **k)\n"
                    f"journal.set_code = _set_code\n")
            res = self._install_fault_cell(fault, "worker", cell, track=True)
            self._pending_inject = {"fault": "fail_apply"}
            return res
        if fault in ("crash", "hang"):
            hang_s = float(kw.get("hang_s", self.check_timeout + 15.0))
            act = ("raise RuntimeError('injected worker crash')" if fault == "crash"
                   else f"import time; time.sleep({hang_s})")
            cell = ("_orig = worker.execute_model\n"
                    "def _boom(*a, **k):\n"
                    "    worker.execute_model = _orig\n"
                    f"    {act}\n"
                    "worker.execute_model = _boom\n")
            return self._install_fault_cell(fault, "worker", cell, track=False)
        if fault == "lost":
            res = self._install_fault_cell(fault, "worker", "import os, threading\nthreading.Timer(0.05, lambda: os._exit(18)).start()\n", False)
            time.sleep(0.15)
            return res
        if fault == "tamper":
            suffix = SCHED_UTILS.split(".", 1)[1]
            cell = (f"import sys\n_m = sys.modules.get({SCHED_UTILS!r}) or next("
                    f"v for k, v in sys.modules.items() if k.endswith('.{suffix}'))\n"
                    f"_m.check_stop.__code__ = _m.check_stop.__code__.replace("
                    f"co_name=_m.check_stop.__code__.co_name)\n")
            return self._install_fault_cell(fault, "core", cell, track=False)
        return {"ok": False, "error": f"unknown fault {fault!r}"}

    def _install_fault_cell(self, fault, role, cell, track):
        tid = f"fault-{fault}-{next(_seq)}"
        plan = {"id": tid, "modules": {}, "accept": "all",
                "repairs": [{"cell": cell, "declares": ["worker"] if role == "worker" else [], "on": [role]}]}
        reply = self._ctl({"op": "txn", "plan": plan}, timeout=15.0)
        ok = reply.get("outcome") == "committed"
        if ok and track:
            self._fault_txns.append(tid)
            self.session.expected[0] = self.session._fingerprint()
        return {"ok": ok, "fault": fault, **({} if ok else {"error": reply.get("error")})}


@atexit.register
def _cleanup_shared():
    if VllmLab._shared is not None:
        VllmLab._shared.stop()
        VllmLab._shared = None
