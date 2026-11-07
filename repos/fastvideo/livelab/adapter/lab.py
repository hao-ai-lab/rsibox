"""FastVideoLab: the live-lab verbs on a running FastVideo VideoGenerator.

Process layout. The lab host is the driver process: it owns the `VideoGenerator` and never
runs engine code itself. The engine always runs in spawned workers (MultiprocExecutor), also
on one GPU, where FastVideo would pick its in-process executor: a respawn then really
re-imports base code, and every rank is reached the same way. Each worker holds one kernel
`LocalParticipant` (adapter/agent.py); the host reaches it through that worker's own pipe, one
rank at a time, with a deadline (FastVideo's `collective_rpc` ignores its timeout; a reply that
misses the deadline leaves the pipe unusable, so the lab is poisoned until a respawn).

Quiesce point. A generation is one `execute_forward` RPC per worker: the denoise loop runs
inside it, and a worker reads its pipe only between requests. So every transaction's phase 2
runs in the gap between two requests; the lab's lock serializes verbs and generations. An
edit can never switch in the middle of a video, and every rank switches in the same gap.

Output. `check` runs one generation of the workload and returns `output` = sha256 prefixes of
the uint8 video frames and the audio waveform (bit-exact comparison), plus timings and, when a
golden is set, PSNR against it.

Never raises: every verb returns an ok-dict.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L
from kernel import receipt as R
from kernel import session as SS
from adapter import agent

H3 = "fastvideo.pipelines.basic.minimax_h3"
DEC = f"{H3}.stages.minimax_h3_decoding"
LAT = f"{H3}.stages.minimax_h3_latent_preparation"
INP = f"{H3}.stages.minimax_h3_input_preparation"
PIPE = f"{H3}.minimax_h3_pipeline"
# modules every H3 worker imports at load: the session fingerprints these (tamper check)
MODULES = (DEC, LAT, INP, f"{H3}.stages.minimax_h3_denoising",
           f"{H3}.stages.minimax_h3_conditioning", PIPE, "fastvideo.models.dits.minimax_h3")
RPC_ERROR = "__fastvideo_rpc_error__"


def finding(check, observed, expected):
    """A kernel finding when the kernel catalogues the check; a plain one otherwise."""
    try:
        return F.finding(check, observed, expected)
    except KeyError:
        return {"check": check, "observed": str(observed), "expected": str(expected)}


def module_file(name):
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def read_module(name):
    with open(module_file(name)) as f:
        return f.read()


def digest(arr_bytes):
    return hashlib.sha256(arr_bytes).hexdigest()[:16]


def output_of(result):
    """{"video": hash, "audio": hash} of one generation's result, and the frames array."""
    import numpy as np
    d = result.to_legacy_dict() if hasattr(result, "to_legacy_dict") else result
    fr = d.get("frames")
    arr = np.stack([np.asarray(f) for f in fr]) if isinstance(fr, list) else np.asarray(fr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    out = {"video": digest(arr.tobytes())}
    au = d.get("audio")
    if au is not None:
        au = au.detach().cpu().contiguous().numpy() if hasattr(au, "detach") else np.asarray(au)
        out["audio"] = digest(au.tobytes())
    return out, arr, d


def psnr(a, b):
    import numpy as np
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 3)


def build_request(config, workload=None):
    """The request of one check: the config's request, workload overrides, frames returned."""
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import RunConfig
    cfg = json.loads(json.dumps(config))
    for k, v in ((workload or {}).get("sampling") or {}).items():
        cfg["request"]["sampling"][k] = v
    if (workload or {}).get("prompt"):
        cfg["request"]["prompt"] = workload["prompt"]
    cfg["request"]["output"].update(save_video=False, return_frames=True)
    return parse_config(RunConfig, cfg).request


def force_multiproc():
    """One GPU runs in a spawned worker too (FastVideo picks the in-process executor there)."""
    from fastvideo.worker import executor as E
    if getattr(E.Executor.get_class, "_fastrsi", False):
        return
    orig = E.Executor.get_class

    def get_class(args, **kw):
        if args.distributed_executor_backend == "mp" and args.num_gpus == 1:
            from fastvideo.worker.multiproc_executor import MultiprocExecutor
            return MultiprocExecutor
        return orig(args, **kw)
    get_class._fastrsi = True
    E.Executor.get_class = staticmethod(get_class)


class FastVideoLab:
    """`config`: a FastVideo run config (dict, `fastvideo generate --config` schema).
    `receipts`: receipt file (JSON lines). `golden`: path of a uint8 frames .npy.
    `zygote`: fork the workers from a preloaded forkserver (services/zygote.py), so a
    `reset("respawn")` skips the workers' imports (zygote respawn). `holder`: also keep the loaded DiT and
    text encoder in a holder process (services/holder.py) that respawned workers map by CUDA
    IPC instead of loading; implies `zygote`."""
    name = "fastvideo"

    def __init__(self, config, receipts=None, golden=None, modules=MODULES, rpc_timeout=60.0,
                 check_timeout=600.0, zygote=False, holder=False):
        self.config = json.loads(json.dumps(config))
        self.writer = R.ReceiptWriter(receipts) if receipts else None
        self.rpc_timeout, self.check_timeout = rpc_timeout, check_timeout
        self.modules = list(modules)
        self.golden = None
        if golden and os.path.exists(golden):
            import numpy as np
            self.golden = np.load(golden)
        self.lock = threading.RLock()
        self.costs = {}
        self.holder = None
        if holder:          # loaded DiT/text encoder outlive the workers (services/holder.py)
            from services import holder as H
            sock = os.path.join(tempfile.gettempdir(), f"fastrsi-holder-{os.getpid()}.sock")
            self.holder = H.start(sock)
            os.environ["FASTRSI_HOLDER_SOCK"] = sock
            zygote = True   # the loader hook is installed by the zygote's worker entry
        if zygote:          # every worker group, respawns included, forks from preloaded imports
            from services import zygote as Z
            self.zygote_preload = Z.enable()
        self._spawn()

    # -- process layout ------------------------------------------------------------------
    # -- engine hooks (a toy engine overrides these; see toy_lab.py) ------------------------
    package = "fastvideo"                      # the package fresh_check overlays
    fresh_module = "adapter.fresh"

    def make_generator(self):
        from fastvideo import VideoGenerator
        from fastvideo.api.parser import parse_config
        from fastvideo.api.schema import RunConfig
        force_multiproc()
        return VideoGenerator.from_config(parse_config(RunConfig, self.config).generator)

    def request(self, workload=None):
        return build_request(self.config, workload)

    def output(self, result):
        """(output to compare bit-exactly, frames array or None, extra fields)"""
        out, frames, d = output_of(result)
        return out, frames, {"generation_time": d.get("generation_time")}

    def patches(self):
        from adapter import patches
        return patches.h3_patches({m: self._base(m) for m in (DEC, LAT, INP, patches.NVD)})

    def _spawn(self):
        t0 = time.perf_counter()
        self.gen = self.make_generator()
        self.costs["load_s"] = round(time.perf_counter() - t0, 3)
        self.executor = self.gen.executor
        self.ranks = len(self.executor.workers)
        self.dirty = None                      # a pipe that missed its deadline
        self.base_text = {}
        self.live_text = [{}]                  # module -> text, one entry per stack depth
        ps = [C.CallParticipant(f"rank{i}", self._caller(i)) for i in range(self.ranks)]
        self.co = C.Coordinator(ps, writer=self.writer, interface_version="1-draft",
                                engine={"name": "fastvideo", "version": self._version(),
                                        "commit": os.environ.get("FASTRSI_ENGINE_COMMIT")},
                                phase2_deadline_s=self.rpc_timeout)
        if hasattr(self, "session"):
            self.session.recovered(self.co)
        else:
            self.session = SS.Session(self.co, self.modules)

    @staticmethod
    def _version():
        try:
            from importlib.metadata import version
            return version("fastvideo")
        except Exception:
            return None

    def rank_call(self, i, method, payload=None, timeout=None):
        """agent.rpc(method, payload) on worker i, through its pipe, with a deadline."""
        if self.dirty is not None:
            raise ConnectionError(f"pipe of rank {self.dirty} missed a deadline; respawn")
        w = self.executor.workers[i]
        w.pipe.send({"method": agent.rpc, "args": (method, payload), "kwargs": {}})
        if not w.pipe.poll(timeout or self.rpc_timeout):
            self.dirty = i
            raise TimeoutError(f"rank {i}: no reply to {method} in {timeout or self.rpc_timeout} s")
        r = w.pipe.recv()
        if isinstance(r, dict) and r.get(RPC_ERROR):
            raise RuntimeError(r.get("error"))
        return r

    def _caller(self, i):
        # no lab lock here: the verbs hold it, and the coordinator may call ranks from its
        # lane threads (one per rank, so one pipe is never used by two threads at once)
        def call(method, payload):
            return self.rank_call(i, method, payload)
        return call

    def close(self):
        with self.lock:
            try:
                self.gen.shutdown()
            except Exception:
                pass
            if self.holder is not None:
                from services import holder as H
                try:
                    H._call(os.environ["FASTRSI_HOLDER_SOCK"], {"op": "stop"}, timeout=10)
                except Exception:
                    pass
                self.holder.join(10)
                if self.holder.is_alive():
                    self.holder.kill()

    # -- verbs ---------------------------------------------------------------------------
    def _base(self, m):
        if m not in self.base_text:
            self.base_text[m] = read_module(m)
        return self.base_text[m]

    def install(self, patch):
        t0 = time.perf_counter()
        with self.lock:
            live = self.live_text[-1]
            modules = {}
            try:
                for m, new in patch["files"].items():
                    modules[m] = {"source": new, "base_hash": L.text_hash(live.get(m) or self._base(m))}
            except Exception as e:
                f = F.finding("patch_invalid", f"{type(e).__name__}: {e}", "importable modules")
                return {"ok": False, "id": None, "outcome": "refused", "codes": ["patch_invalid"],
                        "mechanisms": [], "findings": [f], "cost_s": round(time.perf_counter() - t0, 6)}
            plan = {"modules": modules, "accept": patch.get("accept", []),
                    "undo_resets": patch.get("undo_resets", []), "recipes": ["R0 swap"]}
            res = self.session.install(plan, edit_time=patch.get("edit_time"))
            outcome, rec = res.get("outcome") or "refused", res.get("receipt")
            if outcome == "committed":
                self.live_text.append({**live, **patch["files"]})
            codes = (rec["decisions"]["blocking"] if rec else
                     sorted({f["check"] for f in res.get("findings", [])}))
            return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome,
                    "codes": [] if outcome == "committed" else codes,
                    "mechanisms": rec["actual_recipes"] if rec else [],
                    "findings": rec["findings"] if rec else res.get("findings", []),
                    "edit_to_run_s": rec.get("edit_to_run_s") if rec else None,
                    "receipt": rec, "cost_s": round(time.perf_counter() - t0, 6)}

    def _fail(self, check, observed, expected, t0):
        self.session.fault(check, observed, expected)
        return {"ok": False, "error": observed, "output": None, "proven": None, "witness": {},
                "findings": list(self.session.poisoned()),
                "cost_s": round(time.perf_counter() - t0, 6)}

    def check(self, workload=None):
        t0 = time.perf_counter()
        with self.lock:
            for i in range(self.ranks):                 # every rank answers before the run
                try:
                    self.rank_call(i, "ping", None, timeout=10)
                except Exception as e:
                    return self._fail("participant_failed", f"rank{i}: {type(e).__name__}: {e}",
                                      "every rank answers", t0)
            box = {}

            def job():
                try:
                    box["result"] = self.gen.generate(self.request(workload))
                except BaseException as e:
                    box["error"] = f"{type(e).__name__}: {e}"
            th = threading.Thread(target=job, daemon=True, name="fastrsi-check")
            tg = time.perf_counter()
            th.start()
            th.join(self.check_timeout)
            seconds = round(time.perf_counter() - tg, 4)
            if th.is_alive():
                self.dirty = "check"
                return self._fail("participant_failed", f"check: no result in {self.check_timeout} s",
                                  f"the workload finishes within {self.check_timeout} s", t0)
            if "error" in box:
                return self._fail("participant_failed", f"check: {box['error']}",
                                  "the workload runs", t0)
            out, frames, extra = self.output(box["result"])
            golden = {"output": out}
            if self.golden is not None and frames is not None:
                golden["psnr_vs_golden"] = psnr(frames, self.golden)
            proofs = [self.co.prove(t, golden=golden) for t in self.session.stack]
            witness = {}
            for p in proofs:
                for part, counts in p["receipt"]["witness"].items():
                    witness.setdefault(part, {}).update(counts)
            self.last_frames = frames
            return {"ok": True, "output": out, "seconds": seconds, **extra,
                    "psnr_vs_golden": golden.get("psnr_vs_golden"),
                    "proven": all(p["ok"] for p in proofs) if proofs else None,
                    "proven_by_txn": {t: p["ok"] for t, p in zip(self.session.stack, proofs)},
                    "witness": witness,
                    "findings": [f for p in proofs for f in p["receipt"]["findings"]],
                    "cost_s": round(time.perf_counter() - t0, 6)}

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        with self.lock:
            if kind == "respawn":
                try:
                    self.gen.shutdown()
                except Exception:
                    pass
                self._spawn()
                return {"ok": not self.session.poisoned(), "reset": ["worker group", "session"],
                        "cost_s": round(time.perf_counter() - t0, 6)}
            res = [C._call(p, "reset", "R3") for p in self.co.participants]
            return {"ok": all(r["ok"] for r in res), "reset": ["cuda cache"],
                    "cost_s": round(time.perf_counter() - t0, 6)}

    def rebuild(self, patch):
        """In-place rebuild: every warm worker re-imports the edited modules from `patch["files"]`
        and rebuilds what was built from them (adapter/agent.py `rebuild`): for the edits the
        R0 swap refuses (constructors, new fields, module constants, new classes). The edited
        text becomes the new base: the session starts over (no live edits may be installed).
        `{"revert": True}` rebuilds every module rebuilt so far back to its file on disk."""
        t0 = time.perf_counter()
        with self.lock:
            refuse = None
            if self.session.poisoned():
                refuse = finding("session_poisoned", "the lab is poisoned", "a known state: respawn")
            elif self.session.stack:
                refuse = finding("live_edits_installed", f"{len(self.session.stack)} transaction(s)",
                                   "uninstall the live edits before a rebuild")
            revert = bool(patch.get("revert"))
            files = ({m: read_module(m) for m in getattr(self, "rebuilt", {})} if revert
                     else dict(patch.get("files") or {}))
            if not files and refuse is None:
                refuse = finding("patch_invalid", "no files", "at least one edited module")
            if refuse is not None:
                return {"ok": False, "outcome": "refused", "findings": [refuse], "ranks": {},
                        "cost_s": round(time.perf_counter() - t0, 6)}
            ranks = {}

            def one(i):                     # every rank at once: each has its own pipe
                try:
                    ranks[f"rank{i}"] = self.rank_call(i, "rebuild", {"files": files})
                except Exception as e:
                    ranks[f"rank{i}"] = {"ok": False, "findings": [finding(
                        "participant_failed", f"{type(e).__name__}: {e}", "the rank answers")]}
            lanes = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(self.ranks)]
            for th in lanes:
                th.start()
            for th in lanes:
                th.join()
            ranks = {k: ranks[k] for k in sorted(ranks, key=lambda k: int(k[4:]))}
            oks = [r.get("ok") for r in ranks.values()]
            findings = [f for r in ranks.values() for f in r.get("findings", [])]
            if any(oks) and not all(oks):              # ranks now run different code
                self.session.fault("participant_failed", "rebuild applied on some ranks only",
                                   "every rank rebuilt, or none")
                outcome = "partial"
            elif all(oks):
                outcome = "rebuilt"
                self.rebuilt = {} if revert else {**getattr(self, "rebuilt", {}), **files}
                for m, text in files.items():
                    self.base_text[m] = text
                self.live_text = [{}]
                self.session.recovered(self.co)        # the edited text is the new base
            else:
                outcome = "refused"
            et = patch.get("edit_time")
            return {"ok": outcome == "rebuilt", "outcome": outcome, "findings": findings,
                    "ranks": ranks, "edit_to_rebuilt_s": round(time.time() - et, 4) if et else None,
                    "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        t0 = time.perf_counter()
        with self.lock:
            r = self.session.uninstall(id)
            n = len(r.get("undone", []))
            if n:
                del self.live_text[-n:]
            return {"ok": r["ok"], "undone": r.get("undone", []), "state": self.session.state(),
                    "findings": r.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        with self.lock:
            return self.session.status()

    def repl(self, code, rank=None):
        """Run a cell on every rank (or one): the notebook verb."""
        with self.lock:
            ranks = range(self.ranks) if rank is None else [rank]
            out = {}
            for i in ranks:
                try:
                    out[f"rank{i}"] = self.rank_call(i, "repl", code)
                except Exception as e:
                    out[f"rank{i}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            return {"ok": all(v.get("ok") for v in out.values()), "ranks": out}

    # -- conformance harness -------------------------------------------------------------
    def fresh_check(self, patch, workload=None):
        """The workload in a new process (its own workers) started from base files plus patch:
        a symlink farm of the engine package with the patched modules written in."""
        pkg = importlib.import_module(self.package)
        root = os.path.dirname(os.path.dirname(pkg.__file__))
        tmp = tempfile.mkdtemp(prefix="fvlab-fresh-")
        try:
            for d, dirs, files in os.walk(os.path.join(root, self.package)):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                rel = os.path.relpath(d, root)
                os.makedirs(os.path.join(tmp, rel), exist_ok=True)
                for f in files:
                    os.symlink(os.path.join(d, f), os.path.join(tmp, rel, f))
            for m, text in ((patch or {}).get("files") or {}).items():
                p = os.path.join(tmp, *m.split(".")) + ".py"
                os.unlink(p)
                with open(p, "w") as fh:
                    fh.write(text)
            spec = os.path.join(tmp, "fresh.json")
            with open(spec, "w") as fh:
                json.dump({"config": self.config, "workload": workload}, fh)
            env = {**os.environ, "PYTHONPATH": tmp + os.pathsep + os.environ.get("PYTHONPATH", "")}
            r = subprocess.run([sys.executable, "-m", self.fresh_module, spec], env=env,
                               capture_output=True, text=True, timeout=self.check_timeout + 600,
                               cwd=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        lines = [x for x in r.stdout.splitlines() if x.startswith("FRESH ")]
        if r.returncode or not lines:
            return {"ok": False, "error": r.stderr.strip().splitlines()[-3:], "output": None}
        return {"ok": True, **json.loads(lines[-1][6:])}

    def inject(self, fault, **kw):
        if fault == "hang":                     # must outlast the check's deadline
            kw.setdefault("hang_s", self.check_timeout + 30)
        rank = kw.pop("participant", self.ranks - 1)
        if isinstance(rank, str):
            rank = int(rank.replace("rank", ""))
        with self.lock:
            try:
                return self.rank_call(rank, "inject", {"fault": fault, **kw})
            except Exception as e:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
