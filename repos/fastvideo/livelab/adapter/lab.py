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
import json
import os
import threading
import time

from fastrsi import harness, ipc, lab as lab_base
from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L
from adapter import agent, diffs

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


module_file = harness.module_file
read_module = harness.read_module


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


class FastVideoLab(lab_base.BaseLab):
    """`config`: a FastVideo run config (dict, `fastvideo generate --config` schema).
    `receipts`: receipt file (JSON lines). `golden`: path of a uint8 frames .npy.
    `zygote`: fork the workers from a preloaded forkserver (services/zygote.py), so a
    `reset("respawn")` skips the workers' imports (zygote respawn). `holder`: also keep the loaded DiT, text
    encoder and VAE in a holder process (services/holder.py) that respawned workers map by CUDA
    IPC instead of loading; implies `zygote`."""
    name = "fastvideo"
    threaded_check = True
    package = "fastvideo"                      # the package fresh_check overlays
    fresh_module = "adapter.fresh"

    def __init__(self, config, receipts=None, golden=None, modules=MODULES, rpc_timeout=60.0,
                 check_timeout=600.0, zygote=False, holder=False):
        self.config = json.loads(json.dumps(config))
        self.golden = None
        if golden and os.path.exists(golden):
            import numpy as np
            self.golden = np.load(golden)
        self.holder = self.holder_sock = None
        self._pending_files = {}
        self.rebuilt = {}
        super().__init__(receipts=receipts, check_timeout=check_timeout,
                         rpc_timeout=rpc_timeout, auto_spawn=False)
        self.modules = list(modules)
        if holder:          # loaded DiT, text encoder and VAE outlive the workers (services/holder.py)
            from services import holder as H
            sock = ipc.short_sock_path("fastrsi-holder")
            self.holder = H.start(sock)
            self.holder_sock = sock
            # the workers' loader hook reads it (forked from the zygote, which inherits the env)
            os.environ["FASTRSI_HOLDER_SOCK"] = sock
            zygote = True   # the loader hook is installed by the zygote's worker entry
        if zygote:          # every worker group, respawns included, forks from preloaded imports
            from services import zygote as Z
            self.costs["zygote_preload"] = Z.enable()
        self._spawn()

    # -- engine hooks (a toy engine overrides these; see toy_lab.py) ------------------------
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

    def _start_engine(self):
        self.gen = self.make_generator()
        self.executor = self.gen.executor
        self.ranks = len(self.executor.workers)
        self.dirty = None                      # a pipe that missed its deadline
        self.base_text = {}
        self.rebuilt = {}
        self._pending_files = {}
        ps = [C.CallParticipant(f"rank{i}", self._caller(i)) for i in range(self.ranks)]
        co = C.Coordinator(ps, writer=self.writer, interface_version="1-draft",
                           engine={"name": "fastvideo", "version": self._version(),
                                   "commit": os.environ.get("FASTRSI_ENGINE_COMMIT")},
                           phase2_deadline_s=self.rpc_timeout)
        return co, self.modules

    def _stop_engine(self):
        self.gen.shutdown()

    @staticmethod
    def _version():
        try:
            from importlib.metadata import version
            return version("fastvideo")
        except Exception:
            return None

    def rank_call(self, i, method, payload=None, timeout=None):
        """agent.rpc(method, payload) on worker i, through its pipe, with a deadline."""
        if self.dirty is not None or self._dirty:
            who = self.dirty if self.dirty is not None else "check"
            raise ConnectionError(f"pipe of rank {who} missed a deadline; respawn")
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
            super().close()
            if self.holder is not None:
                from services import holder as H
                try:
                    H.call(self.holder_sock, {"op": "stop"}, timeout=10)
                except Exception:
                    pass
                self.holder.join(10)
                if self.holder.is_alive():
                    self.holder.kill()
                self.holder = None
                for path in (self.holder_sock, self.holder_sock + ".ready"):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                if os.environ.get("FASTRSI_HOLDER_SOCK") == self.holder_sock:
                    del os.environ["FASTRSI_HOLDER_SOCK"]

    # -- BaseLab hooks + FastVideo-specific verbs ----------------------------------------
    def _base(self, m):
        return self._base_source(m)

    def _exists(self, m):
        try:
            return m in self.base_text or module_file(m) is not None
        except ImportError:
            return False

    def files(self, patch, against="live"):
        """{module: new text} of a patch: its `files`, or its `diff` (a unified diff, such as a
        `patches/` file) applied to the live text (base plus every live edit and rebuild), or to
        the files on disk with `against="disk"` (what a fresh process starts from). ValueError
        when the diff does not apply."""
        if patch.get("diff") is None:
            return dict(patch.get("files") or {})
        if against == "disk":
            return diffs.apply(patch["diff"], lambda m: read_module(m) if self._exists(m) else None)
        live = self.live_text[-1]
        return diffs.apply(patch["diff"], lambda m: live.get(m) or (
            self._base(m) if self._exists(m) else None))

    def _build_plan(self, patch, live_map):
        files = self.files(patch)
        self._pending_files = files
        modules = {
            m: {"source": new, "base_hash": L.text_hash(live_map.get(m) or self._base(m))}
            for m, new in files.items()
        }
        plan = {"modules": modules, "accept": patch.get("accept", []),
                "undo_resets": patch.get("undo_resets", []), "recipes": ["R0 swap"]}
        return plan, []

    def _after_commit(self, patch, receipt):
        if self._pending_files:
            self.live_text[-1].update(self._pending_files)
            self._pending_files = {}

    def _probe_participants(self):
        for i in range(self.ranks):
            try:
                self.rank_call(i, "ping", None, timeout=10)
            except Exception as exc:
                return False, f"rank{i}: {type(exc).__name__}: {exc}"
        return True, None

    def _run_workload(self, workload):
        return self.gen.generate(self.request(workload))

    def _format_output(self, raw):
        out, frames, extra = self.output(raw)
        self.last_frames = frames
        extra = dict(extra)
        extra["psnr_vs_golden"] = (
            psnr(frames, self.golden) if self.golden is not None and frames is not None else None
        )
        return out, extra

    def _golden_for(self, output, extra):
        golden = {"output": output}
        if extra.get("psnr_vs_golden") is not None:
            golden["psnr_vs_golden"] = extra["psnr_vs_golden"]
        return golden

    def _reset_state(self):
        res = [C._call(p, "reset", "R3") for p in self.co.participants]  # "R3": CUDA cache
        return {"ok": all(r["ok"] for r in res), "reset": ["cuda cache"]}

    def rebuild(self, patch):
        """In-place rebuild: every warm worker re-imports the edited modules from `patch["files"]`
        and rebuilds what was built from them (adapter/agent.py `rebuild`): for the edits the
        live swap refuses (constructors, new fields, module constants, new classes). The edited
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
            files = {}
            if revert:
                files = {m: read_module(m) for m in getattr(self, "rebuilt", {})}
            elif refuse is None:
                try:
                    files = self.files(patch)
                except Exception as e:                  # a verb never raises
                    refuse = finding("patch_invalid", f"{type(e).__name__}: {e}",
                                     "a diff that applies to the live text")
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
            unknown = [rank for rank, reply in ranks.items()
                       if not reply.get("ok") and not reply.get("clean")]
            if any(oks) and not all(oks):              # ranks now run different code
                self.session.fault("participant_failed", "rebuild applied on some ranks only",
                                   "every rank rebuilt, or none")
                outcome = "partial"
            elif unknown:                              # a refusal that may have left changes
                self.session.fault("participant_failed",
                                   f"{unknown}: rebuild failed without saying nothing changed",
                                   "every rank refused with nothing changed")
                outcome = "failed"
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
        a symlink farm of the engine package with the patched modules written in (a package's
        `__init__.py`, or a new file). Never raises."""
        try:
            overrides = self.files(patch or {}, against="disk")
            with harness.PackageOverlay(self.package, overrides) as ov:
                return harness.run_fresh_subprocess(
                    self.fresh_module,
                    {"config": self.config, "workload": workload},
                    ov.root,
                    timeout=self.check_timeout + 600,
                )
        except Exception as error:
            return {"ok": False, "error": [f"fresh_check: {type(error).__name__}: {error}"], "output": None}

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
