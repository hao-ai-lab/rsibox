"""VerlConformanceLab: `LabUnderTest` over a *real* gated verl job (4 GPUs, 13+ Ray processes).

The lab is a client of the gate protocol (`gate.py` via `adapter/client.py`): every verb is one
request file on the gate directory, served by the trainer at its step gap. It owns the job's
lifetime through a launch command, so `reset("respawn")` and `fresh_check` really restart the job.

Environment (or constructor arguments / `fastrsi conformance verl --real -- ...`):
  FASTRSI_VERL_GATE           gate directory of the live job (shared with the trainer)
  FASTRSI_VERL_LAUNCH         shell command that starts a gated job and exits when it exits; it reads
                              `$GATE` (gate dir) and `$REWARD` (custom reward file) and inherits
                              `$PYTHONPATH` (a fresh process with a patched module gets an overlay first)
  FASTRSI_VERL_REWARD         the custom reward file the job loads (patched by the conformance edits)
  FASTRSI_VERL_CHECK_TIMEOUT  seconds one check may take (default 300; also the hang budget)
  FASTRSI_VERL_BOOT_TIMEOUT   seconds a (re)launch may take to `ready.json` (default 1800)

Output of a check: `{"steps": [{"score_mean", "adv_max", "adv_min", "pg_loss", "rollouts"}]}` per
RL step after a coupled S0 reset; `rollouts` is a digest of the step's dumped (prompt, response,
score, ground truth) rows; `pg_loss` is rounded to 1e-6 (see `ROUND`). Bit-exact repeats need
`rollout.full_determinism=True` and the per-sample seeds of `determinism.py`
(`FASTRSI_RL_SEED_PER_SAMPLE=1`).

Run via the unified CLI:
    python3 -m fastrsi conformance verl --real -- --gate G --launch CMD --reward FILE
or `python3 -m tools.cli conformance --gate G --launch CMD --reward FILE` from `repos/verl`.
"""
import argparse
import atexit
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from adapter import client, gate as G
from fastrsi import harness, ipc, proc
from kernel import findings as F

CORE = "verl.trainer.ppo.core_algos"
METRICS = {"score_mean": "critic/score/mean", "adv_max": "critic/advantages/max",
           "adv_min": "critic/advantages/min", "pg_loss": "actor/pg_loss"}
# pg_loss is bit-exact across resets inside one job but differs at 1e-9 between two processes (FSDP loss
# reduction order), so it is compared at 1e-6; scores, advantages and the rollout digest are exact.
ROUND = {"pg_loss": 6}
_JOBS = {}  # gate dir -> Popen of the launcher this process started
_edit = harness.edit_once
_ORPHAN_PKILL_PATTERN = r"ray::|EngineCore|vLLMHttpServer|verl\.trainer\.main_ppo"


def _digest_rollouts(path):
    if not path or not os.path.isfile(path):
        return None
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(json.dumps([row.get("input"), row.get("output"), row.get("score"), row.get("gts")], sort_keys=True))
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()[:16]


class VerlConformanceLab:
    name = "verl"

    @staticmethod
    def launch_environment(argv):
        """Translate `fastrsi conformance verl --real -- ...` flags into `FASTRSI_VERL_*` env vars."""
        clean = list(argv)[1:] if list(argv)[:1] == ["--"] else list(argv)
        ap = argparse.ArgumentParser(prog="fastrsi conformance verl --real", allow_abbrev=False)
        ap.add_argument("--gate", required=True, help="gate directory shared with the gated verl job")
        ap.add_argument("--reward", required=True, help="custom reward .py file the job loads")
        ap.add_argument("--launch", default=None, help="shell command starting the gated job on $GATE with $REWARD")
        ap.add_argument("--check-timeout", type=float, default=None, help="seconds one check step may take")
        ap.add_argument("--boot-timeout", type=float, default=None, help="seconds a (re)launch may take to ready.json")
        ns = ap.parse_args(clean)
        env = {"FASTRSI_VERL_GATE": str(Path(ns.gate).resolve()), "FASTRSI_VERL_REWARD": os.path.realpath(ns.reward)}
        for attr, key in (("launch", "FASTRSI_VERL_LAUNCH"), ("check_timeout", "FASTRSI_VERL_CHECK_TIMEOUT"),
                          ("boot_timeout", "FASTRSI_VERL_BOOT_TIMEOUT")):
            if (val := getattr(ns, attr)) is not None:
                env[key] = str(val)
        return env

    def __init__(self, receipts=None, gate=None, launch=None, reward=None, check_timeout=None, boot_timeout=None):
        env = os.environ
        self.gate = Path(gate or env["FASTRSI_VERL_GATE"]).resolve()
        self.launch = launch or env.get("FASTRSI_VERL_LAUNCH")
        self.reward = os.path.realpath(reward or env["FASTRSI_VERL_REWARD"])
        self.check_timeout = float(check_timeout or env.get("FASTRSI_VERL_CHECK_TIMEOUT", 300))
        self.boot_timeout = float(boot_timeout or env.get("FASTRSI_VERL_BOOT_TIMEOUT", 1800))
        self.receipts, self.poison, self.stack_patches, self.costs = receipts, [], [], {}
        self._live_stack = [{}]
        self._receipts_offset = self._receipt_size()
        self._ensure_job()

    # -- the job -------------------------------------------------------------------------
    def _receipt_size(self):
        p = self.gate / "receipts.jsonl"
        return p.stat().st_size if p.exists() else 0

    def _info(self, gate=None):
        target = Path(gate) if gate is not None else self.gate
        data = ipc.read_json(target / "gate.json")
        return data if isinstance(data, dict) else {}

    def _alive(self, gate=None):
        pid = self._info(gate).get("pid")
        return isinstance(pid, int) and not isinstance(pid, bool) and proc.pid_alive(pid)

    def _reap_job(self, gate, timeout_s=5.0):
        gate_key = str(gate)
        job = _JOBS.get(gate_key)
        if job is None:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            job.wait(timeout=timeout_s)
        if job.poll() is not None:
            _JOBS.pop(gate_key, None)

    def _ensure_job(self):
        if (self.gate / "ready.json").exists() and self._alive():
            return
        if not self.launch:
            raise RuntimeError(f"{self.gate}: no live gated job and no FASTRSI_VERL_LAUNCH to start one")
        self._start_job(self.gate)

    def _start_job(self, gate, reward=None, pythonpath=None):
        gate.mkdir(parents=True, exist_ok=True)
        for stale in gate.glob("*.json"):
            stale.unlink()
        env = {**os.environ, "GATE": str(gate), "REWARD": reward or self.reward}
        if pythonpath:
            env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t0 = time.time()
        with open(gate.parent / f"{gate.name}-launch.log", "a", encoding="utf-8") as log:
            _JOBS[str(gate)] = subprocess.Popen(
                self.launch, shell=True, env=env, start_new_session=True,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        self._wait_ready(gate, t0)
        self.costs["boot_s"] = round(time.time() - t0, 3)
        return self.costs["boot_s"]

    def _wait_ready(self, gate, since):
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            data = ipc.read_json(gate / "ready.json")
            if isinstance(data, dict) and data.get("t", 0) >= since:
                return
            job = _JOBS.get(str(gate))
            if job is not None and job.poll() is not None:
                rc = job.returncode
                _JOBS.pop(str(gate), None)
                raise RuntimeError(
                    f"{gate}: the launch command exited with {rc} before the gate was ready; "
                    f"last errors in {gate.name}-launch.log: {self._launch_errors(gate)}"
                )
            time.sleep(0.5)
        raise RuntimeError(f"{gate}: no ready.json within {self.boot_timeout}s")

    def _launch_errors(self, gate, n=3):
        try:
            lines = (gate.parent / f"{gate.name}-launch.log").read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        return [ln[-200:] for ln in lines if "Error" in ln and "UserWarning" not in ln][-n:]

    def _wait_gone(self, gate, timeout_s):
        deadline, job = time.monotonic() + timeout_s, _JOBS.get(str(gate))
        while time.monotonic() < deadline:
            if not self._alive(gate) and (job is None or job.poll() is not None):
                return True
            time.sleep(0.5)
        return False

    def _gpus_busy(self):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        cmd = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        if vis is not None:
            vis = vis.strip()
            if not vis or vis.lower() in ("none", "void", "nodevfiles"):
                return []
            if vis.lower() != "all":
                cmd[1:1] = ["-i", vis]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            return [p for p in out.split() if p.strip()]
        except (OSError, subprocess.SubprocessError):
            return []

    def _pkill_orphans(self):
        # Fallback for orphaned Ray/vLLM workers that reparented out of the launcher tree.
        subprocess.run(["pkill", "-9", "-u", os.environ.get("USER", ""), "-f", _ORPHAN_PKILL_PATTERN], check=False)
        time.sleep(3.0)

    def _wait_gpus_free(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while self._gpus_busy() and time.monotonic() < deadline:
            time.sleep(1.0)
        if self._gpus_busy():
            self._pkill_orphans()

    def _kill_job(self, gate):
        job, pid = _JOBS.get(str(gate)), self._info(gate).get("pid")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if job is not None and job.poll() is None:
                proc.kill_tree(job.pid, sig, include_pg=True)
            if isinstance(pid, int) and not isinstance(pid, bool):
                proc.kill_tree(pid, sig, include_pg=False)
            if self._wait_gone(gate, 20):
                break
        self._reap_job(gate)
        self._wait_gpus_free(90)
        self._reap_job(gate)

    def _stop_job(self, gate=None):
        gate = Path(gate) if gate is not None else self.gate
        t0, killed = time.time(), False
        if self._alive(gate):
            with contextlib.suppress(Exception):
                client.request(gate, stop=True, timeout=60.0)
            if not self._wait_gone(gate, 120):
                self._kill_job(gate)
                killed = True
        job = _JOBS.get(str(gate))
        if not killed and (self._alive(gate) or (job is not None and job.poll() is None)):
            self._kill_job(gate)
            killed = True
        self._reap_job(gate)
        if not killed:
            self._wait_gpus_free(60)
        self.costs["stop_s"] = round(time.time() - t0, 3)

    # -- requests & patch stack ----------------------------------------------------------
    def _request(self, **kw):
        """One gate request, mirrored to `<gate>-client.jsonl` (a relaunch wipes the gate's own files)."""
        t0 = time.time()
        try:
            resp = client.request(self.gate, **kw)
        except client.GateError as err:
            self._log({"t": t0, "req": _jsonable(kw), "error": str(err), "s": round(time.time() - t0, 3)})
            raise
        self._log({"t": t0, "req": _jsonable(kw),
                   "resp": {k: v for k, v in resp.items() if k not in ("req", "steps_run")},
                   "s": round(time.time() - t0, 3)})
        return resp

    def _log(self, record):
        harness.append_jsonl(self.gate.parent / f"{self.gate.name}-client.jsonl", record, suppress_errors=True)

    def _fault(self, check, observed, expected):
        f = F.finding(check, observed, expected)
        self.poison.append(f)
        return f

    def _refused(self, t0, what="install"):
        finding = F.finding("session_poisoned", "; ".join(f["check"] for f in self.poison), "a session in a known state")
        return harness.install_refusal(["session_poisoned"], [finding], t0=t0)

    def _fail_check(self, error, t0, *, fault_observed=None, fault_expected=None):
        if fault_expected is not None:
            self._fault("participant_failed", fault_observed if fault_observed is not None else error, fault_expected)
        return harness.check_failure(error, list(self.poison), t0=t0)

    def _output(self, resp):
        steps = []
        for st in resp.get("steps_run") or []:
            m = st.get("metrics") or {}
            vals = {k: m.get(v) for k, v in METRICS.items()}
            steps.append({**{k: (round(x, ROUND[k]) if k in ROUND and isinstance(x, float) else x) for k, x in vals.items()},
                          "rollouts": _digest_rollouts(st.get("rollouts"))})
        return {"steps": steps}

    def _disk_source(self, mod):
        path = G.path_for_key(mod)
        return Path(path).read_text(encoding="utf-8") if path else None

    def _sources(self):
        return harness.SourceStack(self._disk_source, getattr(self, "_live_stack", [{}]), min_len=1)

    def _live_source(self, mod):
        return self._sources().live_source(mod)

    def files(self, patch, against="live"):
        """Return `{module_or_path: text}` for a `{"files": ...}` or `{"diff": ...}` patch."""
        return self._sources().resolve(patch, against=against)

    # -- the five verbs -------------------------------------------------------------------
    def install(self, patch):
        t0 = time.perf_counter()
        if self.poison:
            return self._refused(t0)
        res = self._install(patch)
        if res["ok"]:
            self.stack_patches.append(patch)
        return {**res, "cost_s": round(time.perf_counter() - t0, 6)}

    def _install(self, patch):
        try:
            resolved_files = self.files(patch, against="live")
        except ValueError as err:
            return harness.install_refusal(["patch_invalid"], [F.finding("patch_invalid", str(err), "a valid patch")])
        try:
            r = self._request(
                verb="install", steps=0, timeout=self.check_timeout,
                extra={"sources": resolved_files, "accept": patch.get("accept", []),
                       "undo_resets": patch.get("undo_resets", [])},
            )
        except client.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return harness.install_refusal(["participant_failed"], list(self.poison), outcome="aborted")
        outcome = r.get("outcome") or ("committed" if r.get("ok") else "refused")
        codes = [] if outcome == "committed" else (r.get("codes") or sorted({f["check"] for f in r.get("findings", [])}) or ["refused"])
        if outcome == "committed" and hasattr(self, "_live_stack"):
            self._sources().push(dict(resolved_files))
        if r.get("restart_required"):
            self._fault("session_poisoned", r.get("error") or "gate session poisoned", "a session in a known state")
        return {"ok": outcome == "committed", "id": r.get("txn"), "outcome": outcome, "codes": codes,
                "mechanisms": r.get("mechanisms", []), "findings": r.get("findings", []),
                "install_s": r.get("install_s"), "receipt": r}

    def check(self, workload=None):
        t0 = time.perf_counter()
        if self.poison:
            return self._fail_check("session poisoned", t0)
        steps = int((workload or {}).get("steps", 1))
        try:
            rs = self._request(verb="reset", steps=0, timeout=self.check_timeout)
            if not rs.get("ok"):
                return self._fail_check(rs.get("error"), t0, fault_observed=f"S0 restore failed: {rs.get('error')}",
                                        fault_expected="coupled S0 restores")
            r = self._request(verb="check", steps=steps, timeout=self.check_timeout * steps)
        except client.GateError as err:
            return self._fail_check(str(err), t0, fault_expected="the RL step completes in time")
        if not r.get("steps_run"):
            return self._fail_check(r.get("error") or "no step ran", t0, fault_expected="the RL step completes")
        live = bool(self._request(verb="status", steps=0, timeout=60.0).get("txns")) if "proven" not in r else True
        return {"ok": True, "output": self._output(r), "seconds": r.get("verdict_s"),
                "proven": r.get("proven") if live else None, "witness": r.get("witness") or {},
                "findings": r.get("findings") or [], "reset_s": rs.get("cost_s"),
                "cost_s": round(time.perf_counter() - t0, 6),
                **({"abstained": r["abstained"]} if r.get("abstained") else {})}

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        if kind == "respawn":
            if self._alive() and not self.poison:
                self._stop_job()
            else:
                self._kill_job(self.gate)
            self.poison, self.stack_patches = [], []
            self._sources().clear()
            try:
                boot = self._start_job(self.gate)
            except RuntimeError as err:
                return {"ok": False, "error": str(err), "reset": [], "cost_s": round(time.perf_counter() - t0, 6)}
            return {"ok": True, "reset": ["ray_job", "coupled_s0", "session"], "boot_s": boot,
                    "cost_s": round(time.perf_counter() - t0, 6)}
        try:
            r = self._request(verb="reset", steps=0, timeout=self.check_timeout)
        except client.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return {"ok": False, "error": str(err), "reset": [], "cost_s": round(time.perf_counter() - t0, 6)}
        return {"ok": bool(r.get("ok")), "error": r.get("error"), "reset": r.get("reset", []),
                "digests": r.get("digests"), "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        t0 = time.perf_counter()
        try:
            r = self._request(verb="uninstall", steps=0, timeout=self.check_timeout, extra={"id": id} if id else None)
        except client.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return {"ok": False, "undone": [], "state": "poisoned", "findings": list(self.poison),
                    "cost_s": round(time.perf_counter() - t0, 6)}
        if r.get("undone"):
            n = len(r["undone"])
            del self.stack_patches[-n:]
            if hasattr(self, "_live_stack"):
                self._sources().pop(n)
        return {"ok": bool(r.get("ok")), "undone": r.get("undone", []), "state": r.get("state"),
                "findings": r.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        try:
            r = self._request(verb="status", steps=0, timeout=60.0) if not self.poison else {}
        except client.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            r = {}
        poison = list(self.poison) + list(r.get("poison") or [])
        state = "poisoned" if poison else r.get("state", "poisoned")
        return {"ok": state != "poisoned", "state": state, "txns": r.get("txns", []),
                "fingerprint": r.get("fingerprint"), "poison": poison}

    # -- the harness ---------------------------------------------------------------------
    def _base(self, key):
        return Path(G.path_for_key(key)).read_text(encoding="utf-8")

    def patches(self):
        core, rew = self._base(CORE), self._base(self.reward)
        norm = "                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)\n"
        x2, x3 = _edit(core, norm, norm[:-1] + " * 2.0\n"), _edit(core, norm, norm[:-1] + " * 3.0\n")
        reward_x2 = _edit(rew, "return SCORE if answer == str(ground_truth) else FORMAT_SCORE",
                          "return SCORE * 2 if answer == str(ground_truth) else FORMAT_SCORE")
        agg_sig = ("def agg_loss(\n    loss_mat: torch.Tensor,\n    loss_mask: torch.Tensor,\n"
                   "    loss_agg_mode: str,\n    dp_size: int = 1,\n    batch_num_tokens: Optional[int] = None,\n"
                   "    global_batch_size: Optional[int] = None,\n    loss_scale_factor: Optional[int] = None,\n):\n")
        multi = _edit(
            _edit(x2, agg_sig, agg_sig + "    loss_agg_mode = str(loss_agg_mode)\n"),
            '    # Strip the optional \'+\' suffix so e.g. "k3+" dispatches to "k3".\n',
            '    kl_penalty = str(kl_penalty)\n    # Strip the optional \'+\' suffix so e.g. "k3+" dispatches to "k3".\n',
        )
        return {
            "noop": {"files": {CORE: _edit(core, norm, norm[:-1] + " * 1.0\n")}},
            "visible": {"files": {CORE: x2}},
            "chain": [{"files": {CORE: x2}}, {"files": {CORE: x3}}, {"files": {self.reward: reward_x2}}],
            "multi": {"files": {CORE: multi, self.reward: reward_x2}},
            "unreached": {"files": {self.reward: _edit(rew, "    return 0.0\n\n\nFORMAT_SCORE = format_score()",
                                                       "    return 0.5\n\n\nFORMAT_SCORE = format_score()")}},
            "refused": {
                "syntax": {"files": {CORE: _edit(core, norm, norm.replace("scores[i] = (", "scores[i] = = ("))}},
                "closure": {"files": {self.reward: _edit(rew, "checks = [lambda m: m is not None]",
                                                         "checks = [lambda m: m is not None and checks]")}},
                "module_state": {"files": {self.reward: _edit(rew, "SCORE = 1.0\n", "SCORE = 2.0\n")}},
                "empty": {"files": {self.reward: _edit(
                    rew, 'def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):',
                    'def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):  # note')}},
            },
        }

    def _overlay(self, tmp, files):
        overlay, reward, by_pkg = None, None, {}
        for key, text in files.items():
            path = G.path_for_key(key)
            if path == self.reward:
                reward = os.path.join(tmp, os.path.basename(path))
                Path(reward).write_text(text, encoding="utf-8")
            elif os.path.isfile(key):
                raise ValueError(f"{key}: a fresh process can only take the job's reward file or an importable module")
            else:
                by_pkg.setdefault(key.split(".")[0], {})[key] = text
        if by_pkg:
            overlay = os.path.join(tmp, "overlay")
            for top, overrides in by_pkg.items():
                harness.build_package_overlay(top, overrides, dst=overlay)
        return overlay, reward

    def fresh_check(self, patch, workload=None):
        steps = int((workload or {}).get("steps", 1))
        if not self.launch:
            return {"ok": False, "error": "no FASTRSI_VERL_LAUNCH: cannot start a fresh job", "output": None}
        try:
            files = self.files(patch, against="disk")
        except ValueError as err:
            return {"ok": False, "error": str(err), "output": None}
        t0, tmp = time.perf_counter(), tempfile.mkdtemp(prefix="verllab-fresh-")
        try:
            self._stop_job()
            if files:
                overlay, reward = self._overlay(tmp, files)
                fresh = self.gate.with_name(self.gate.name + "-fresh")
                self._start_job(fresh, reward=reward, pythonpath=overlay)
                try:
                    out = self._output(client.request(fresh, verb="check", steps=steps, timeout=self.check_timeout * steps))
                finally:
                    self._stop_job(fresh)
                self._start_job(self.gate)
            else:  # a fresh base: start it on the live gate and keep it as the live job
                self._start_job(self.gate)
                out = self._output(client.request(self.gate, verb="check", steps=steps, timeout=self.check_timeout * steps))
            self._replay()
            return {"ok": True, "output": out, "cost_s": round(time.perf_counter() - t0, 6)}
        except Exception as err:
            self._fault("participant_failed", f"fresh_check: {type(err).__name__}: {err}", "a fresh job boots and steps")
            return {"ok": False, "error": f"{type(err).__name__}: {err}", "output": None}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _replay(self):
        """Re-install the live edits a relaunch dropped, so the lab comes back in the state it left."""
        patches, self.stack_patches = self.stack_patches, []
        self._sources().clear()
        for p in patches:
            res = self._install(p)
            if not res["ok"]:
                label = sorted(p.get("files") or ()) if "files" in p else ["<diff>"]
                self._fault("undo_incomplete", f"replaying {label} after relaunch: {res['outcome']} {res['codes']}",
                            "the live edits install again")
                return
            self.stack_patches.append(p)

    def inject(self, fault, **kw):
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 5.0)
        participant = kw.pop("participant", "reward0" if fault in ("fail_commit", "lost") else "driver")
        try:
            return self._request(verb="inject", steps=0, timeout=60.0,
                                 extra={"fault": fault, "participant": participant, "kwargs": kw})
        except client.GateError as err:
            return {"ok": False, "error": str(err)}

    def close(self):
        if not self.poison and self._alive():
            with contextlib.suppress(Exception):
                st = self._request(verb="status", steps=0, timeout=60.0)
                if st.get("txns"):
                    self._request(verb="uninstall", steps=0, timeout=self.check_timeout, extra={"id": st["txns"][0]})
                self._request(verb="reset", steps=0, timeout=self.check_timeout)
        if self.receipts:
            with contextlib.suppress(Exception):
                src = self.gate / "receipts.jsonl"
                if src.exists():
                    with open(src, "rb") as fin, open(self.receipts, "ab") as fout:
                        fin.seek(self._receipts_offset)
                        fout.write(fin.read())


def _jsonable(kw):
    """The request without the full source texts (their keys and sizes instead)."""
    out = dict(kw)
    extra = dict(out.get("extra") or {})
    if extra.get("sources"):
        extra["sources"] = {k: f"<{len(v)} chars>" for k, v in extra["sources"].items()}
    if extra:
        out["extra"] = extra
    return out


@atexit.register
def _cleanup_jobs():
    """Stop and reap every launcher process this process spawned (never touches external jobs)."""
    if not _JOBS:
        return
    lab = VerlConformanceLab.__new__(VerlConformanceLab)
    lab.costs = {}
    for gate_str in list(_JOBS):
        gate = Path(gate_str)
        try:
            with contextlib.suppress(Exception):
                lab._stop_job(gate)
        finally:
            job = _JOBS.pop(gate_str, None)
            if job is not None and job.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill_tree(job.pid, signal.SIGKILL, include_pg=True)
                with contextlib.suppress(Exception):
                    job.wait(timeout=5.0)
