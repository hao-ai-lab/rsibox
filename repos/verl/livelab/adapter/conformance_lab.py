"""VerlConformanceLab: `LabUnderTest` over a *real* gated verl job (4 GPUs, 13+ Ray processes).

The lab is a client of the gate protocol (`gate.py`): every verb is one request file on the
gate directory, served by the trainer at its step gap. It owns the job's lifetime through a
launch command, so `reset("respawn")` and `fresh_check` really restart the job.

Environment (or constructor arguments):
  FASTRSI_VERL_GATE      gate directory of the live job (shared with the trainer)
  FASTRSI_VERL_LAUNCH    shell command that starts a gated job and exits when it exits; it reads
                         `$GATE` (gate dir) and `$REWARD` (custom reward file) and inherits
                         `$PYTHONPATH` (a fresh process with a patched module gets an overlay first)
  FASTRSI_VERL_REWARD    the custom reward file the job loads (patched by the conformance edits)
  FASTRSI_VERL_CHECK_TIMEOUT  seconds one check may take (default 300; also the hang budget)
  FASTRSI_VERL_BOOT_TIMEOUT   seconds a (re)launch may take to `ready.json` (default 1800)

Output of a check: `{"steps": [{"score_mean", "adv_max", "adv_min", "pg_loss", "rollouts"}]}` per
RL step after a coupled S0 reset; `rollouts` is a digest of the step's dumped (prompt, response,
score, ground truth) rows; `pg_loss` is rounded to 1e-6 (see `ROUND`). Bit-exact repeats need `rollout.full_determinism=True` and the
per-sample seeds of `determinism.py` (`FASTRSI_RL_SEED_PER_SAMPLE=1`).
Run from `core/livelab`:
    python3 -m conformance --lab adapter.conformance_lab:VerlConformanceLab
or `python3 -m tools.cli conformance --gate G --launch CMD --reward FILE` from `repos/verl`.
"""
import hashlib, importlib, json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from kernel import findings as F
from adapter import gate as G

CORE = "verl.trainer.ppo.core_algos"
METRICS = {"score_mean": "critic/score/mean", "adv_max": "critic/advantages/max", "adv_min": "critic/advantages/min", "pg_loss": "actor/pg_loss"}
# pg_loss is bit-exact across resets inside one job but differs at 1e-9 between two processes (FSDP loss
# reduction order), so it is compared at 1e-6; scores, advantages and the rollout digest are exact.
ROUND = {"pg_loss": 6}
_JOBS = {}      # gate dir -> Popen of the launcher this process started


def _edit(src, old, new):
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
    return src.replace(old, new)


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

    def __init__(self, receipts=None, gate=None, launch=None, reward=None, check_timeout=None, boot_timeout=None):
        env = os.environ
        self.gate = Path(gate or env["FASTRSI_VERL_GATE"]).resolve()
        self.launch = launch or env.get("FASTRSI_VERL_LAUNCH")
        self.reward = os.path.realpath(reward or env["FASTRSI_VERL_REWARD"])
        self.check_timeout = float(check_timeout or env.get("FASTRSI_VERL_CHECK_TIMEOUT", 300))
        self.boot_timeout = float(boot_timeout or env.get("FASTRSI_VERL_BOOT_TIMEOUT", 1800))
        self.receipts, self.poison, self.stack_patches, self.costs = receipts, [], [], {}
        self._receipts_offset = self._receipt_size()
        self._ensure_job()

    # -- the job -------------------------------------------------------------------------
    def _receipt_size(self):
        p = self.gate / "receipts.jsonl"
        return p.stat().st_size if p.exists() else 0

    def _info(self, gate=None):
        try:
            return json.loads(((gate or self.gate) / "gate.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _alive(self, gate=None):
        pid = self._info(gate).get("pid")
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True

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
        log = open(gate.parent / f"{gate.name}-launch.log", "a")
        t0 = time.time()
        _JOBS[str(gate)] = subprocess.Popen(self.launch, shell=True, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self._wait_ready(gate, t0)
        self.costs["boot_s"] = round(time.time() - t0, 3)
        return self.costs["boot_s"]

    def _wait_ready(self, gate, since):
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            ready = gate / "ready.json"
            if ready.exists():
                try:
                    if json.loads(ready.read_text(encoding="utf-8")).get("t", 0) >= since:
                        return
                except ValueError:
                    pass
            proc = _JOBS.get(str(gate))
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"{gate}: the launch command exited with {proc.returncode} before the gate was ready; "
                                   f"last errors in {gate.name}-launch.log: {self._launch_errors(gate)}")
            time.sleep(0.5)
        raise RuntimeError(f"{gate}: no ready.json within {self.boot_timeout}s")

    def _launch_errors(self, gate, n=3):
        try:
            lines = (gate.parent / f"{gate.name}-launch.log").read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        return [ln[-200:] for ln in lines if "Error" in ln and "UserWarning" not in ln][-n:]

    def _wait_gone(self, gate, timeout_s):
        deadline, proc = time.monotonic() + timeout_s, _JOBS.get(str(gate))
        while time.monotonic() < deadline:
            if not self._alive(gate) and (proc is None or proc.poll() is not None):
                return True
            time.sleep(0.5)
        return False

    def _gpus_busy(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
            return [p for p in out.split() if p.strip()]
        except (OSError, subprocess.SubprocessError):
            return []

    def _kill_job(self, gate):
        proc, pid = _JOBS.get(str(gate)), self._info(gate).get("pid")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc is not None and proc.poll() is None:
                with _suppress():
                    os.killpg(proc.pid, sig)
            if isinstance(pid, int):
                with _suppress():
                    os.kill(pid, sig)
            if self._wait_gone(gate, 20):
                break
        deadline = time.monotonic() + 90
        while self._gpus_busy() and time.monotonic() < deadline:
            time.sleep(1.0)
        if self._gpus_busy():
            subprocess.run(["pkill", "-9", "-u", os.environ.get("USER", ""), "-f", r"ray::|EngineCore|vLLMHttpServer|verl\.trainer\.main_ppo"], check=False)
            time.sleep(3.0)

    def _stop_job(self, gate=None):
        from tools import cli
        gate = gate or self.gate
        t0 = time.time()
        if self._alive(gate):
            with _suppress():
                cli.request(gate, stop=True, timeout=60.0)
            if not self._wait_gone(gate, 120):
                self._kill_job(gate)
        deadline = time.monotonic() + 60
        while self._gpus_busy() and time.monotonic() < deadline:
            time.sleep(1.0)
        self.costs["stop_s"] = round(time.time() - t0, 3)

    # -- requests ------------------------------------------------------------------------
    def _request(self, **kw):
        """One gate request, mirrored to `<gate>-client.jsonl` (a relaunch wipes the gate's own files)."""
        from tools import cli
        t0 = time.time()
        try:
            resp = cli.request(self.gate, **kw)
        except cli.GateError as err:
            self._log({"t": t0, "req": _jsonable(kw), "error": str(err), "s": round(time.time() - t0, 3)})
            raise
        self._log({"t": t0, "req": _jsonable(kw), "resp": {k: v for k, v in resp.items() if k not in ("req", "steps_run")}, "s": round(time.time() - t0, 3)})
        return resp

    def _log(self, record):
        with _suppress():
            with open(self.gate.parent / f"{self.gate.name}-client.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def _fault(self, check, observed, expected):
        f = F.finding(check, observed, expected)
        self.poison.append(f)
        return f

    def _refused(self, t0, what="install"):
        return {"ok": False, "id": None, "outcome": "refused", "codes": ["session_poisoned"], "mechanisms": [],
                "findings": [F.finding("session_poisoned", "; ".join(f["check"] for f in self.poison), "a session in a known state")], "cost_s": round(time.perf_counter() - t0, 6)}

    def _output(self, resp):
        steps = []
        for st in resp.get("steps_run") or []:
            m = st.get("metrics") or {}
            vals = {k: m.get(v) for k, v in METRICS.items()}
            steps.append({**{k: (round(x, ROUND[k]) if k in ROUND and isinstance(x, float) else x) for k, x in vals.items()}, "rollouts": _digest_rollouts(st.get("rollouts"))})
        return {"steps": steps}

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
        from tools import cli
        try:
            r = self._request(verb="install", steps=0, timeout=self.check_timeout,
                              extra={"sources": patch["files"], "accept": patch.get("accept", []), "undo_resets": patch.get("undo_resets", [])})
        except cli.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return {"ok": False, "id": None, "outcome": "aborted", "codes": ["participant_failed"], "mechanisms": [], "findings": list(self.poison)}
        outcome = r.get("outcome") or ("committed" if r.get("ok") else "refused")
        codes = [] if outcome == "committed" else (r.get("codes") or sorted({f["check"] for f in r.get("findings", [])}) or ["refused"])
        if r.get("restart_required"):
            self._fault("session_poisoned", r.get("error") or "gate session poisoned", "a session in a known state")
        return {"ok": outcome == "committed", "id": r.get("txn"), "outcome": outcome, "codes": codes, "mechanisms": r.get("mechanisms", []),
                "findings": r.get("findings", []), "install_s": r.get("install_s"), "receipt": r}

    def check(self, workload=None):
        from tools import cli
        t0 = time.perf_counter()
        if self.poison:
            return {"ok": False, "error": "session poisoned", "output": None, "proven": None, "witness": {}, "findings": list(self.poison), "cost_s": round(time.perf_counter() - t0, 6)}
        steps = int((workload or {}).get("steps", 1))
        try:
            rs = self._request(verb="reset", steps=0, timeout=self.check_timeout)
            if not rs.get("ok"):
                self._fault("participant_failed", f"S0 restore failed: {rs.get('error')}", "coupled S0 restores")
                return {"ok": False, "error": rs.get("error"), "output": None, "proven": None, "witness": {}, "findings": list(self.poison), "cost_s": round(time.perf_counter() - t0, 6)}
            r = self._request(verb="check", steps=steps, timeout=self.check_timeout * steps)
        except cli.GateError as err:
            self._fault("participant_failed", str(err), "the RL step completes in time")
            return {"ok": False, "error": str(err), "output": None, "proven": None, "witness": {}, "findings": list(self.poison), "cost_s": round(time.perf_counter() - t0, 6)}
        if not r.get("steps_run"):
            self._fault("participant_failed", r.get("error") or "no step ran", "the RL step completes")
            return {"ok": False, "error": r.get("error") or "no step ran", "output": None, "proven": None, "witness": {}, "findings": list(self.poison), "cost_s": round(time.perf_counter() - t0, 6)}
        live = bool(self._request(verb="status", steps=0, timeout=60.0).get("txns")) if "proven" not in r else True
        return {"ok": True, "output": self._output(r), "seconds": r.get("verdict_s"), "proven": r.get("proven") if live else None,
                "witness": r.get("witness") or {}, "findings": r.get("findings") or [], "reset_s": rs.get("cost_s"), "cost_s": round(time.perf_counter() - t0, 6),
                **({"abstained": r["abstained"]} if r.get("abstained") else {})}

    def reset(self, kind="state"):
        from tools import cli
        t0 = time.perf_counter()
        if kind == "respawn":
            if self._alive() and not self.poison:
                self._stop_job()
            else:
                self._kill_job(self.gate)
            self.poison, self.stack_patches = [], []
            try:
                boot = self._start_job(self.gate)
            except RuntimeError as err:
                return {"ok": False, "error": str(err), "reset": [], "cost_s": round(time.perf_counter() - t0, 6)}
            return {"ok": True, "reset": ["ray_job", "coupled_s0", "session"], "boot_s": boot, "cost_s": round(time.perf_counter() - t0, 6)}
        try:
            r = self._request(verb="reset", steps=0, timeout=self.check_timeout)
        except cli.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return {"ok": False, "error": str(err), "reset": [], "cost_s": round(time.perf_counter() - t0, 6)}
        return {"ok": bool(r.get("ok")), "error": r.get("error"), "reset": r.get("reset", []), "digests": r.get("digests"), "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        from tools import cli
        t0 = time.perf_counter()
        try:
            r = self._request(verb="uninstall", steps=0, timeout=self.check_timeout, extra={"id": id} if id else None)
        except cli.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            return {"ok": False, "undone": [], "state": "poisoned", "findings": list(self.poison), "cost_s": round(time.perf_counter() - t0, 6)}
        if r.get("undone"):
            del self.stack_patches[-len(r["undone"]):]
        return {"ok": bool(r.get("ok")), "undone": r.get("undone", []), "state": r.get("state"), "findings": r.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        from tools import cli
        try:
            r = self._request(verb="status", steps=0, timeout=60.0) if not self.poison else {}
        except cli.GateError as err:
            self._fault("participant_failed", str(err), "the gated job answers")
            r = {}
        poison = list(self.poison) + list(r.get("poison") or [])
        state = "poisoned" if poison else r.get("state", "poisoned")
        return {"ok": state != "poisoned", "state": state, "txns": r.get("txns", []), "fingerprint": r.get("fingerprint"), "poison": poison}

    # -- the harness ---------------------------------------------------------------------
    def _base(self, key):
        return Path(G.path_for_key(key)).read_text(encoding="utf-8")

    def patches(self):
        core, rew = self._base(CORE), self._base(self.reward)
        norm = "                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)\n"
        x2, x3 = _edit(core, norm, norm[:-1] + " * 2.0\n"), _edit(core, norm, norm[:-1] + " * 3.0\n")
        reward_x2 = _edit(rew, "return SCORE if answer == str(ground_truth) else FORMAT_SCORE", "return SCORE * 2 if answer == str(ground_truth) else FORMAT_SCORE")
        agg_sig = "def agg_loss(\n    loss_mat: torch.Tensor,\n    loss_mask: torch.Tensor,\n    loss_agg_mode: str,\n    dp_size: int = 1,\n    batch_num_tokens: Optional[int] = None,\n    global_batch_size: Optional[int] = None,\n    loss_scale_factor: Optional[int] = None,\n):\n"
        multi = _edit(_edit(x2, agg_sig, agg_sig + "    loss_agg_mode = str(loss_agg_mode)\n"),
                      "    # Strip the optional '+' suffix so e.g. \"k3+\" dispatches to \"k3\".\n", "    kl_penalty = str(kl_penalty)\n    # Strip the optional '+' suffix so e.g. \"k3+\" dispatches to \"k3\".\n")
        return {
            "noop": {"files": {CORE: _edit(core, norm, norm[:-1] + " * 1.0\n")}},
            "visible": {"files": {CORE: x2}},
            "chain": [{"files": {CORE: x2}}, {"files": {CORE: x3}}, {"files": {self.reward: reward_x2}}],
            "multi": {"files": {CORE: multi, self.reward: reward_x2}},
            "unreached": {"files": {self.reward: _edit(rew, "    return 0.0\n\n\nFORMAT_SCORE = format_score()", "    return 0.5\n\n\nFORMAT_SCORE = format_score()")}},
            "refused": {
                "syntax": {"files": {CORE: _edit(core, norm, norm.replace("scores[i] = (", "scores[i] = = ("))}},
                "closure": {"files": {self.reward: _edit(rew, "checks = [lambda m: m is not None]", "checks = [lambda m: m is not None and checks]")}},
                "module_state": {"files": {self.reward: _edit(rew, "SCORE = 1.0\n", "SCORE = 2.0\n")}},
                "empty": {"files": {self.reward: _edit(rew, "def compute_score(data_source=None, solution_str=\"\", ground_truth=\"\", extra_info=None):", "def compute_score(data_source=None, solution_str=\"\", ground_truth=\"\", extra_info=None):  # note")}},
            },
        }

    def _overlay(self, tmp, files):
        overlay, reward = None, None
        for key, text in files.items():
            path = G.path_for_key(key)
            if path == self.reward:
                reward = os.path.join(tmp, os.path.basename(path))
                Path(reward).write_text(text, encoding="utf-8")
            elif os.path.isfile(key):
                raise ValueError(f"{key}: a fresh process can only take the job's reward file or an importable module")
            else:
                top = key.split(".")[0]
                if overlay is None:
                    overlay, pkg = os.path.join(tmp, "overlay"), os.path.dirname(importlib.import_module(top).__file__)
                    for d, dirs, fns in os.walk(pkg):
                        dirs[:] = [x for x in dirs if x != "__pycache__"]
                        rel = os.path.join(overlay, top, os.path.relpath(d, pkg))
                        os.makedirs(rel, exist_ok=True)
                        for fn in fns:
                            os.symlink(os.path.join(d, fn), os.path.join(rel, fn))
                target = os.path.join(overlay, *key.split(".")) + ".py"
                if os.path.lexists(target):
                    os.unlink(target)
                Path(target).write_text(text, encoding="utf-8")
        return overlay, reward

    def fresh_check(self, patch, workload=None):
        from tools import cli
        files, steps = (patch or {}).get("files") or {}, int((workload or {}).get("steps", 1))
        if not self.launch:
            return {"ok": False, "error": "no FASTRSI_VERL_LAUNCH: cannot start a fresh job", "output": None}
        t0, tmp = time.perf_counter(), tempfile.mkdtemp(prefix="verllab-fresh-")
        try:
            self._stop_job()
            if not files:                       # a fresh base: start it on the live gate and keep it as the live job
                self._start_job(self.gate)
                out = self._output(cli.request(self.gate, verb="check", steps=steps, timeout=self.check_timeout * steps))
                self._replay()
                return {"ok": True, "output": out, "cost_s": round(time.perf_counter() - t0, 6)}
            overlay, reward = self._overlay(tmp, files)
            fresh = self.gate.with_name(self.gate.name + "-fresh")
            self._start_job(fresh, reward=reward, pythonpath=overlay)
            try:
                out = self._output(cli.request(fresh, verb="check", steps=steps, timeout=self.check_timeout * steps))
            finally:
                self._stop_job(fresh)
            self._start_job(self.gate)
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
        for p in patches:
            res = self._install(p)
            if not res["ok"]:
                self._fault("undo_incomplete", f"replaying {sorted(p['files'])} after relaunch: {res['outcome']} {res['codes']}", "the live edits install again")
                return
            self.stack_patches.append(p)

    def inject(self, fault, **kw):
        from tools import cli
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 5.0)
        participant = kw.pop("participant", "reward0" if fault in ("fail_commit", "lost") else "driver")
        try:
            return self._request(verb="inject", steps=0, timeout=60.0, extra={"fault": fault, "participant": participant, "kwargs": kw})
        except cli.GateError as err:
            return {"ok": False, "error": str(err)}

    def close(self):
        if not self.poison and self._alive():
            with _suppress():
                st = self._request(verb="status", steps=0, timeout=60.0)
                if st.get("txns"):
                    self._request(verb="uninstall", steps=0, timeout=self.check_timeout, extra={"id": st["txns"][0]})
                self._request(verb="reset", steps=0, timeout=self.check_timeout)
        if self.receipts:
            with _suppress():
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


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
