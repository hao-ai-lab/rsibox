"""Engine-agnostic lab building blocks: `BaseLab`, `RecaptureCoordinator`, and `RoleEndpoint`.

Built strictly on `core/livelab/kernel/` (stdlib only, zero engine imports). Eliminates
duplicated `LabUnderTest` verb plumbing (`install`, `check`, `reset`, `uninstall`, `status`,
`live_text` stack tracking, witness/proof aggregation, poison-on-fault), multi-role abstain
dispatch, and post-commit rebuild + bounded in-gap rollback across engines.
"""
import contextlib
import json
import os
import threading
import time
import traceback

from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L
from kernel import receipt as R
from kernel import session as SS

from . import harness as H


# -- Multi-role abstaining participant endpoint ------------------------------------------

ENDPOINT_METHODS = (
    "prepare", "validate", "commit", "abort", "undo", "status",
    "proof", "witness_check", "reset", "at_base", "fingerprint",
)


class RoleEndpoint:
    """Participant-side dispatcher for one role (and optional rank) of a multi-role engine.

    Filters incoming plans by each module/repair's `"on"` role list (default: every role),
    abstains cleanly when a plan targets only other roles, and guarantees every `dispatch()`
    call returns an ok-dict tagged with `pid`, `role`, optional `rank`, and `side_s` without
    ever raising across an RPC boundary.
    """

    def __init__(self, role: str, scope: dict | None = None, rank: int | None = None, **participant):
        self.role = role
        self.rank = rank
        self.scope = dict(scope or {})
        self.lp = C.LocalParticipant(role, scope=self.scope, **participant)
        self._abstain: dict[str, str] = {}

    def who(self) -> str:
        return self.role if self.rank is None else f"{self.role} rank {self.rank}"

    def subplan(self, plan: dict) -> dict:
        """Modules and repair cells of `plan` targeting this role."""
        mods = {
            mod: spec
            for mod, spec in (plan.get("modules") or {}).items()
            if self.role in spec.get("on", [self.role])
        }
        reps = [
            rep
            for rep in (plan.get("repairs") or [])
            if self.role in rep.get("on", [self.role])
        ]
        return {**plan, "modules": mods, "repairs": reps}

    def prepare(self, plan: dict) -> dict:
        """Prepare this role's sub-plan, or abstain when the plan targets only other roles."""
        sub = self.subplan(plan)
        if sub["modules"] or sub["repairs"] or not (plan.get("modules") or plan.get("repairs")):
            self._abstain.pop(plan.get("id"), None)
            return self.lp.prepare(sub)
        self._abstain[plan["id"]] = "prepared"
        return F.ok(staged={}, abstain=True, note=f"no module or cell for role {self.role}")

    def local(self, method: str, payload) -> dict:
        """Dispatch to `LocalParticipant`, or answer consistently for an abstained transaction."""
        txn, _ = C.txn_ref(payload)
        if not isinstance(txn, str) or txn not in self._abstain:
            return getattr(self.lp, method)(payload)
        nxt = {"commit": "committed", "abort": "aborted", "undo": "undone"}.get(method)
        if nxt and not (method == "abort" and self._abstain[txn] == "committed"):
            self._abstain[txn] = nxt
        if method == "status":
            return F.ok(txn=txn, state=self._abstain[txn], abstain=True)
        if method == "proof":
            return F.ok(hashes={}, witness={}, abstain=True)
        if method == "witness_check":
            return F.ok(counts={}, findings=[], abstain=True)
        return F.ok(abstain=True)

    def dispatch(self, method: str, payload=None) -> dict:
        t0 = time.perf_counter()
        try:
            reply = self._dispatch(method, payload)
        except BaseException as exc:  # never raise across an RPC transport
            reply = F.not_ok(
                f"{method} on {self.who()}: {type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-1500:],
            )
        if not isinstance(reply, dict) or "ok" not in reply:
            reply = F.not_ok(f"{method}: bad reply {reply!r}")
        reply.setdefault("pid", os.getpid())
        reply.setdefault("role", self.role)
        if self.rank is not None:
            reply["rank"] = self.rank
        reply["side_s"] = round(time.perf_counter() - t0, 6)
        return reply

    def _dispatch(self, method: str, payload) -> dict:
        if method not in ENDPOINT_METHODS:
            return F.not_ok(f"unknown method {method}")
        if method == "prepare":
            return self.prepare(payload)
        if method == "reset":
            self._abstain.clear()
            return self.lp.reset(payload)
        if method == "fingerprint":
            return self.lp.fingerprint(payload)
        return self.local(method, payload)


# -- Post-commit rebuild coordinator (shared across graph-capturing engines) -------------


class RecaptureCoordinator(C.StepGapCoordinator):
    """`StepGapCoordinator` plus a post-commit rebuild hook (e.g. selective CUDA graph
    recapture) executed in the same step gap with automatic bounded in-gap rollback on failure.

    Parameters:
    - `rebuild(txn, spec) -> dict`: invoked right after commit when `plan["rebuild"]` contains
      any key in `rebuild_triggers` (default `("recapture", "R1")`). Must return
      `{"ok": bool, "ranks": [...], ...}`.
    - `post_commit_reset`: optional reset kind (e.g. `"state"`) applied to every participant
      in the gap right after a successful commit (and rebuild).
    - `rank_fields`: per-rank keys copied into `receipt["rebuild"]["ranks"]`.
    """

    DEFAULT_RANK_FIELDS = (
        "ok", "error", "seconds", "captured", "modes", "wrappers", "graphs", "eager", "memory", "skipped",
    )

    def __init__(
        self,
        *args,
        rebuild=None,
        rebuild_triggers: tuple[str, ...] = ("recapture", "R1"),
        post_commit_reset: str | None = None,
        rebuild_participant: str = "workers",
        rank_fields: tuple[str, ...] = DEFAULT_RANK_FIELDS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rebuild = rebuild
        self.rebuild_triggers = tuple(rebuild_triggers)
        self.post_commit_reset = post_commit_reset
        self.rebuild_participant = rebuild_participant
        self.rank_fields = tuple(rank_fields)

    def _phase2(self, txn, tl):
        out = super()._phase2(txn, tl)
        if out["outcome"] != "committed":
            return out
        plan = self.txns[txn]["plan"]
        rebuild_keys = plan.get("rebuild") or ()
        matched = [key for key in self.rebuild_triggers if key in rebuild_keys]
        if self.rebuild is not None and matched:
            spec = dict(plan.get("recapture") or plan.get("r1") or {})
            t0 = self.clock()
            reply = self.rebuild(txn, spec)
            tl["rebuilt"] = self.clock()
            ranks = reply.get("ranks") or []
            dur = round(tl["rebuilt"] - t0, 6)
            route_name = matched[0]
            out["rebuild"] = {
                "ok": bool(reply.get("ok")),
                "routes": [route_name],
                "seconds": {route_name: dur, "R1": dur, "recapture": dur},
                "spec": spec,
                "ranks": {
                    str(item.get("rank")): {
                        k: item.get(k) for k in self.rank_fields if k in item
                    }
                    for item in ranks
                },
            }
            if not reply.get("ok"):
                extra = [f for item in ranks for f in item.get("findings", [])] or [
                    F.finding(
                        "rebuild_failed",
                        str(reply.get("error")),
                        "rebuild succeeds on every rank",
                    )
                ]
                out["extra"] += [{**f, "participant": self.rebuild_participant} for f in extra]
                rb = out["rollback"] = {"status": "undone", "participants": {}}
                res_map = self.txns[txn]["res"]
                undo_deadline = (
                    time.time() + self.phase2_deadline_s if self.phase2_deadline_s else None
                )
                missed: set = set()
                for part in reversed(self.participants):
                    rb["participants"][part.name] = res_map[part.name]["undo"] = self._bounded(
                        part, "undo", txn, undo_deadline, missed
                    )
                if not all(u["ok"] for u in rb["participants"].values()):
                    rb["status"] = "partial"
                    with self._lock:
                        self.txns[txn]["pending_rollback"] = [
                            part
                            for part in reversed(self.participants)
                            if not rb["participants"][part.name]["ok"]
                        ]
                out["outcome"] = "rolled_back"
                tl.pop("committed", None)
                tl["gap_end"] = self.clock()
                return out
            self.txns[txn]["rebuilt"] = [route_name]

        if self.post_commit_reset:
            deadline = time.time() + self.phase2_deadline_s if self.phase2_deadline_s else None
            missed = set()
            for part in self.participants:
                self._bounded(part, "reset", self.post_commit_reset, deadline, missed)
            tl["gap_end"] = self.clock()
        return out


# -- Reusable `LabUnderTest` base class --------------------------------------------------


class BaseLab:
    """Thread-safe, never-raising `LabUnderTest` base over `kernel.session.Session`.

    Subclasses implement only the engine-specific lifecycle and workload hooks:
    - `_start_engine() -> (coordinator, modules)`
    - `_stop_engine() -> None`
    - `_run_workload(workload) -> output`
    - `_reset_state() -> dict | list[str]`
    And, for conformance (`core/livelab/conformance`), the three harness methods:
    - `patches() -> dict`
    - `fresh_check(patch, workload=None) -> dict`
    - `inject(fault, **kwargs) -> dict`

    Optional hooks for richer adapters: `_base_source(key)`, `_build_plan(patch, live_map)`,
    `_after_commit(patch, receipt)`, `_after_uninstall(undone)`, `_probe_participants()`,
    `_format_output(raw)`, `_golden_for(output, extra)`.
    """

    name = "base"
    threaded_check = False

    def __init__(
        self,
        *,
        receipts: str | None = None,
        check_timeout: float = 60.0,
        rpc_timeout: float = 30.0,
        auto_spawn: bool = True,
        weight_store=None,
        recovery_driver=None,
    ):
        self.receipts = receipts
        self.writer = R.ReceiptWriter(receipts) if receipts else None
        self.check_timeout = float(check_timeout)
        self.rpc_timeout = float(rpc_timeout)
        self.weight_store = weight_store
        self.recovery_driver = recovery_driver
        self.lock = threading.RLock()
        self.costs: dict[str, float] = {}
        self.base_text: dict[str, str] = {}
        self.live_text: list[dict[str, str]] = [{}]
        self._dirty = False
        self.co = None
        self.session = None
        self.modules: list[str] = []
        if auto_spawn:
            self._spawn()

    # -- lifecycle hooks (subclasses override) -------------------------------------------
    def _start_engine(self):
        raise NotImplementedError("subclass must implement _start_engine() -> (coordinator, modules)")

    def _stop_engine(self) -> None:
        return None

    def _run_workload(self, workload):
        raise NotImplementedError("subclass must implement _run_workload(workload)")

    def _reset_state(self):
        return {"ok": True, "reset": ["state"]}

    def quiesce_hooks(self):
        """Optional `rsibox.warm.QuiesceHooks` for snapshot/shadow integration."""
        return None

    # -- optional customization hooks ----------------------------------------------------
    def _base_source(self, key: str) -> str:
        if key not in self.base_text:
            self.base_text[key] = H.read_module(key)
        return self.base_text[key]

    @staticmethod
    def _disk_source(key: str) -> str | None:
        try:
            return H.read_module(key)
        except (ImportError, OSError):
            return None

    def files(self, patch: dict | None, against: str = "live") -> dict[str, str]:
        """Resolve `{module: new_text}` from `patch["files"]` or unified `patch["diff"]`.

        With `against="live"` (default), diffs apply to the lab's live text (base plus every
        committed edit so far); with `against="disk"`, diffs apply to the files on disk.
        Raises `ValueError` when a diff does not apply cleanly.
        """
        live = self.live_text[-1]

        def current_live(mod: str) -> str | None:
            if mod in live:
                return live[mod]
            try:
                return self._base_source(mod)
            except (ImportError, OSError):
                return None

        return H.resolve_patch_files(
            patch, current_live, self._disk_source, against=against, reject_mixed=False
        )

    def _build_plan(self, patch: dict, live_map: dict[str, str]):
        modules = {}
        for mod, new_src in patch["files"].items():
            base_src = live_map.get(mod)
            if base_src is None:
                base_src = self._base_source(mod)
            modules[mod] = {"source": new_src, "base_hash": L.text_hash(base_src)}
        plan = {
            "modules": modules,
            "accept": list(patch.get("accept", [])),
            "undo_resets": list(patch.get("undo_resets", [])),
            "recipes": list(patch.get("recipes", ["R0 swap"])),
        }
        return plan, []

    def _after_commit(self, patch: dict, receipt: dict | None) -> None:
        return None

    def _after_uninstall(self, undone: list[str]) -> None:
        return None

    def _probe_participants(self) -> tuple[bool, str | None]:
        for participant in self.co.participants:
            reply = C._call(participant, "status", "ping")
            if not reply.get("ok"):
                return False, f"{participant.name}: {reply.get('error')}"
        return True, None

    def _format_output(self, raw):
        return raw, {}

    def _golden_for(self, output, extra: dict) -> dict:
        return {"output_hash": L.text_hash(json.dumps(output, sort_keys=True))}

    def _clean_witness(self, raw_witness: dict, proof_ok: bool) -> dict:
        return raw_witness or {}

    # -- internal helpers ----------------------------------------------------------------
    def _spawn(self) -> None:
        t0 = time.perf_counter()
        self._dirty = False
        co, modules = self._start_engine()
        self.costs["load_s"] = round(time.perf_counter() - t0, 4)
        self.co = co
        self.modules = list(modules)
        self.live_text = [dict(self.base_text)]
        if self.session is not None:
            self.session.recovered(self.co)
        else:
            self.session = SS.Session(self.co, self.modules)

    def _refuse_install(self, outcome: str, codes: list[str], findings: list[dict], t0: float) -> dict:
        return {**H.install_refusal(codes, findings, outcome=outcome, t0=t0), "receipt": None}

    def _fail_check(self, observed: str, expected: str, t0: float, *, subject: str | None = None) -> dict:
        self.session.fault("participant_failed", observed, expected, subject=subject)
        return H.check_failure(observed, list(self.session.poisoned()), t0=t0)

    def _execute_workload(self, workload):
        if not self.threaded_check:
            return self._run_workload(workload)
        box: dict = {}

        def runner():
            try:
                box["result"] = self._run_workload(workload)
            except BaseException as exc:
                box["error"] = exc

        thread = threading.Thread(target=runner, daemon=True, name=f"rsibox-{self.name}-check")
        thread.start()
        thread.join(self.check_timeout)
        if thread.is_alive():
            self._dirty = True
            raise TimeoutError(f"no result in {self.check_timeout} s")
        if "error" in box:
            raise box["error"]
        return box["result"]

    # -- five INTERFACE verbs + close ----------------------------------------------------
    def install(self, patch: dict) -> dict:
        t0 = time.perf_counter()
        with self.lock:
            if self.session.poisoned():
                refused = self.session.install({"modules": {}})
                codes = sorted({f["check"] for f in refused.get("findings", [])})
                return self._refuse_install("refused", codes, refused.get("findings", []), t0)
            live_map = self.live_text[-1]
            try:
                if patch.get("diff") is not None:
                    resolved_files = self.files(patch)
                    effective_patch = {**patch, "files": resolved_files}
                else:
                    resolved_files = dict(patch.get("files") or {})
                    effective_patch = patch
                plan, plan_findings = self._build_plan(effective_patch, live_map)
            except Exception as exc:
                finding = F.finding(
                    "patch_invalid", f"{type(exc).__name__}: {exc}", "importable modules"
                )
                return self._refuse_install("refused", ["patch_invalid"], [finding], t0)
            if plan is None:
                codes = sorted({f["check"] for f in plan_findings})
                return self._refuse_install("refused", codes, plan_findings, t0)
            res = self.session.install(plan, edit_time=effective_patch.get("edit_time"))
            outcome = res.get("outcome") or "refused"
            rec = res.get("receipt")
            if outcome == "committed":
                self.live_text.append({**live_map, **resolved_files})
                self._after_commit(effective_patch, rec)
            codes = (
                rec["decisions"]["blocking"]
                if rec
                else sorted({f["check"] for f in res.get("findings", [])})
            )
            mechanisms = list(rec["actual_recipes"]) if rec else []
            if outcome == "committed" and ((rec or {}).get("rebuild") or {}).get("ok"):
                for recipe in plan.get("recipes") or ():
                    if recipe not in mechanisms:
                        mechanisms.append(recipe)
            return {
                "ok": outcome == "committed",
                "id": res.get("txn"),
                "outcome": outcome,
                "codes": [] if outcome == "committed" else codes,
                "mechanisms": mechanisms,
                "findings": list(rec["findings"]) if rec else list(res.get("findings", [])),
                "edit_to_run_s": rec.get("edit_to_run_s") if rec else None,
                "receipt": rec,
                "cost_s": round(time.perf_counter() - t0, 6),
            }

    def check(self, workload: dict | None = None) -> dict:
        t0 = time.perf_counter()
        with self.lock:
            ok_probe, probe_err = self._probe_participants()
            if not ok_probe:
                return self._fail_check(
                    probe_err or "participant probe failed", "every participant answers", t0
                )
            t_run = time.perf_counter()
            try:
                raw = self._execute_workload(workload)
            except Exception as exc:
                return self._fail_check(
                    f"check: {type(exc).__name__}: {exc}",
                    f"the workload finishes within {self.check_timeout} s",
                    t0,
                )
            seconds = round(time.perf_counter() - t_run, 6)
            output, extra = self._format_output(raw)
            golden = self._golden_for(output, extra)
            proofs = [self.co.prove(tid, golden=golden) for tid in self.session.stack]
            witness: dict = {}
            abstained: set[str] = set()
            for proof in proofs:
                rec = proof.get("receipt") or {}
                abstained.update(rec.get("abstained") or ())
                cleaned = self._clean_witness(rec.get("witness") or {}, bool(proof.get("ok")))
                for part, counts in cleaned.items():
                    witness.setdefault(part, {}).update(counts)
            return {
                "ok": True,
                "output": output,
                "seconds": seconds,
                "proven": all(p["ok"] for p in proofs) if proofs else None,
                "proven_by_txn": {t: p["ok"] for t, p in zip(self.session.stack, proofs)},
                "witness": witness,
                **({"abstained": sorted(abstained)} if abstained else {}),
                "findings": [f for p in proofs for f in (p.get("receipt") or {}).get("findings", [])],
                "cost_s": round(time.perf_counter() - t0, 6),
                **extra,
            }

    def reset(self, kind: str = "state") -> dict:
        t0 = time.perf_counter()
        with self.lock:
            if kind == "respawn":
                if self.recovery_driver is not None:
                    return self.recovery_driver.respawn(
                        self, self.weight_store, reason="reset_respawn"
                    )
                with contextlib.suppress(Exception):
                    self._stop_engine()
                self._spawn()
                return {"ok": not self.session.poisoned(), "reset": ["worker group", "session"],
                        "cost_s": round(time.perf_counter() - t0, 6)}
            try:
                res = self._reset_state()
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "reset": [],
                        "cost_s": round(time.perf_counter() - t0, 6)}
            extra = {k: v for k, v in res.items() if k not in ("ok", "reset", "cost_s")} if isinstance(res, dict) else {}
            return {
                "ok": bool(res.get("ok", True)) if isinstance(res, dict) else True,
                "reset": list(res.get("reset", [])) if isinstance(res, dict) else list(res),
                **extra, "cost_s": round(time.perf_counter() - t0, 6),
            }

    def uninstall(self, id: str | None = None) -> dict:
        t0 = time.perf_counter()
        with self.lock:
            res = self.session.uninstall(id)
            undone = list(res.get("undone", []))
            if undone:
                del self.live_text[-len(undone):]
                self._after_uninstall(undone)
            return {
                "ok": bool(res.get("ok")), "undone": undone, "state": self.session.state(),
                "findings": list(res.get("findings", [])), "cost_s": round(time.perf_counter() - t0, 6),
            }

    def status(self) -> dict:
        with self.lock:
            return self.session.status()

    def close(self) -> None:
        with self.lock:
            with contextlib.suppress(Exception):
                self._stop_engine()

    # -- conformance harness defaults ----------------------------------------------------
    def patches(self) -> dict:
        raise NotImplementedError("subclass must implement patches()")

    def fresh_check(self, patch: dict | None, workload: dict | None = None) -> dict:
        raise NotImplementedError("subclass must implement fresh_check(patch, workload)")

    def inject(self, fault: str, **kwargs) -> dict:
        raise NotImplementedError("subclass must implement inject(fault, **kwargs)")
