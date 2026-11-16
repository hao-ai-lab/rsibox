"""SglangLab: live-lab verbs over `StepGapCoordinator` + `RecaptureCoordinator` across TP ranks."""
import contextlib, functools, hashlib, importlib, importlib.util, itertools, json, os, select, signal, threading, time
from multiprocessing.connection import Listener
from adapter import agent
from adapter.fixture.harness import ConformanceHarness
from kernel import coordinator as C, findings as F, locate as L, receipt as R, session as SS

EAGER_LAYERS = (".srt.layers.sampler", ".srt.layers.logits_processor")
_counters = itertools.count(1)

def is_graph_module(module_name):
    return not module_name.endswith(EAGER_LAYERS) and (".srt.models." in module_name or ".srt.layers." in module_name)

class RecaptureCoordinator(C.StepGapCoordinator):
    def __init__(self, *args, rebuild=None, flush_on_commit=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild, self.flush_on_commit = rebuild, flush_on_commit
    def _phase2(self, txn, timeline):
        out = super()._phase2(txn, timeline)
        if out["outcome"] != "committed": return out
        rebuild_keys = self.txns[txn]["plan"].get("rebuild") or ()
        if self.rebuild is not None and ("recapture" in rebuild_keys or "R1" in rebuild_keys):
            spec, started = dict(self.txns[txn]["plan"].get("recapture") or {}), self.clock()
            reply = self.rebuild(txn, spec)
            timeline["rebuilt"] = self.clock()
            ranks, dur = reply.get("ranks") or [], round(timeline["rebuilt"] - started, 6)
            route = "recapture" if "recapture" in rebuild_keys else "R1"
            keys = ("ok", "error", "seconds", "captured", "graphs", "eager", "skipped")
            out["rebuild"] = {"ok": bool(reply.get("ok")), "routes": [route], "spec": spec,
                              "seconds": {route: dur, "R1": dur, "recapture": dur},
                              "ranks": {str(r.get("rank")): {k: r.get(k) for k in keys} for r in ranks}}
            if not reply.get("ok"):
                extra = [f for r in ranks for f in r.get("findings", [])] or [
                    F.finding("rebuild_failed", str(reply.get("error")), "recapture on every rank")]
                out["extra"] += [{**f, "participant": "workers"} for f in extra]
                rollback, res_map = {"status": "undone", "participants": {}}, self.txns[txn]["res"]
                deadline, missed = (time.time() + self.phase2_deadline_s if self.phase2_deadline_s else None), set()
                for part in reversed(self.participants):
                    rollback["participants"][part.name] = res_map[part.name]["undo"] = self._bounded(part, "undo", txn, deadline, missed)
                if not all(r["ok"] for r in rollback["participants"].values()): rollback["status"] = "partial"
                out.update(rollback=rollback, outcome="rolled_back")
                timeline.pop("committed", None)
                timeline["gap_end"] = self.clock()
                return out
            self.txns[txn]["rebuilt"] = [route]
        if self.flush_on_commit:
            deadline, missed = (time.time() + self.phase2_deadline_s if self.phase2_deadline_s else None), set()
            for part in self.participants:
                self._bounded(part, "reset", "state", deadline, missed)
        timeline["gap_end"] = self.clock()
        return out

class SglangLab(ConformanceHarness):
    name, scheduler_module, io_module = "sglang", "sglang.srt.managers.scheduler", "sglang.srt.managers.io_struct"
    def __init__(self, config, modules, receipts=None, rpc_timeout=30.0, check_timeout=120.0):
        self.config, self.modules = json.loads(json.dumps(config)), list(modules)
        self.writer = R.ReceiptWriter(receipts) if receipts else None
        self.rpc_timeout, self.check_timeout = float(rpc_timeout), float(check_timeout)
        self.lock, self.costs, self.engine, self.sock_path = threading.RLock(), {}, None, None
        self.rank_conns, self.rank_locks, self.dirty = {}, {}, False
        self._spawn()
    def engine_class(self):
        return importlib.import_module("sglang.srt.entrypoints.engine").Engine
    def default_workload(self):
        return {"prompts": [f"Write one sentence about {t}." for t in ("lighthouses", "rivers", "mountains", "trains")],
                "max_new_tokens": 16, "temperature": 0.0, "ignore_eos": True}
    def run_generate(self, workload):
        merged = {**self.default_workload(), **(workload or {})}
        params = {"temperature": float(merged["temperature"]), "max_new_tokens": int(merged["max_new_tokens"]),
                  "ignore_eos": bool(merged.get("ignore_eos", True))}
        tokens = [x.get("output_ids") or x["text"] for x in self.engine.generate(prompt=merged["prompts"], sampling_params=params)]
        return {"digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16], "tokens": tokens}
    def _spawn(self):
        started, tp_size = time.perf_counter(), int(self.config.get("tp_size", 1))
        self.sock_path = f"/tmp/fsgl-{os.getpid()}-{next(_counters)}.sock"
        if len(self.sock_path) >= 104: raise ValueError(f"AF_UNIX path too long: {self.sock_path}")
        with contextlib.suppress(FileNotFoundError): os.unlink(self.sock_path)
        listener, accepted, errors = Listener(self.sock_path, "AF_UNIX", backlog=max(8, tp_size * 2)), {}, []

        def accept_all():
            try:
                while len(accepted) < tp_size:
                    conn = listener.accept()
                    accepted[int(conn.recv()["rank"])] = conn
            except Exception as exc:
                errors.append(exc)
            finally:
                with contextlib.suppress(OSError): listener.close()

        accept_thread = threading.Thread(target=accept_all, daemon=True)
        accept_thread.start()
        base_cls = self.engine_class()
        entry = functools.partial(agent.run_scheduler_entry, self.sock_path, self.scheduler_module, base_cls.run_scheduler_process_func)
        self.engine = type(base_cls)("_HookedEngine", (base_cls,), {"run_scheduler_process_func": staticmethod(entry)})(**self.config)
        if hasattr(signal, "SIGQUIT"): signal.signal(signal.SIGQUIT, signal.SIG_IGN)
        with contextlib.suppress(Exception): self.engine.tokenizer_manager._subprocess_watchdog.stop()
        accept_thread.join(self.rpc_timeout)
        if errors or len(accepted) != tp_size:
            self._shutdown_engine()
            raise RuntimeError(f"scheduler handshake failed: {errors or list(accepted)}")
        self.costs["load_s"], self.ranks, self.rank_conns, self.dirty = round(time.perf_counter() - started, 3), tp_size, accepted, False
        self.rank_locks, self.base_text, self.live_text, ver = {r: threading.Lock() for r in range(tp_size)}, {}, [{}], "0"
        with contextlib.suppress(Exception): ver = importlib.metadata.version("sglang")
        participants = [C.CallParticipant(f"rank{r}", lambda m, p, r=r: self.rank_call(r, m, p)) for r in range(tp_size)]
        self.co = RecaptureCoordinator(participants, gap=self.at_gap, writer=self.writer, interface_version="1-draft",
                                       engine={"name": self.name, "version": ver, "commit": None},
                                       phase2_deadline_s=self.rpc_timeout, rebuild=self._rebuild_all_ranks)
        if hasattr(self, "session"): self.session.recovered(self.co)
        else:
            self.session = SS.Session(self.co, self.modules)
    def rank_call(self, rank, method, payload=None, timeout=None):
        if self.dirty: raise ConnectionError("control channel dirty; respawn required")
        conn, deadline = self.rank_conns[rank], timeout if timeout is not None else self.rpc_timeout
        with self.rank_locks[rank]:
            try:
                conn.send({"id": next(_counters), "method": method, "payload": payload})
                if not conn.poll(deadline):
                    self.dirty = True
                    raise TimeoutError(f"rank{rank}: no reply to {method} in {deadline} s")
                return conn.recv()["reply"]
            except (EOFError, OSError) as exc:
                self.dirty = True
                raise ConnectionError(f"rank{rank}: {type(exc).__name__}: {exc}") from exc
    def _fanout(self, method, payload, timeout=None):
        replies, errors = {}, {}
        threads = [threading.Thread(target=lambda r=r: self._run_rank(r, method, payload, timeout, replies, errors), daemon=True)
                   for r in range(self.ranks)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return replies, errors
    def _run_rank(self, rank, method, payload, timeout, replies, errors):
        try:
            replies[rank] = self.rank_call(rank, method, payload, timeout)
        except Exception as exc:
            errors[rank] = exc
    def at_gap(self, fn, timeout=None):
        wait_s, gap_id, io_ref = (timeout if timeout is not None else self.rpc_timeout), f"gap-{next(_counters)}", importlib.import_module(self.io_module)
        req, chan = io_ref.RpcReqInput(method="fastrsi_gap", parameters={"gap_id": gap_id}), self.engine.send_to_rpc
        (io_ref.sock_send if hasattr(io_ref, "sock_send") else lambda ch, r: (ch.send_pyobj if hasattr(ch, "send_pyobj") else ch.send)(r))(chan, req)
        parked, errors = self._fanout("park", gap_id, timeout=wait_s)
        if errors or not all(r.get("ok") for r in parked.values()):
            self.dirty = True
            raise TimeoutError(f"failed to park TP ranks: {next(iter(errors.values()), 'refused')}")
        try:
            return fn()
        finally:
            if not self.dirty:
                self._fanout("release_gap", gap_id, timeout=wait_s)
                ready = (chan.poll(int(wait_s * 1000)) if hasattr(chan, "recv_pyobj") else
                         chan.poll(wait_s) if hasattr(chan, "poll") else bool(select.select([chan], [], [], wait_s)[0]))
                if not ready:
                    self.dirty = True
                    raise TimeoutError(f"RPC barrier timed out after {wait_s} s")
                (io_ref.sock_recv if hasattr(io_ref, "sock_recv") else lambda ch: (ch.recv_pyobj if hasattr(ch, "recv_pyobj") else ch.recv)())(chan)
    def _rebuild_all_ranks(self, txn, spec):
        replies, errors = self._fanout("recapture", {"txn": txn, **spec})
        ranks = [F.not_ok(f"rank{r}: {errors[r]}", [F.finding("rebuild_failed", f"rank{r}: {errors[r]}", "recapture ok")],
                          rank=r) if r in errors else replies[r] for r in range(self.ranks)]
        return {"ok": all(x.get("ok") for x in ranks), "ranks": ranks, "error": "recapture failed"}
    def recapture_extend(self, txn, sizes=None):
        with self.lock:
            if self.session.poisoned(): return F.not_ok("session poisoned", [F.finding("session_poisoned", "poisoned", "clean")])
            ext_id = f"ext-{next(_counters)}"

            def job():
                replies, errors = self._fanout("recapture_extend", {"txn": txn, "sizes": sizes, "extend_id": ext_id})
                ok = not errors and all(r.get("ok") for r in replies.values())
                if not ok: self._fanout("recapture_extend_drop", {"txn": txn, "extend_id": ext_id})
                return {"ok": ok, "ranks": replies, "errors": {r: str(e) for r, e in errors.items()}}

            try:
                return self.at_gap(job, self.rpc_timeout)
            except Exception as exc:
                self.session.fault("participant_failed", f"recapture_extend: {exc}", "gap completes")
                return F.not_ok(str(exc), list(self.session.poisoned()))
    def _shutdown_engine(self):
        for conn in list(self.rank_conns.values()):
            with contextlib.suppress(OSError): conn.close()
        self.rank_conns.clear()
        if self.engine is not None:
            with contextlib.suppress(Exception): self.engine.shutdown()
            self.engine = None
        if self.sock_path:
            with contextlib.suppress(OSError): os.unlink(self.sock_path)
    def close(self):
        with self.lock:
            self._shutdown_engine()
    def _base(self, module_name):
        if module_name not in self.base_text:
            with open(importlib.util.find_spec(module_name).origin) as handle:
                self.base_text[module_name] = handle.read()
        return self.base_text[module_name]
    def install(self, patch):
        started = time.perf_counter()
        with self.lock:
            live, modules = self.live_text[-1], {}
            try:
                for name, new_src in patch["files"].items():
                    modules[name] = {"source": new_src, "base_hash": L.text_hash(live.get(name) or self._base(name))}
            except Exception as exc:
                bad = F.finding("patch_invalid", f"{type(exc).__name__}: {exc}", "importable modules")
                return {"ok": False, "id": None, "outcome": "refused", "codes": ["patch_invalid"],
                        "mechanisms": [], "findings": [bad], "cost_s": round(time.perf_counter() - started, 6)}
            rebuild = patch.get("rebuild")
            if rebuild is None:
                rebuild = ["recapture"] if (patch.get("recapture", True) and any(is_graph_module(m) for m in modules)) else []
            plan = {"modules": modules, "accept": patch.get("accept", []), "undo_resets": patch.get("undo_resets", ["R3"]),
                    "recipes": ["R0 swap"] + (["R1 recapture"] if rebuild else []), "rebuild": list(rebuild),
                    **({"recapture": {"sizes": list(patch["sizes"])}} if "sizes" in patch else {})}
            res = self.session.install(plan, edit_time=patch.get("edit_time"))
            outcome, rec = res.get("outcome") or "refused", res.get("receipt")
            if outcome == "committed": self.live_text.append({**live, **patch["files"]})
            codes = rec["decisions"]["blocking"] if rec else sorted({f["check"] for f in res.get("findings", [])})
            mechs = list(rec["actual_recipes"]) if rec else []
            if outcome == "committed" and rebuild and "R1 recapture" not in mechs: mechs.append("R1 recapture")
            return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome,
                    "codes": [] if outcome == "committed" else codes, "mechanisms": mechs,
                    "findings": rec["findings"] if rec else res.get("findings", []),
                    "receipt": rec, "cost_s": round(time.perf_counter() - started, 6)}
    def _fail_check(self, observed, expected, started):
        self.session.fault("participant_failed", observed, expected)
        return {"ok": False, "error": observed, "output": None, "proven": None, "witness": {},
                "findings": list(self.session.poisoned()), "cost_s": round(time.perf_counter() - started, 6)}
    def check(self, workload=None):
        started, box = time.perf_counter(), {}
        with self.lock:
            for rank in range(self.ranks):
                try:
                    self.rank_call(rank, "ping", None, timeout=min(5.0, self.rpc_timeout))
                except Exception as exc:
                    return self._fail_check(f"rank{rank}: {type(exc).__name__}: {exc}", "every rank answers", started)
            thread = threading.Thread(target=self._run_workload_into, args=(workload, box), daemon=True)
            gen_t0 = time.perf_counter()
            thread.start()
            thread.join(self.check_timeout)
            seconds = round(time.perf_counter() - gen_t0, 4)
            if thread.is_alive() or "error" in box:
                self.dirty = True
                return self._fail_check(f"check: {box.get('error') or f'no result in {self.check_timeout} s'}", "completes", started)
            output = box["output"]
            proofs = [self.co.prove(t, golden={"output_hash": L.text_hash(json.dumps(output, sort_keys=True))}) for t in self.session.stack]
            witness = {}
            for proof in proofs:
                for part, counts in proof["receipt"]["witness"].items():
                    witness.setdefault(part, {}).update(counts)
            return {"ok": True, "output": output, "seconds": seconds, "witness": witness,
                    "proven": all(p["ok"] for p in proofs) if proofs else None,
                    "findings": [f for p in proofs for f in p["receipt"]["findings"]], "cost_s": round(time.perf_counter() - started, 6)}
    def _run_workload_into(self, workload, box):
        try:
            box["output"] = self.run_generate(workload)
        except BaseException as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"
    def reset(self, kind="state"):
        started = time.perf_counter()
        with self.lock:
            if kind == "respawn":
                self._shutdown_engine()
                self._spawn()
                return {"ok": not self.session.poisoned(), "reset": ["worker group", "session"], "cost_s": round(time.perf_counter() - started, 6)}
            try:
                replies = self.at_gap(lambda: [C._call(p, "reset", "state") for p in self.co.participants], self.rpc_timeout)
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "reset": [], "cost_s": round(time.perf_counter() - started, 6)}
            ok = all(r.get("ok") for r in replies)
            return {"ok": ok, "reset": ["radix_cache", "kv_pool", "mamba_pool", "rng"] if ok else [], "cost_s": round(time.perf_counter() - started, 6)}
    def uninstall(self, id=None):
        started = time.perf_counter()
        with self.lock:
            reply = self.session.uninstall(id)
            if reply.get("undone"): del self.live_text[-len(reply["undone"]):]
            return {"ok": reply["ok"], "undone": reply.get("undone", []), "state": self.session.state(),
                    "findings": reply.get("findings", []), "cost_s": round(time.perf_counter() - started, 6)}
    def status(self):
        with self.lock:
            return self.session.status()
