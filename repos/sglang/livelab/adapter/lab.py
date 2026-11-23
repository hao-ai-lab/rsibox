"""SglangLab: live-lab verbs over `rsibox.lab.BaseLab` + `RecaptureCoordinator` across TP ranks."""
import contextlib, functools, hashlib, importlib, importlib.metadata, itertools, json, os, select, signal, socket, sys, threading
from multiprocessing.connection import Listener
from adapter import agent
from adapter.fixture.harness import ConformanceHarness
from rsibox import ipc
from rsibox.lab import BaseLab, RecaptureCoordinator
from kernel import coordinator as C, findings as F, locate as L

EAGER_LAYERS = (".srt.layers.sampler", ".srt.layers.logits_processor")
_counters = itertools.count(1)
_FORK_SUBPROCS_LOCK = threading.Lock()
_FORK_SUBPROCS_STATE = {"preloaded": False, "clear_nvml_env": False}

def is_graph_module(module_name):
    return not module_name.endswith(EAGER_LAYERS) and (".srt.models." in module_name or ".srt.layers." in module_name)

class SglangLab(ConformanceHarness, BaseLab):
    name, scheduler_module, io_module = "sglang", "sglang.srt.managers.scheduler", "sglang.srt.managers.io_struct"
    threaded_check = True
    def __init__(self, config, modules, receipts=None, rpc_timeout=30.0, check_timeout=120.0, fork_subprocs=False,
                 weight_cache_dir=None, weight_store=None, recovery_driver=None):
        self.config, self._init_modules = json.loads(json.dumps(config)), list(modules)
        self.engine, self.sock_path, self.ranks, self.rank_conns, self.rank_locks = None, None, 0, {}, {}
        self.fork_subprocs, self.fork_subprocs_fallback, self._fork_subprocs_restore = bool(fork_subprocs), [], None
        raw_wc = os.fspath(weight_cache_dir).strip() if weight_cache_dir else ""
        self.weight_cache_dir = os.path.abspath(raw_wc) if raw_wc else None
        if self.weight_cache_dir and weight_store is not None and getattr(weight_store, "name", "none") != "none":
            raise ValueError(
                f"cannot pass both weight_cache_dir and weight_store ({getattr(weight_store, 'name', type(weight_store).__name__)})"
            )
        self._weight_cache_env_restore = None
        super().__init__(
            receipts=receipts,
            check_timeout=check_timeout,
            rpc_timeout=rpc_timeout,
            weight_store=weight_store,
            recovery_driver=recovery_driver,
        )
    @property
    def dirty(self): return self._dirty
    @dirty.setter
    def dirty(self, value): self._dirty = bool(value)
    _base = BaseLab._base_source
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
    @staticmethod
    def _preload_fork_subprocs():
        import forksafe
        from services import zygote as sgl_zygote

        reasons = forksafe.poison_reasons()
        if reasons:
            return False, list(reasons)
        with _FORK_SUBPROCS_LOCK:
            if not _FORK_SUBPROCS_STATE["preloaded"]:
                guards, restore_triton = sgl_zygote.install_guards(("torch", *sgl_zygote.SUBPROC_PRELOAD))
                _FORK_SUBPROCS_STATE["clear_nvml_env"] = bool(guards.get("nvml_env_set"))
                try:
                    sgl_zygote.preload(sgl_zygote.SUBPROC_PRELOAD)
                except SystemExit as exc:
                    forksafe.unpatch_child(None, clear_env=_FORK_SUBPROCS_STATE["clear_nvml_env"])
                    _FORK_SUBPROCS_STATE["clear_nvml_env"] = False
                    return False, list(forksafe.poison_reasons() or [str(exc)])
                except BaseException:
                    forksafe.unpatch_child(None, clear_env=_FORK_SUBPROCS_STATE["clear_nvml_env"])
                    _FORK_SUBPROCS_STATE["clear_nvml_env"] = False
                    raise
                finally:
                    restore_triton()
                _FORK_SUBPROCS_STATE["preloaded"] = True
        return True, []

    def _prepare_fork_subprocs(self):
        import multiprocessing
        from services import zygote as sgl_zygote

        if self._fork_subprocs_restore is not None:
            with contextlib.suppress(Exception):
                self._fork_subprocs_restore()
            self._fork_subprocs_restore = None

        self.fork_subprocs_fallback = []
        ok, reasons = self._preload_fork_subprocs()
        if not ok:
            self.fork_subprocs_fallback = reasons
            return False

        prev_method = multiprocessing.get_start_method(allow_none=True)
        prev_numa = os.environ.get("SGLANG_NUMA_BIND_V2")
        restore_mp = sgl_zygote.enable_fork_subprocs(
            clear_nvml_env=_FORK_SUBPROCS_STATE["clear_nvml_env"]
        )

        def _cleanup():
            if restore_mp is not None:
                restore_mp()
            with contextlib.suppress(RuntimeError, ValueError):
                multiprocessing.set_start_method(prev_method, force=True)
            if prev_numa is None:
                os.environ.pop("SGLANG_NUMA_BIND_V2", None)
            else:
                os.environ["SGLANG_NUMA_BIND_V2"] = prev_numa

        self._fork_subprocs_restore = _cleanup
        multiprocessing.set_start_method("spawn", force=True)
        os.environ["SGLANG_NUMA_BIND_V2"] = "0"
        return True

    def _restore_weight_cache_env(self):
        if self._weight_cache_env_restore is not None:
            with contextlib.suppress(Exception):
                self._weight_cache_env_restore()
            self._weight_cache_env_restore = None

    def _prepare_weight_cache(self):
        from services import weights as wc
        tp_size = int(self.config.get("tp_size", 1))
        res = wc.probe(self.weight_cache_dir, tp_size=tp_size)
        finding = wc.daemon_finding(res, self.weight_cache_dir)
        if finding is not None:
            raise RuntimeError(f"weight-cache daemon not ready: {finding['observed']}")
        self._restore_weight_cache_env()
        cenv = wc.client_env(self.weight_cache_dir)
        prev = {k: os.environ.get(k) for k in cenv}
        os.environ.update(cenv)
        def _restore():
            for k, old in prev.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
        self._weight_cache_env_restore = _restore

    @staticmethod
    def _build_gms_server_args(cfg):
        """Construct SGLang `ServerArgs` and bind Dynamo GMS via `setup_gms()` + `declare_resolution()`.

        Follows upstream `gpu_memory_service/examples/sglang_gms_smoke.py`: bare SGLang
        rejects `load_format="gms"` unless `setup_gms(server_args)` registers `GMSModelLoader`
        and enables the GMS memory saver via `declare_resolution()` (SGLang >=0.5.21).
        """
        try:
            from sglang.srt.arg_groups.overrides import declare_resolution
            from sglang.srt.server_args import ServerArgs
        except ImportError as exc:
            raise RuntimeError(
                "Dynamo GMS for SGLang requires SGLang >=0.5.21 exposing "
                "sglang.srt.arg_groups.overrides.declare_resolution()"
            ) from exc
        try:
            from gpu_memory_service.integrations.sglang import setup_gms
        except ImportError as exc:
            raise RuntimeError(
                "Dynamo GMS for SGLang requires `gpu_memory_service` installed "
                "(or passed via DynamoGmsStore.extra_pythonpath)"
            ) from exc

        init_keys = {k: cfg[k] for k in ("model_path", "host", "port") if k in cfg}
        overrides = {
            k: v for k, v in cfg.items()
            if k not in ("model_path", "host", "port", "load_format")
        }
        server_args = ServerArgs(**init_keys)
        if overrides:
            declare_resolution(server_args, "rsibox.gms", **overrides)
        loader_cls = setup_gms(server_args)
        declare_resolution(server_args, "rsibox.gms.loader", load_format=loader_cls)
        # Upstream setup_gms() reads raw getattr(server_args, "model_loader_extra_config")
        # which stays '{}' under SGLang >=0.5.21 declare_resolution(); sync from resolved overrides.
        agent.sync_sglang_gms_globals(server_args)
        return server_args

    def _prepare_weight_store(self):
        """Prepare an optional pluggable `rsibox.warm.WeightStore` (e.g. `DynamoGmsStore`)."""
        store = self.weight_store
        if store is None or getattr(store, "name", "none") == "none":
            return None
        tp_size = int(self.config.get("tp_size", 1))
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        uuid_map_json = None
        if store.name == "dynamo_gms" and vis is not None and vis.strip():
            from rsibox.warm.dynamo_gms import ENV_GMS_DEVICE_UUID_MAP, build_device_uuid_map
            resolver = getattr(store, "_resolve_targets", None)
            targets = resolver(visible_devices=vis, tp=tp_size) if callable(resolver) else None
            if targets and any(logical != phys for logical, (phys, _) in enumerate(targets)):
                uuid_map_json = json.dumps(build_device_uuid_map(targets))
        prev_allow_remap = getattr(store, "allow_remapped_visible_devices", False)
        if uuid_map_json and hasattr(store, "allow_remapped_visible_devices"):
            store.allow_remapped_visible_devices = True
        try:
            pr = store.probe(tp=tp_size, visible_devices=vis)
        finally:
            if uuid_map_json and hasattr(store, "allow_remapped_visible_devices"):
                store.allow_remapped_visible_devices = prev_allow_remap
        if not pr.ok:
            raise RuntimeError(f"weight store ({store.name}) not ready: {pr.error}")
        explicit = {
            k for k in ("load_format", "model_loader_extra_config")
            if self.config.get(k) is not None
        }
        delta = store.launch_delta(engine="sglang", explicit_flags=explicit, explicit_values=self.config)
        if not delta.ok:
            raise RuntimeError(f"weight store ({store.name}) config conflict: {delta.findings[0]['observed']}")

        # Restore any previous store env/sys.path before re-computing additions (safe on re-entry).
        self._restore_weight_cache_env()
        extra_py = [p for p in (getattr(store, "extra_pythonpath", ()) or ()) if p]
        env_updates = dict(delta.env or {})
        if uuid_map_json:
            env_updates[ENV_GMS_DEVICE_UUID_MAP] = uuid_map_json
        added_sys_paths = []
        if extra_py:
            cur_py = [p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
            for entry in extra_py:
                if entry not in cur_py:
                    cur_py.append(entry)
                if entry not in sys.path:
                    sys.path.append(entry)
                    added_sys_paths.append(entry)
            env_updates["PYTHONPATH"] = os.pathsep.join(cur_py)

        if env_updates or added_sys_paths:
            prev = {k: os.environ.get(k) for k in env_updates}
            os.environ.update(env_updates)
            def _restore():
                for k, old in prev.items():
                    if old is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old
                for entry in reversed(added_sys_paths):
                    with contextlib.suppress(ValueError):
                        sys.path.remove(entry)
            self._weight_cache_env_restore = _restore

        try:
            merged = dict(self.config)
            if store.name == "dynamo_gms":
                for cg_key in ("cuda_graph_backend_decode", "cuda_graph_backend_prefill"):
                    cg_val = merged.get(cg_key)
                    if cg_val is not None and cg_val != "disabled":
                        raise RuntimeError(
                            f"SGLang with Dynamo GMS V0 does not support pauseable CUDA graphs "
                            f"({cg_key}={cg_val!r}); set {cg_key}='disabled' or omit it"
                        )
                if merged.get("disable_cuda_graph") is False:
                    raise RuntimeError(
                        "SGLang with Dynamo GMS V0 does not support pauseable CUDA graphs "
                        "(disable_cuda_graph=False)"
                    )
                merged.setdefault("cuda_graph_backend_decode", "disabled")
                merged.setdefault("cuda_graph_backend_prefill", "disabled")
                merged_extra = store.merged_loader_config(merged.get("model_loader_extra_config"))
                if merged_extra:
                    merged["model_loader_extra_config"] = json.dumps(merged_extra)
                elif "model_loader_extra_config" in merged and merged["model_loader_extra_config"] is None:
                    merged.pop("model_loader_extra_config", None)
                return {"server_args": self._build_gms_server_args(merged)}
            if delta.args:
                raise RuntimeError(
                    f"weight store ({store.name}) returned CLI args {list(delta.args)} "
                    "that SglangLab cannot map automatically"
                )
            return merged
        except BaseException:
            self._restore_weight_cache_env()
            raise

    def _start_engine(self):
        listener, accept_thread, accepted, errors = None, None, {}, []
        try:
            # weight_cache_dir sets env vars only (before fork preload), whereas
            # _prepare_weight_store() imports sglang/torch/GMS and must run AFTER forksafe guards.
            if self.weight_cache_dir:
                self._prepare_weight_cache()
            if self.fork_subprocs:
                self._prepare_fork_subprocs()
            gms_cfg = None
            if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
                gms_cfg = self._prepare_weight_store()
            tp_size = int(self.config.get("tp_size", 1))
            self.sock_path = ipc.short_sock_path("fsgl")
            with contextlib.suppress(FileNotFoundError): os.unlink(self.sock_path)
            listener = Listener(self.sock_path, "AF_UNIX", backlog=max(8, tp_size * 2))

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
            if gms_cfg is not None:
                engine_cfg = gms_cfg
            elif self.weight_cache_dir:
                from services import weights as wc
                engine_cfg = wc.client_config(self.config)
            else:
                engine_cfg = self.config
            self.engine = type(base_cls)("_HookedEngine", (base_cls,), {"run_scheduler_process_func": staticmethod(entry)})(**engine_cfg)
            if hasattr(signal, "SIGQUIT"): signal.signal(signal.SIGQUIT, signal.SIG_IGN)
            with contextlib.suppress(Exception): self.engine.tokenizer_manager._subprocess_watchdog.stop()
            accept_thread.join(self.rpc_timeout)
            if errors or len(accepted) != tp_size:
                raise RuntimeError(f"scheduler handshake failed: {errors or list(accepted)}")
            self.ranks, self.rank_conns, self.dirty = tp_size, accepted, False
            self.rank_locks, self.base_text, ver = {r: threading.Lock() for r in range(tp_size)}, {}, "0"
            with contextlib.suppress(Exception): ver = importlib.metadata.version("sglang")
            participants = [C.CallParticipant(f"rank{r}", lambda m, p, r=r: self.rank_call(r, m, p)) for r in range(tp_size)]
            co = RecaptureCoordinator(participants, gap=self.at_gap, writer=self.writer, interface_version="1-draft",
                                      engine={"name": self.name, "version": ver, "commit": None},
                                      phase2_deadline_s=self.rpc_timeout, rebuild=self._rebuild_all_ranks,
                                      post_commit_reset="state")
            return co, self._init_modules
        except BaseException:
            if listener is not None:
                with contextlib.suppress(AttributeError, OSError):
                    listener._listener._socket.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(OSError):
                    listener.close()
            if accept_thread is not None:
                accept_thread.join(timeout=1.0)
            for conn in list(accepted.values()):
                with contextlib.suppress(OSError):
                    conn.close()
            accepted.clear()
            self._shutdown_engine()
            raise
    def _stop_engine(self):
        self._shutdown_engine()
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
        req, chan = io_ref.RpcReqInput(method="rsibox_gap", parameters={"gap_id": gap_id}), self.engine.send_to_rpc
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
            self.sock_path = None
        if self._fork_subprocs_restore is not None:
            with contextlib.suppress(Exception):
                self._fork_subprocs_restore()
            self._fork_subprocs_restore = None
        self._restore_weight_cache_env()
    def _build_plan(self, patch, live_map):
        modules = {name: {"source": new_src, "base_hash": L.text_hash(live_map.get(name) or self._base(name))}
                   for name, new_src in patch["files"].items()}
        rebuild = patch.get("rebuild")
        if rebuild is None:
            rebuild = ["recapture"] if (patch.get("recapture", True) and any(is_graph_module(m) for m in modules)) else []
        plan = {"modules": modules, "accept": list(patch.get("accept", [])), "undo_resets": list(patch.get("undo_resets", ["R3"])),
                "recipes": ["R0 swap"] + (["R1 recapture"] if rebuild else []), "rebuild": list(rebuild),
                **({"recapture": {"sizes": list(patch["sizes"])}} if "sizes" in patch else {})}
        return plan, []
    def _probe_participants(self):
        for rank in range(self.ranks):
            try: self.rank_call(rank, "ping", None, timeout=min(5.0, self.rpc_timeout))
            except Exception as exc: return False, f"rank{rank}: {type(exc).__name__}: {exc}"
        return True, None
    def _run_workload(self, workload):
        try: return self.run_generate(workload)
        except BaseException:
            self.dirty = True
            raise
    def _reset_state(self):
        replies = self.at_gap(lambda: [C._call(p, "reset", "state") for p in self.co.participants], self.rpc_timeout)
        ok = all(r.get("ok") for r in replies)
        return {"ok": ok, "reset": ["radix_cache", "kv_pool", "mamba_pool", "rng"] if ok else []}
