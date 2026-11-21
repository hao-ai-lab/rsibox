"""Selective CUDA-graph recapture (recipe R1) on one vLLM worker rank.

A graph replays the kernels it recorded and runs no Python, so a committed edit of code that
runs inside captured graphs (a Python op, or a Triton kernel rebuilt in place by `jit.py`)
changes nothing a replay does. R1 records the graphs again from the new code, in the same step
gap as the commit, without torch.compile (no Dynamo, no Inductor: graphs record the compiled
code as it is, and the compiled code calls the swapped op or the rebuilt kernel).

How vLLM keeps graphs (V1, `vllm/compilation/cuda_graph.py`, `v1/cudagraph_dispatcher.py`):

- Every `CUDAGraphWrapper` (one FULL wrapper around the model, one PIECEWISE wrapper per piece
  of a compiled model; `BreakableCUDAGraphWrapper` for breakable graphs) keeps a dict
  `BatchDescriptor -> entry` (`concrete_cudagraph_entries`, or `entries`). A missing entry is
  captured on the next call in that mode, which is legal only while capture is enabled.
- The dispatcher's `cudagraph_keys[mode]` is the only list of descriptors a step may use; a
  step whose descriptor is not there runs in the next mode that has it, or eager.
- All graphs share one memory pool and never run at the same time, so a capture may reuse
  memory other graphs use between their own kernels.

What R1 does, per mode it recaptures (default: every mode with graphs; `modes` narrows it):

1. It puts a new, empty dict on every wrapper of that mode (the old dict is kept, untouched)
   and empties the mode's dispatcher keys, so no step can replay a graph of the old code.
2. It captures the chosen descriptors (`sizes`: num_tokens values; default every one), largest
   first, PIECEWISE before FULL, as `capture_model` does (`_warmup_and_capture`: warm-up runs,
   then the capture), with capture enabled, under `witness.capture()`: calls made here count
   as capture calls, never as live steps. Each descriptor goes back into the dispatcher's keys
   when its graphs exist. Pieces of one descriptor are captured together: a piece reads the
   previous piece's output at the address it recorded, so they are never mixed across edits.
3. A descriptor it did not choose stays out of the keys: such steps run eager (or in a mode R1
   did not touch), on the new code. `extend` captures more of them later, in a gap.

Undo is a journal entry (recorded after the capture, in the transaction's journal segment):
put the old dicts and keys back. The base graphs never stopped existing, so undo captures
nothing and costs microseconds; the new graphs are dropped and their pool memory is reused by
the next capture. A capture that fails puts the old state back before it returns.

Not here: ubatched (DBO) graphs, encoder graphs, drafter-only modes, DP > 1 (the host refuses).
V2 model runner (`ModelCudaGraphManager`) is supported via `_vllm_v2_engine` alongside V1.
"""
import contextlib
import sys
import time

from kernel import findings as F

ENTRY_ATTRS = ("concrete_cudagraph_entries", "entries")
ORDER = ("PIECEWISE", "FULL")      # capture_model's order


def name(mode):
    return getattr(mode, "name", None) or str(mode)


class Engine:
    """What R1 needs from one worker's model runner. vLLM: `vllm_engine(worker)`; the golden
    tests give a stand-in with the same semantics.

    wrappers() -> [graph wrappers]; keys: {mode: set of descriptors} (the dispatcher's own
    sets, changed in place); capture(mode, desc): warm up and capture one descriptor;
    session(): context in which capture is legal; memory() -> {name: bytes}."""

    def __init__(self, wrappers, keys, capture, session=contextlib.nullcontext,
                 memory=dict, sync=lambda: None):
        self.wrappers, self.keys, self.capture = wrappers, keys, capture
        self.session, self.memory, self.sync = session, memory, sync


_STREAMS = {}         # device -> the capture stream every R1 on it uses


def _cuda_memory(torch_mod):
    torch_mod.cuda.synchronize()
    free, _total = torch_mod.cuda.mem_get_info()
    return {"reserved": torch_mod.cuda.memory_reserved(),
            "allocated": torch_mod.cuda.memory_allocated(), "device_free": free}


@contextlib.contextmanager
def _capture_session(device, torch_mod, parallel_state, set_capturing, on_enter, on_exit):
    # vLLM's graph_capture() makes a new stream per call, stranding pool blocks (~132 MiB/capture);
    # reusing one stream per device keeps the caching allocator's pool blocks reusable.
    stream = _STREAMS.get(device)
    if stream is None:
        stream = _STREAMS[device] = torch_mod.cuda.Stream(device=device)
    ctx = parallel_state.GraphCaptureContext(stream)
    set_capturing(True)
    try:
        with contextlib.ExitStack() as stack:
            on_enter(stack, ctx)
            for grp in (parallel_state.get_tp_group(), parallel_state.get_pp_group(),
                        parallel_state.get_dp_group()):
                stack.enter_context(grp.graph_capture(ctx))
            yield
    finally:
        set_capturing(False)
        on_exit()


class _V2FullWrapper:
    """Exposes V2 `ModelCudaGraphManager.graphs` as a FULL graph wrapper for R1."""

    def __init__(self, mgr, full_mode):
        self._mgr, self.runtime_mode = mgr, full_mode

    @property
    def entries(self):
        return self._mgr.graphs

    @entries.setter
    def entries(self, value):
        self._mgr.graphs = value


class _V2KeySet(set):
    """Mode key set over `mgr._fastrsi_orig_candidates`; mutations re-filter `mgr._candidates`."""

    def __init__(self, mgr, mode, initial=()):
        super().__init__(initial)
        self._mgr, self._mode = mgr, mode

    def _sync(self):
        keys = self._mgr._fastrsi_r1_keys
        for k, descs in self._mgr._fastrsi_orig_candidates.items():
            self._mgr._candidates[k] = [
                d for d in descs if d.cg_mode not in keys or d in keys[d.cg_mode]
            ]

    def add(self, elem): super().add(elem); self._sync()
    def discard(self, elem): super().discard(elem); self._sync()
    def remove(self, elem): super().remove(elem); self._sync()
    def clear(self): super().clear(); self._sync()
    def update(self, *others): super().update(*others); self._sync()


def _v2_helpers(runner):
    mr = sys.modules.get(type(runner).__module__)
    def _pick(mod_name, attr, fallback):
        try:
            return getattr(__import__(mod_name, fromlist=[attr]), attr)
        except (ImportError, AttributeError):
            return getattr(mr, attr, None) or fallback
    freeze_gc = _pick("vllm.utils.gc_utils", "freeze_gc_for_cudagraph_capture",
                      getattr(runner, "_freeze_gc", contextlib.nullcontext))
    lora_hook = _pick("vllm.v1.worker.gpu.lora_utils", "create_lora_capture_hook", lambda *_a: None)
    try:
        import vllm.v1.worker.gpu.cudagraph_utils as cg_utils
    except ImportError:
        cg_utils = None
    return freeze_gc, lora_hook, cg_utils


def _vllm_v2_engine(runner, mgr, kinds, torch_mod, set_capturing, parallel_state):
    """The Engine of a vLLM V2 GPU worker (`ModelCudaGraphManager`; None when no graphs)."""
    if not mgr.needs_capture() or (not getattr(mgr, "_fastrsi_r1_keys", None)
                                   and not getattr(mgr, "_graphs_captured", False)
                                   and not getattr(mgr, "graphs", None)):
        return None
    if getattr(mgr, "ubatch_runner", None) is not None:
        raise NotImplementedError("R1 does not support V2 ubatched (DBO) CUDA graphs")
    ms = getattr(runner, "model_state", None)
    enc = getattr(ms, "encoder_runner", None) if getattr(ms, "supports_mm_inputs", False) else None
    if enc is not None and callable(getattr(enc, "has_cudagraph", None)) and enc.has_cudagraph():
        raise NotImplementedError("R1 does not support V2 encoder CUDA graphs")

    if not hasattr(mgr, "_fastrsi_r1_keys"):
        mgr._fastrsi_orig_candidates = {k: list(v) for k, v in mgr._candidates.items()}
        by_mode = {}
        for descs in mgr._fastrsi_orig_candidates.values():
            for d in descs:
                if name(d.cg_mode) != "NONE":
                    by_mode.setdefault(d.cg_mode, set()).add(d)
        mgr._fastrsi_r1_keys = {m: _V2KeySet(mgr, m, ds) for m, ds in by_mode.items()}
    r1_keys = mgr._fastrsi_r1_keys
    if not any(r1_keys.values()) and not any(mgr._fastrsi_orig_candidates.values()):
        return None
    full_mode = next((m for m in r1_keys if name(m) == "FULL"), None)
    full_wrapper = _V2FullWrapper(mgr, full_mode) if full_mode is not None else None
    freeze_gc, lora_hook, cg_utils = _v2_helpers(runner)

    def wrappers():
        cands = ([full_wrapper] if full_wrapper else []) + [
            w for k in kinds for w in list(k._all_instances)] + [getattr(mgr, "breakable_cg_runner", None)]
        return list({id(w): w for w in cands if w is not None and _attr(w)}.values())

    def capture(mode, desc):
        orig = mgr._capture_descs
        mgr._capture_descs = {mode: [desc]}
        try:
            pcp = getattr(runner, "pcp_manager", None)
            bufs = pcp.input_buffers if pcp is not None else runner.input_buffers
            lora_cfg = getattr(runner, "lora_config", None)
            mgr.capture(
                runner.model, runner.model_state, bufs, runner.intermediate_tensors,
                runner.block_tables, runner.attn_groups, runner.kv_cache_config,
                pcp_manager=pcp, has_lora=lora_cfg is not None,
                use_aux_hidden_state_outputs=getattr(runner, "use_aux_hidden_state_outputs", False),
                lora_capture_hook=lora_hook(lora_cfg, runner))
            torch_mod.cuda.synchronize()
        finally:
            mgr._capture_descs = orig

    def session():
        mods = list({id(m): m for m in (sys.modules.get(type(mgr).__module__), cg_utils)
                     if m is not None and hasattr(m, "graph_capture")}.values())
        saved = []

        def on_enter(stack, ctx):
            for m in mods:
                saved.append((m, m.graph_capture))
                m.graph_capture = lambda *_a, _c=ctx, **_k: contextlib.nullcontext(_c)
            stack.enter_context(freeze_gc())
            setup_loras = getattr(runner, "maybe_setup_dummy_loras", None)
            if callable(setup_loras) and (cm := setup_loras(getattr(runner, "lora_config", None))) is not None:
                stack.enter_context(cm)

        def on_exit():
            for m, orig_gc in saved:
                m.graph_capture = orig_gc
            kv = getattr(runner, "kv_connector", None)
            if kv is not None and callable(getattr(kv, "reset_capture_state", None)):
                kv.reset_capture_state()

        return _capture_session(runner.device, torch_mod, parallel_state, set_capturing, on_enter, on_exit)

    return Engine(wrappers, r1_keys, capture, session, lambda: _cuda_memory(torch_mod),
                  torch_mod.cuda.synchronize)


def vllm_engine(worker):
    """The Engine of a vLLM V1 or V2 GPU worker (None when it runs no graphs)."""
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed import parallel_state
    kinds = [CUDAGraphWrapper]
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        kinds.append(BreakableCUDAGraphWrapper)
    except ImportError:
        pass
    runner = worker.model_runner
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        mgr = getattr(runner, "cudagraph_manager", None)
        if mgr is not None:
            return _vllm_v2_engine(
                runner, mgr, kinds, torch, set_cudagraph_capturing_enabled, parallel_state
            )
        raise NotImplementedError(f"R1 knows vLLM's V1/V2 GPUModelRunner; this worker runs "
                                  f"{type(runner).__name__}")
    if not any(dispatcher.cudagraph_keys.values()):
        return None

    def wrappers():
        return [wrapper for kind in kinds for wrapper in list(kind._all_instances)]

    def capture(mode, desc):
        runner._warmup_and_capture(desc, cudagraph_runtime_mode=mode)
        torch.cuda.synchronize()

    def session():
        return _capture_session(
            runner.device, torch, parallel_state, set_cudagraph_capturing_enabled,
            lambda stack, _ctx: stack.enter_context(runner._freeze_gc()),
            lambda: runner.maybe_remove_all_loras(runner.lora_config),
        )

    return Engine(wrappers, dispatcher.cudagraph_keys, capture, session,
                  lambda: _cuda_memory(torch), torch.cuda.synchronize)


def _attr(wrapper):
    return next((attr for attr in ENTRY_ATTRS if isinstance(getattr(wrapper, attr, None), dict)), None)


def _size(desc):
    return getattr(desc, "num_tokens", desc)


def graphs(engine):
    """{"keys": {mode: [num_tokens]}, "wrappers": [{mode, entries}]}: what the worker can
    replay now (the `graphs` op; tests and receipts compare it before and after)."""
    if engine is None:
        return {"keys": {}, "wrappers": []}
    wrappers = [(wrapper, _attr(wrapper)) for wrapper in engine.wrappers()]
    return {"keys": {name(mode): sorted((_size(desc) for desc in descriptors), reverse=True)
                     for mode, descriptors in engine.keys.items()},
            "wrappers": [{"mode": name(getattr(wrapper, "runtime_mode", None)),
                          "entries": len(getattr(wrapper, attr))}
                         for wrapper, attr in wrappers if attr]}


def _drop_descriptors(engine, wrappers, pairs):
    for mode, desc in pairs:
        engine.keys[mode].discard(desc)
        for wrapper, attr in wrappers:
            if getattr(wrapper, "runtime_mode", None) in (None, mode):
                getattr(wrapper, attr).pop(desc, None)


class Recapture:
    """R1 state of one rank: what each transaction replaced (for extend)."""

    def __init__(self):
        self.live = {}       # txn -> {"modes", "chosen": {mode: set}, "left": {mode: [desc]}}

    def _select(self, engine, modes, sizes):
        """Partition dispatcher keys and wrapper dicts into chosen and left sets."""
        keys = engine.keys
        selected = [mode for mode in keys if keys[mode] and (modes is None or name(mode) in modes)]
        old_keys = {mode: set(keys[mode]) for mode in selected}
        wrappers = [(wrapper, _attr(wrapper)) for wrapper in engine.wrappers()
                    if _attr(wrapper) and (getattr(wrapper, "runtime_mode", None) is None
                                           or getattr(wrapper, "runtime_mode") in selected)]
        old_entries = [(wrapper, attr, getattr(wrapper, attr)) for wrapper, attr in wrappers]
        want = None if sizes is None else {int(s) for s in sizes}
        chosen = {mode: [desc for desc in old_keys[mode] if want is None or _size(desc) in want]
                  for mode in selected}
        left = {mode: [desc for desc in old_keys[mode] if desc not in chosen[mode]]
                for mode in selected}
        return selected, old_keys, wrappers, old_entries, chosen, left

    def _swap_dicts(self, wrappers, selected, engine):
        """Install empty entry dicts on selected wrappers and clear dispatcher keys."""
        for wrapper, attr in wrappers:
            setattr(wrapper, attr, {})
        for mode in selected:
            engine.keys[mode].clear()

    def run(self, engine, txn, journal, witness, modes=None, sizes=None):
        """Recapture after txn's commit. Returns an ok-dict with per-descriptor seconds and
        memory before/after; on failure the old graphs are back and the reply is not-ok."""
        if engine is None:
            return F.ok(skipped="this worker runs no graphs", seconds=0.0)
        if txn in self.live:
            return F.not_ok(f"R1 already ran for {txn}; use extend")
        if not journal.marks or journal.marks[-1][0] != txn:
            return F.not_ok(f"{txn} is not this rank's newest committed transaction: its undo "
                            "would not restore the graphs R1 replaces")

        selected, old_keys, wrappers, old_entries, chosen, left = self._select(
            engine, modes, sizes
        )

        def restore():
            for wrapper, attr, old_dict in old_entries:
                setattr(wrapper, attr, old_dict)
            for mode in selected:
                engine.keys[mode].clear()
                engine.keys[mode].update(old_keys[mode])

        mem0 = engine.memory()
        try:
            self._swap_dicts(wrappers, selected, engine)
            result = self._capture(engine, chosen, witness)
            if not result["ok"]:
                restore()
                engine.sync()
                return result
        except BaseException:
            restore()
            engine.sync()
            raise

        journal.record_call(restore, f"R1 graphs of {txn}")
        self.live[txn] = {
            "left": left,
            "restore": restore,
            "wrappers": wrappers,
            "old_keys": old_keys,
        }

        def forget():
            self.live.pop(txn, None)
        journal.record_call(forget, f"R1 state of {txn}")

        return F.ok(**result["report"], modes=[name(mode) for mode in selected],
                    wrappers=len(wrappers),
                    eager={name(mode): sorted((_size(desc) for desc in descriptors), reverse=True)
                           for mode, descriptors in left.items() if descriptors},
                    memory={"before": mem0, "after": engine.memory()})

    def extend(self, engine, txn, journal, witness, sizes=None, extend_id=None):
        """Capture more of the descriptors R1 left out for txn (default: all of them).
        extend_id names this extend so `drop_extend` can take it back (another rank's failed)."""
        state = self.live.get(txn)
        if state is None:
            return F.not_ok(f"no R1 for {txn} on this rank")
        state.pop("extend", None)      # only the newest extend can be dropped
        if not journal.marks or journal.marks[-1][0] != txn:
            return F.not_ok(f"{txn} is not this rank's newest committed transaction")
        want = None if sizes is None else {int(s) for s in sizes}
        chosen = {mode: [desc for desc in descriptors if want is None or _size(desc) in want]
                  for mode, descriptors in state["left"].items()}
        result = self._capture(engine, chosen, witness)
        if not result["ok"]:
            return result
        for mode, descriptors in chosen.items():
            state["left"][mode] = [desc for desc in state["left"][mode] if desc not in descriptors]
        wrappers = state["wrappers"]

        def unextend():
            _drop_descriptors(engine, wrappers,
                              [(mode, desc) for mode, descs in chosen.items() for desc in descs])
            for mode in chosen:
                state["left"][mode] = [desc for desc in state["old_keys"][mode]
                                       if desc not in engine.keys[mode]]

        journal.record_call(unextend, f"R1 extend of {txn}")
        if extend_id is not None:
            state["extend"] = (extend_id, journal.entries[-1], unextend)
        return F.ok(**result["report"])

    def drop_extend(self, txn, journal, extend_id):
        """Take back the extend `extend_id` of txn on this rank: its graphs and keys go, its
        descriptors are left out again and its journal entry is removed. Only the newest
        extend, while its entry is still the journal's newest. A rank where that extend did
        not capture anything (it failed, or never ran) has nothing to drop: ok, dropped False."""
        state = self.live.get(txn) or {}
        last = state.get("extend")
        if last is None or last[0] != extend_id:
            return F.ok(dropped=False)
        _, entry, unextend = state.pop("extend")
        if not journal.entries or journal.entries[-1] is not entry:
            return F.not_ok(f"extend {extend_id} of {txn} is no longer this rank's newest change")
        journal.entries.pop()
        unextend()
        return F.ok(dropped=True)

    def _capture(self, engine, chosen, witness):
        """Capture `chosen` ({mode: [desc]}); every key goes in just before its capture (the
        runner's dummy run asks the dispatcher for the mode). On error the keys of this call
        come out again, so no step dispatches to a half-captured descriptor."""
        done, timings = [], []
        t0 = time.perf_counter()
        order = sorted(chosen, key=lambda mode: ORDER.index(name(mode)) if name(mode) in ORDER else 9)
        try:
            with witness.capture(), engine.session():
                for mode in order:
                    for desc in sorted(chosen[mode], key=_size, reverse=True):
                        t = time.perf_counter()
                        engine.keys[mode].add(desc)
                        done.append((mode, desc))
                        engine.capture(mode, desc)
                        timings.append([name(mode), _size(desc), round(time.perf_counter() - t, 6)])
        except Exception as e:
            wrappers = [(wrapper, _attr(wrapper)) for wrapper in engine.wrappers() if _attr(wrapper)]
            _drop_descriptors(engine, wrappers, done)
            engine.sync()
            return F.not_ok(f"R1 capture: {type(e).__name__}: {e}", [F.finding(
                "rebuild_failed", f"capture of {name(done[-1][0]) if done else '?'} "
                f"{_size(done[-1][1]) if done else '?'} raised {type(e).__name__}: {e}",
                "every chosen descriptor captured")], captured=timings)
        return F.ok(report={"captured": timings, "graphs": len(timings),
                            "seconds": round(time.perf_counter() - t0, 6)})
