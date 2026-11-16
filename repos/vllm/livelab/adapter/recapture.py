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

Not here: vLLM's V2 model runner (`CudaGraphManager`; R1 says so), ubatched (DBO) graphs,
encoder graphs, drafter-only modes, DP > 1 (the host refuses).
"""
import contextlib
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


def vllm_engine(worker):
    """The Engine of a vLLM V1 GPU worker (None when it runs no graphs)."""
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
    if dispatcher is None:     # the V2 runner keeps graphs in CudaGraphManager.graphs: not bound yet
        raise NotImplementedError(f"R1 knows vLLM's V1 GPUModelRunner; this worker runs "
                                  f"{type(runner).__name__} (VLLM_USE_V2_MODEL_RUNNER=0)")
    if not any(dispatcher.cudagraph_keys.values()):
        return None

    def wrappers():
        return [wrapper for kind in kinds for wrapper in list(kind._all_instances)]

    def capture(mode, desc):
        runner._warmup_and_capture(desc, cudagraph_runtime_mode=mode)
        torch.cuda.synchronize()

    @contextlib.contextmanager
    def session():
        # vLLM's graph_capture() makes a new stream per call, and the caching allocator
        # reuses a pool's free blocks only on the stream that freed them: every recapture
        # then strands the last one's blocks (132 MiB per full recapture, Qwen3-8B). One
        # stream per device for every R1 keeps them reusable. Same groups as graph_capture.
        stream = _STREAMS.get(runner.device)
        if stream is None:
            stream = _STREAMS[runner.device] = torch.cuda.Stream(device=runner.device)
        ctx = parallel_state.GraphCaptureContext(stream)
        set_cudagraph_capturing_enabled(True)
        try:
            with runner._freeze_gc(), contextlib.ExitStack() as stack:
                for group in (parallel_state.get_tp_group(),
                              parallel_state.get_pp_group(),
                              parallel_state.get_dp_group()):
                    stack.enter_context(group.graph_capture(ctx))
                yield
        finally:
            set_cudagraph_capturing_enabled(False)
            runner.maybe_remove_all_loras(runner.lora_config)

    def memory():
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info()
        return {"reserved": torch.cuda.memory_reserved(), "allocated":
                torch.cuda.memory_allocated(), "device_free": free}
    return Engine(wrappers, dispatcher.cudagraph_keys, capture, session, memory,
                  torch.cuda.synchronize)


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
            for mode, descriptors in chosen.items():
                for desc in descriptors:
                    engine.keys[mode].discard(desc)
                    for wrapper, attr in wrappers:
                        if getattr(wrapper, "runtime_mode", None) in (None, mode):
                            getattr(wrapper, attr).pop(desc, None)
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
            for mode, desc in done:
                engine.keys[mode].discard(desc)
                for wrapper, attr in wrappers:
                    if getattr(wrapper, "runtime_mode", None) in (None, mode):
                        getattr(wrapper, attr).pop(desc, None)
            engine.sync()
            return F.not_ok(f"R1 capture: {type(e).__name__}: {e}", [F.finding(
                "rebuild_failed", f"capture of {name(done[-1][0]) if done else '?'} "
                f"{_size(done[-1][1]) if done else '?'} raised {type(e).__name__}: {e}",
                "every chosen descriptor captured")], captured=timings)
        return F.ok(report={"captured": timings, "graphs": len(timings),
                            "seconds": round(time.perf_counter() - t0, 6)})
