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
    from vllm.distributed import parallel_state as ps
    kinds = [CUDAGraphWrapper]
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        kinds.append(BreakableCUDAGraphWrapper)
    except ImportError:
        pass
    runner = worker.model_runner
    disp = getattr(runner, "cudagraph_dispatcher", None)
    if disp is None:     # the V2 runner keeps graphs in CudaGraphManager.graphs: not bound yet
        raise NotImplementedError(f"R1 knows vLLM's V1 GPUModelRunner; this worker runs "
                                  f"{type(runner).__name__} (VLLM_USE_V2_MODEL_RUNNER=0)")
    if not any(disp.cudagraph_keys.values()):
        return None

    def wrappers():
        return [w for k in kinds for w in list(k._all_instances)]

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
        ctx = ps.GraphCaptureContext(stream)
        set_cudagraph_capturing_enabled(True)
        try:
            with runner._freeze_gc(), contextlib.ExitStack() as st:
                for g in (ps.get_tp_group(), ps.get_pp_group(), ps.get_dp_group()):
                    st.enter_context(g.graph_capture(ctx))
                yield
        finally:
            set_cudagraph_capturing_enabled(False)
            runner.maybe_remove_all_loras(runner.lora_config)

    def memory():
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info()
        return {"reserved": torch.cuda.memory_reserved(), "allocated":
                torch.cuda.memory_allocated(), "device_free": free}
    return Engine(wrappers, disp.cudagraph_keys, capture, session, memory,
                  torch.cuda.synchronize)


def _attr(w):
    return next((a for a in ENTRY_ATTRS if isinstance(getattr(w, a, None), dict)), None)


def _size(d):
    return getattr(d, "num_tokens", d)


def graphs(eng):
    """{"keys": {mode: [num_tokens]}, "wrappers": [{mode, entries}]}: what the worker can
    replay now (the `graphs` op; tests and receipts compare it before and after)."""
    if eng is None:
        return {"keys": {}, "wrappers": []}
    ws = [(w, _attr(w)) for w in eng.wrappers()]
    return {"keys": {name(m): sorted((_size(d) for d in ds), reverse=True)
                     for m, ds in eng.keys.items()},
            "wrappers": [{"mode": name(getattr(w, "runtime_mode", None)),
                          "entries": len(getattr(w, a))} for w, a in ws if a]}


class Recapture:
    """R1 state of one rank: what each transaction replaced (for extend)."""

    def __init__(self):
        self.live = {}       # txn -> {"modes", "chosen": {mode: set}, "left": {mode: [desc]}}

    def run(self, eng, txn, journal, witness, modes=None, sizes=None):
        """Recapture after txn's commit. Returns an ok-dict with per-descriptor seconds and
        memory before/after; on failure the old graphs are back and the reply is not-ok."""
        if eng is None:
            return F.ok(skipped="this worker runs no graphs", seconds=0.0)
        if txn in self.live:
            return F.not_ok(f"R1 already ran for {txn}; use extend")
        if not journal.marks or journal.marks[-1][0] != txn:
            return F.not_ok(f"{txn} is not this rank's newest committed transaction: its undo "
                            "would not restore the graphs R1 replaces")
        keys = eng.keys
        sel = [m for m in keys if keys[m] and (modes is None or name(m) in modes)]
        old_keys = {m: set(keys[m]) for m in sel}
        ws = [(w, _attr(w)) for w in eng.wrappers()
              if _attr(w) and (getattr(w, "runtime_mode", None) is None
                               or getattr(w, "runtime_mode") in sel)]
        old = [(w, a, getattr(w, a)) for w, a in ws]
        want = None if sizes is None else {int(s) for s in sizes}

        def restore():
            for w, a, d in old:
                setattr(w, a, d)
            for m in sel:
                keys[m].clear()
                keys[m].update(old_keys[m])
        mem0 = eng.memory()
        try:
            for w, a in ws:
                setattr(w, a, {})
            for m in sel:
                keys[m].clear()
            chosen = {m: [d for d in old_keys[m] if want is None or _size(d) in want] for m in sel}
            left = {m: [d for d in old_keys[m] if d not in chosen[m]] for m in sel}
            r = self._capture(eng, chosen, witness)
            if not r["ok"]:
                restore()
                eng.sync()
                return r
        except BaseException:
            restore()
            eng.sync()
            raise
        journal.record_call(restore, f"R1 graphs of {txn}")
        self.live[txn] = {"left": left, "restore": restore, "ws": ws, "old_keys": old_keys}

        def forget():
            self.live.pop(txn, None)
        journal.record_call(forget, f"R1 state of {txn}")
        return F.ok(**r["report"], modes=[name(m) for m in sel], wrappers=len(ws),
                    eager={name(m): sorted((_size(d) for d in ds), reverse=True)
                           for m, ds in left.items() if ds},
                    memory={"before": mem0, "after": eng.memory()})

    def extend(self, eng, txn, journal, witness, sizes=None):
        """Capture more of the descriptors R1 left out for txn (default: all of them)."""
        st = self.live.get(txn)
        if st is None:
            return F.not_ok(f"no R1 for {txn} on this rank")
        if not journal.marks or journal.marks[-1][0] != txn:
            return F.not_ok(f"{txn} is not this rank's newest committed transaction")
        want = None if sizes is None else {int(s) for s in sizes}
        chosen = {m: [d for d in ds if want is None or _size(d) in want]
                  for m, ds in st["left"].items()}
        r = self._capture(eng, chosen, witness)
        if not r["ok"]:
            return r
        for m, ds in chosen.items():
            st["left"][m] = [d for d in st["left"][m] if d not in ds]
        ws = st["ws"]

        def unextend():
            for m, ds in chosen.items():
                for d in ds:
                    eng.keys[m].discard(d)
                    for w, a in ws:
                        if getattr(w, "runtime_mode", None) in (None, m):
                            getattr(w, a).pop(d, None)
                st["left"][m] = [d for d in st["old_keys"][m] if d not in eng.keys[m]]
        journal.record_call(unextend, f"R1 extend of {txn}")
        return F.ok(**r["report"])

    def _capture(self, eng, chosen, witness):
        """Capture `chosen` ({mode: [desc]}); every key goes in just before its capture (the
        runner's dummy run asks the dispatcher for the mode). On error the keys of this call
        come out again, so no step dispatches to a half-captured descriptor."""
        done, per = [], []
        t0 = time.perf_counter()
        order = sorted(chosen, key=lambda m: ORDER.index(name(m)) if name(m) in ORDER else 9)
        try:
            with witness.capture(), eng.session():
                for m in order:
                    for d in sorted(chosen[m], key=_size, reverse=True):
                        t = time.perf_counter()
                        eng.keys[m].add(d)
                        done.append((m, d))
                        eng.capture(m, d)
                        per.append([name(m), _size(d), round(time.perf_counter() - t, 6)])
        except Exception as e:
            ws = [(w, _attr(w)) for w in eng.wrappers() if _attr(w)]
            for m, d in done:
                eng.keys[m].discard(d)
                for w, a in ws:
                    if getattr(w, "runtime_mode", None) in (None, m):
                        getattr(w, a).pop(d, None)
            eng.sync()
            return F.not_ok(f"R1 capture: {type(e).__name__}: {e}", [F.finding(
                "rebuild_failed", f"capture of {name(done[-1][0]) if done else '?'} "
                f"{_size(done[-1][1]) if done else '?'} raised {type(e).__name__}: {e}",
                "every chosen descriptor captured")], captured=per)
        return F.ok(report={"captured": per, "graphs": len(per),
                            "seconds": round(time.perf_counter() - t0, 6)})
