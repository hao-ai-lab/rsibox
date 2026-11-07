"""Witness: count calls of swapped code, keyed `module:qualname:firstline`.

The swap injects `__fastrsi_hit__("<key>")` as the first statement of each new body (after
the docstring). `__fastrsi_hit__` is a global of the
module, bound to the participant's `Witness.hook` (a plain function that calls `hit`); the
swap binds it through the journal. Undo never deletes the global: code that Dynamo transformed
or a graph captured while the edit was installed may still name it, and a deleted global killed
an engine with NameError at the next step (a live graph-mode run). Undo binds the no-op `stub`
instead (where the swap added the name) and drops the counter keys.

Calls made while the engine records a graph (CUDA graph capture, and the warm-up runs of a
recapture) are counted apart, in `capture_counts`: a graph replay runs no Python, so those
calls prove only that the new code was captured, never that a step ran it. The engine says when
it captures: `Witness(capturing=fn)` asks fn() at every call (vLLM: the CUDA stream is
capturing), and `with witness.capture():` marks a whole recapture.

The counter is safe inside code a tracer compiles: `trace_safe()` (called by the swap when it
binds the hook) marks the hook for torch.compile with `torch._dynamo.assume_constant_result`,
so Dynamo calls it once while tracing instead of tracing its dict update into the graph (a
traced counter failed a recompile: "modifying buffers ... cudagraph inside the compiler"). The
hook is a function of the key alone: Dynamo needs every argument of a constant-result call to
be a constant, and a bound method's `self` is not. A
call made while tracing counts as a capture call when the engine marks its recompile with
`capture()`; compiled code then never calls it.

`check(keys, old_codes)` turns live counts into findings, so a swap that never ran is never a
pass:
- `running_frame`: zero live calls and a frame of the old code is still running (a loop
  entered before the swap keeps executing the old bytecode);
- `zero_witness`: zero live calls otherwise. With calls during capture it says how many and
  routes to R1 (the code runs only inside graphs), noting that only a device-side probe (a
  counter the captured code increments) or an output change proves a replay runs it.
"""
import ast
import contextlib
import gc
import inspect
import sys
import types

from . import findings as F
from .locate import all_objects, sliced

HIT = "__fastrsi_hit__"


DEVICE_PROBE = ("calls during capture prove the graph holds the new code, not that a replay "
                "ran it: a device-side probe (a counter the captured code increments) or an "
                "output change is needed to prove the replay")


class Witness:
    def __init__(self, capturing=None):
        self.counts = {}            # live calls: the code ran in a step
        self.capture_counts = {}    # calls while a graph was recorded (see the module doc)
        self.capturing = capturing  # fn() -> bool, asked at every call; None: never
        self._capture = 0           # > 0 inside `capture()`
        self.hook = _hook(self)     # what the swap binds as __fastrsi_hit__
        trace_safe(self.hook)

    def _capturing(self):
        if self._capture:
            return True
        if self.capturing is None:
            return False
        try:
            return bool(self.capturing())
        except Exception:           # a probe that fails counts the call as live
            return False

    def hit(self, key):
        c = self.capture_counts if self._capturing() else self.counts
        c[key] = c.get(key, 0) + 1

    @contextlib.contextmanager
    def capture(self):
        """Every call inside this block counts as a capture call (a recapture: its warm-up
        runs execute the model eagerly, and are not steps either)."""
        self._capture += 1
        try:
            yield self
        finally:
            self._capture -= 1

    def arm(self, key, journal=None):
        """A counter for something a repair cell changed (a kernel it rebuilt, a cache it
        re-made): sets counts[key] to 0 (through `journal` when given) and returns a hook
        that counts one call of `key` per call, whatever its arguments (usable as a Triton
        pre_run_hook)."""
        if journal is not None:
            journal.set_item(self.counts, key, 0)
        else:
            self.counts[key] = 0
        self.capture_counts[key] = 0

        def hook(*_a, **_k):
            self.hit(key)
        return hook

    def snapshot(self):
        return dict(self.counts)

    def check(self, keys, old_codes=None):
        """keys: witness keys to prove. old_codes: {key: [code objects the swap replaced]}.
        Returns ok-dict with `counts` and one finding per key that did not run."""
        old = [c for cs in (old_codes or {}).values() for c in cs]
        live = live_codes(suspended=suspendable(old), codes=old)
        out = []
        for k in keys:
            n, cap = self.counts.get(k, 0), self.capture_counts.get(k, 0)
            if n:
                continue
            running = [c for c in (old_codes or {}).get(k, ()) if c in live]
            if running:
                out.append(F.finding("running_frame", "0 calls; a frame of the old code is "
                                     "still running", "calls of the new code", subject=k))
            elif cap:
                f = F.finding("zero_witness", f"0 live calls since the swap; {cap} calls while "
                              "graphs were captured (a replay runs no Python)",
                              "at least one live call, or a device-side probe that counts "
                              "replays", subject=k)
                f["actions"] = [{"action": "prove the replay with a device-side probe or an "
                                 "output change", "cost": "seconds"}] + f["actions"]
                out.append({**f, "route": "R1", "note": DEVICE_PROBE})
            else:
                out.append(F.finding("zero_witness", "0 calls since the swap",
                                     "at least one call", subject=k))
        res = F.ok(counts={k: self.counts.get(k, 0) for k in keys},
                   capture_counts={k: self.capture_counts.get(k, 0) for k in keys},
                   ran={k: ran(self.capture_counts.get(k, 0), self.counts.get(k, 0))
                        for k in keys})
        res["findings"] = out
        return res


def stub(key):
    """What `__fastrsi_hit__` is bound to after undo: counts nothing, raises nothing."""
    return None


def _hook(witness):
    def __fastrsi_hit__(key):
        witness.hit(key)
    return __fastrsi_hit__


def trace_safe(fn):
    """Mark a witness hook for Dynamo (see the module doc) once torch is loaded; idempotent.
    The kernel imports no torch: without it there is nothing to mark. Returns whether fn is
    marked."""
    torch = sys.modules.get("torch")
    if torch is None or getattr(fn, "_dynamo_marked_constant", False):
        return getattr(fn, "_dynamo_marked_constant", False)
    try:
        torch._dynamo.assume_constant_result(fn)
    except Exception:
        return False
    return getattr(fn, "_dynamo_marked_constant", True)


def ran(capture, live):
    """The proof's one line per key."""
    return f"ran during capture {capture}, ran live {live}"


SUSPENDABLE = (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
               | inspect.CO_ITERABLE_COROUTINE)


def suspendable(codes):
    """Can a frame of any of these codes be suspended (generator, coroutine)? Only then can
    it be running without being on a thread's stack."""
    return any(c.co_flags & SUSPENDABLE for c in codes)


def stack_codes():
    """Code objects with a frame on some thread's stack right now (cheap: no heap scan)."""
    codes = set()
    for fr in sys._current_frames().values():
        while fr is not None:
            codes.add(fr.f_code)
            fr = fr.f_back
    return codes


def _frame(o):
    t = type(o)
    return (o.cr_frame if t is types.CoroutineType else o.gi_frame
            if t is types.GeneratorType else o.ag_frame
            if t is types.AsyncGeneratorType else None)


def live_codes(suspended=True, codes=None):
    """Code objects with a running frame: on any thread's stack and, with suspended=True, in
    a suspended generator / coroutine (an event loop waiting in `await` is on no thread's
    stack). Thread stacks are cheap. codes: the codes the caller cares about; then suspended
    frames are found from those codes' referrers (a suspended generator's frame refers to its
    code), only for the codes that can be suspended, with one C-level `gc.get_referrers` call
    over the objects made since the engine's `gc.freeze()` (vLLM freezes after start-up). A
    generator made before the freeze and still suspended is not seen: the price of not
    traversing the whole heap (with the frozen heap included, the call alone took 0.2-0.3 s
    per stage on a loaded Qwen3-8B worker). Without codes, every suspended frame is found
    with a heap scan (one get_objects, then a sliced pass): 0.42 s of a 0.49 s stage per rank
    for a generator target in load-time code (DeepSeek-V4.1 TP4)."""
    out = set()
    for fr in sys._current_frames().values():
        while fr is not None:
            out.add(fr.f_code)
            fr = fr.f_back
    if not suspended:
        return out
    if codes is not None:
        want = [c for c in codes if c.co_flags & SUSPENDABLE]
        if not want:
            return out
        refs = gc.get_referrers(*want)     # objects made since gc.freeze() (see above)
        try:
            for o in refs:
                fr = _frame(o)
                while fr is not None:
                    out.add(fr.f_code)
                    fr = fr.f_back
        finally:
            del refs
        return out
    codes = out
    objs = all_objects()
    try:
        for o in sliced(objs):
            fr = _frame(o)
            while fr is not None:
                codes.add(fr.f_code)
                fr = fr.f_back
    finally:
        del objs
    return codes


def inject(node, key):
    """Insert the counter call at the top of a def, lambda or generator expression (in place)."""
    call = ast.Call(ast.Name(HIT, ast.Load()), [ast.Constant(key)], [])
    if isinstance(node, ast.Lambda):
        node.body = ast.copy_location(
            ast.BoolOp(ast.Or(), [ast.copy_location(call, node.body), node.body]), node.body)
        return
    if isinstance(node, ast.GeneratorExp):    # counts once per item produced, not per call
        node.elt = ast.copy_location(
            ast.BoolOp(ast.Or(), [ast.copy_location(call, node.elt), node.elt]), node.elt)
        return
    body = node.body
    doc = isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        and isinstance(body[0].value.value, str)
    at = body[1] if doc and len(body) > 1 else body[0]
    stmt = ast.copy_location(ast.Expr(ast.copy_location(call, at)), at)
    body.insert(1 if doc else 0, stmt)
