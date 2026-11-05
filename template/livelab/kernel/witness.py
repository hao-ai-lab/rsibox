"""Witness: count calls of swapped code, keyed `module:qualname:firstline`.

The swap injects `__fastrsi_hit__("<key>")` as the first statement of each new body (after
the docstring), as `fastrsi_lab_cov.install_batch` does. `__fastrsi_hit__` is a global of the
module, bound to the participant's `Witness.hit`; the swap adds it through the journal, so undo
removes it and the counter keys with it.

`check(keys, old_codes)` turns counts into findings, so a swap that never ran is never a pass:
- `running_frame`: zero calls and a frame of the old code is still running (a loop entered
  before the swap keeps executing the old bytecode);
- `zero_witness`: zero calls otherwise.
"""
import ast
import inspect
import sys
import types

from . import findings as F
from .locate import all_objects, sliced

HIT = "__fastrsi_hit__"


class Witness:
    def __init__(self):
        self.counts = {}

    def hit(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def snapshot(self):
        return dict(self.counts)

    def check(self, keys, old_codes=None):
        """keys: witness keys to prove. old_codes: {key: [code objects the swap replaced]}.
        Returns ok-dict with `counts` and one finding per key that did not run."""
        old = [c for cs in (old_codes or {}).values() for c in cs]
        live = live_codes(suspended=suspendable(old))
        out = []
        for k in keys:
            n = self.counts.get(k, 0)
            if n:
                continue
            running = [c for c in (old_codes or {}).get(k, ()) if c in live]
            if running:
                out.append(F.finding("running_frame", "0 calls; a frame of the old code is "
                                     "still running", "calls of the new code", subject=k))
            else:
                out.append(F.finding("zero_witness", "0 calls since the swap",
                                     "at least one call", subject=k))
        res = F.ok(counts={k: self.counts.get(k, 0) for k in keys})
        res["findings"] = out
        return res


SUSPENDABLE = (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
               | inspect.CO_ITERABLE_COROUTINE)


def suspendable(codes):
    """Can a frame of any of these codes be suspended (generator, coroutine)? Only then can
    it be running without being on a thread's stack."""
    return any(c.co_flags & SUSPENDABLE for c in codes)


def live_codes(suspended=True):
    """Code objects with a running frame: on any thread's stack and, with suspended=True, in
    a suspended generator / coroutine (an event loop waiting in `await` is on no thread's
    stack). Thread stacks are cheap; suspended frames need a heap scan (one get_objects,
    then a sliced pass), so callers ask for them only when a code can be suspended."""
    codes = set()
    for fr in sys._current_frames().values():
        while fr is not None:
            codes.add(fr.f_code)
            fr = fr.f_back
    if not suspended:
        return codes
    objs = all_objects()
    try:
        for o in sliced(objs):
            t = type(o)
            fr = (o.cr_frame if t is types.CoroutineType else o.gi_frame
                  if t is types.GeneratorType else o.ag_frame
                  if t is types.AsyncGeneratorType else None)
            while fr is not None:
                codes.add(fr.f_code)
                fr = fr.f_back
    finally:
        del objs
    return codes


def inject(node, key):
    """Insert the counter call at the top of a def or lambda node (in place)."""
    call = ast.Call(ast.Name(HIT, ast.Load()), [ast.Constant(key)], [])
    if isinstance(node, ast.Lambda):
        node.body = ast.copy_location(
            ast.BoolOp(ast.Or(), [ast.copy_location(call, node.body), node.body]), node.body)
        return
    body = node.body
    doc = isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        and isinstance(body[0].value.value, str)
    at = body[1] if doc and len(body) > 1 else body[0]
    stmt = ast.copy_location(ast.Expr(ast.copy_location(call, at)), at)
    body.insert(1 if doc else 0, stmt)
