"""Witness: count calls of swapped code, keyed `module:qualname:firstline`.

The swap injects `__rsibox_hit__("<key>")` as the first statement of each new body (after
the docstring). `__rsibox_hit__` is a global of the
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
import functools
import gc
import inspect
import sys
import time
import types

from . import findings as F
from .locate import _FROZEN_GENS, all_objects, sliced

HIT = "__rsibox_hit__"


DEVICE_PROBE = ("calls during capture prove the graph holds the new code, not that a replay "
                "ran it: a device-side probe (a counter the captured code increments) or an "
                "output change is needed to prove the replay")


class Witness:
    def __init__(self, capturing=None):
        self.counts = {}            # live calls: the code ran in a step
        self.capture_counts = {}    # calls while a graph was recorded (see the module doc)
        self.capturing = capturing  # fn() -> bool, asked at every call; None: never
        self._capture = 0           # > 0 inside `capture()`
        self.probe_errors = {}      # calls whose `capturing` probe raised: never live
        self.first_hit = {}         # key -> time.monotonic() of its first live call
        self.first_hit_wall = {}    # the same on the wall clock (other hosts)
        self.hook = _hook(self)     # what the swap binds as __rsibox_hit__
        trace_safe(self.hook)

    def _capturing(self):
        """True, False, or None when the probe raised (the call is then not proven live)."""
        if self._capture:
            return True
        if self.capturing is None:
            return False
        try:
            return bool(self.capturing())
        except Exception:           # fail closed: a probe that fails proves nothing
            return None

    def hit(self, key):
        cap = self._capturing()
        if cap is None:
            self.probe_errors[key] = self.probe_errors.get(key, 0) + 1
            return
        c = self.capture_counts if cap else self.counts
        c[key] = c.get(key, 0) + 1
        if not cap and key not in self.first_hit:
            self.first_hit[key] = time.monotonic()
            self.first_hit_wall[key] = time.time()

    def reset(self, key):
        """Forget a key's capture count, probe errors and first-hit time (apply: a new swap's
        counters start at 0; the live count goes through the journal)."""
        self.capture_counts[key] = 0
        for d in (self.probe_errors, self.first_hit, self.first_hit_wall):
            d.pop(key, None)

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
        self.reset(key)

        def hook(*_a, **_k):
            self.hit(key)
        return hook

    def arm_launch(self, key, label, journal=None):
        """A Triton launch hook for one kernel that runs the code of `key` (the kernel itself
        or a caller that inlines it): every launch counts `key` and the kernel's own key
        `launch_key(key, label)`, so the proof says which kernel launched. `key` itself is
        armed by `arm` (once per edit)."""
        own = launch_key(key, label)
        if journal is not None:
            journal.set_item(self.counts, own, 0)
        else:
            self.counts[own] = 0
        self.reset(own)

        def hook(*_a, **_k):
            self.hit(key)
            self.hit(own)
        return hook

    def launches(self, key):
        """{kernel label: live launches} of the kernels armed for `key` (arm_launch)."""
        pre = key + LAUNCH_SEP
        return {k[len(pre):]: n for k, n in self.counts.items() if k.startswith(pre)}

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
            probe = self.probe_errors.get(k, 0)
            if probe:
                out.append(F.finding("zero_witness", f"0 proven live calls; {probe} call(s) "
                                     "whose capture probe raised (not counted: fail closed)",
                                     "at least one live call with a working probe", subject=k))
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
                        for k in keys},
                   probe_errors={k: self.probe_errors[k] for k in keys if k in self.probe_errors},
                   first_hit={k: self.first_hit[k] for k in keys if k in self.first_hit},
                   first_hit_wall={k: self.first_hit_wall[k] for k in keys
                                   if k in self.first_hit_wall})
        res["findings"] = out
        return res


LAUNCH_SEP = "@launch:"


def launch_key(key, label):
    """The per-kernel counter of a Triton launch that runs `key`'s code."""
    return f"{key}{LAUNCH_SEP}{label}"


def stub(key):
    """What `__rsibox_hit__` is bound to after undo: counts nothing, raises nothing."""
    return None


def _hook(witness):
    def __rsibox_hit__(key):
        witness.hit(key)
    return __rsibox_hit__


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
    traversing the whole frozen heap of a loaded worker. Without codes, every suspended frame
    is found with a heap scan (one get_objects, then a sliced pass), which can be most of a
    stage's time."""
    out = set()
    for fr in sys._current_frames().values():
        while fr is not None:
            out.add(fr.f_code)
            fr = fr.f_back
    if not suspended:
        return out
    if codes is not None:
        want = {c for c in codes if c.co_flags & SUSPENDABLE}
        if not want:
            return out
        refs = gc.get_referrers(*want)     # objects made since gc.freeze() (see above)
        try:
            for obj in (*refs, *list(_FROZEN_GENS)):
                fr = _frame(obj)
                while fr is not None:
                    if fr.f_code in want:
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


_RESTART_INSTRUMENTED_TARGETS = {}
_RESTART_BY_NAME = {}       # bare target name -> [wrapper per module]
_RESTART_RECORDS = {}       # (id(module), target_name) -> provenance dict
_RESTART_MODULE_META = {}   # id(module) -> {had_hit, prev_hit, module_hit, had_counts}
_MISSING = object()


def instrument_module_functions(module, function_names):
    """Wrap named functions or `class.method` targets on `module` with a lightweight
    `__rsibox_hit__` call counter (preserving `functools.wraps` metadata) for
    code-restart runs (`RSIBOX_OVERLAY` or zygote `--refresh`).

    Returns a dictionary mapping each target name to its instrumented wrapper.
    """
    mod_id = id(module)
    mod_dict = getattr(module, "__dict__", None)
    had_counts = (isinstance(mod_dict, dict) and "__rsibox_restart_counts__" in mod_dict) or (
        mod_dict is None and hasattr(module, "__rsibox_restart_counts__")
    )
    restart_counts = getattr(module, "__rsibox_restart_counts__", None)
    if not isinstance(restart_counts, dict):
        restart_counts = {}
        setattr(module, "__rsibox_restart_counts__", restart_counts)

    if mod_id not in _RESTART_MODULE_META:
        prev_hit = mod_dict.get(HIT, _MISSING) if isinstance(mod_dict, dict) else getattr(module, HIT, _MISSING)
        _RESTART_MODULE_META[mod_id] = {
            "had_hit": prev_hit is not _MISSING,
            "prev_hit": prev_hit,
            "had_counts": had_counts,
            "module_hit": None,
        }

    def module_hit(target_key):
        restart_counts[target_key] = restart_counts.get(target_key, 0) + 1

    trace_safe(module_hit)
    setattr(module, HIT, module_hit)
    _RESTART_MODULE_META[mod_id]["module_hit"] = module_hit

    if isinstance(function_names, str):
        target_list = [function_names]
    else:
        target_list = list(function_names)

    instrumented = {}
    module_name = getattr(module, "__name__", "<module>")
    for target_name in target_list:
        parts = target_name.split(".")
        parent_obj = module
        for part in parts[:-1]:
            parent_obj = getattr(parent_obj, part)
        leaf_name = parts[-1]
        try:
            raw_attr = inspect.getattr_static(parent_obj, leaf_name)
        except AttributeError:
            raw_attr = getattr(parent_obj, leaf_name)

        descriptor_kind = None
        if isinstance(raw_attr, staticmethod):
            descriptor_kind = "staticmethod"
            orig_callable = raw_attr.__func__
        elif isinstance(raw_attr, classmethod):
            descriptor_kind = "classmethod"
            orig_callable = raw_attr.__func__
        elif isinstance(raw_attr, property) and raw_attr.fget is not None:
            descriptor_kind = "property"      # count the getter, keep the descriptor
            orig_callable = raw_attr.fget
        else:
            orig_callable = getattr(parent_obj, leaf_name)
            raw_attr = orig_callable

        existing = _RESTART_RECORDS.get((mod_id, target_name))
        if getattr(orig_callable, "__rsibox_target__", None) == target_name:
            orig_callable = getattr(orig_callable, "__wrapped__", orig_callable)
        saved_raw_attr = existing["orig_raw_attr"] if existing is not None else raw_attr
        saved_orig_callable = existing["orig_callable"] if existing is not None else orig_callable

        restart_counts[target_name] = 0

        def make_wrapper(func_obj, registered_name):
            @functools.wraps(func_obj)
            def wrapped_target(*args, **kwargs):
                module_hit(registered_name)
                wrapped_target.__rsibox_calls__ = restart_counts[registered_name]
                return func_obj(*args, **kwargs)

            wrapped_target.__rsibox_target__ = registered_name
            wrapped_target.__rsibox_module__ = module_name
            wrapped_target.__rsibox_calls__ = 0
            wrapped_target.__rsibox_counts__ = restart_counts
            wrapped_target.__rsibox_hit__ = module_hit
            return wrapped_target

        wrapper = make_wrapper(saved_orig_callable, target_name)
        if descriptor_kind == "staticmethod":
            setattr(parent_obj, leaf_name, staticmethod(wrapper))
        elif descriptor_kind == "classmethod":
            setattr(parent_obj, leaf_name, classmethod(wrapper))
        elif descriptor_kind == "property":
            prev_prop = saved_raw_attr if isinstance(saved_raw_attr, property) else raw_attr
            setattr(parent_obj, leaf_name, prev_prop.getter(wrapper))
        else:
            setattr(parent_obj, leaf_name, wrapper)
            if parent_obj is module:
                prev_aliases = existing["wrapper"] if existing is not None else saved_orig_callable
                for loaded_mod in list(sys.modules.values()):
                    loaded_dict = getattr(loaded_mod, "__dict__", None)
                    if isinstance(loaded_dict, dict):
                        for attr_key, attr_val in list(loaded_dict.items()):
                            if attr_val is prev_aliases or attr_val is saved_orig_callable:
                                loaded_dict[attr_key] = wrapper

        instrumented[target_name] = wrapper
        _RESTART_RECORDS[(mod_id, target_name)] = {
            "module": module,
            "module_name": module_name,
            "parent_obj": parent_obj,
            "leaf_name": leaf_name,
            "descriptor_kind": descriptor_kind,
            "orig_raw_attr": saved_raw_attr,
            "orig_callable": saved_orig_callable,
            "wrapper": wrapper,
        }
        # the bare name may be instrumented in several modules (one per rank): keep them all,
        # one per module; collect reads the smallest count (a module that never ran it: 0)
        same_name = [other for other in _RESTART_BY_NAME.get(target_name, ())
                     if getattr(other, "__rsibox_module__", None) != module_name]
        _RESTART_BY_NAME[target_name] = same_name + [wrapper]
        _RESTART_INSTRUMENTED_TARGETS[target_name] = wrapper
        _RESTART_INSTRUMENTED_TARGETS[f"{module_name}:{target_name}"] = wrapper

    return instrumented


def uninstrument_module_functions(module, function_names=None):
    """Remove hooks installed by :func:`instrument_module_functions` from ``module``.

    When ``function_names`` is ``None``, every instrumented target on ``module`` is restored;
    otherwise only those named targets are restored. Restores original functions/descriptors,
    rebinds alias globals in ``sys.modules`` back to the original callable, cleans up global
    restart registries, and removes ``__rsibox_hit__`` / ``__rsibox_restart_counts__`` once no
    instrumented targets remain on ``module``. Returns ``{target_name: restored_obj}``.
    """
    mod_id = id(module)
    if function_names is None:
        target_list = [name for (mid, name) in list(_RESTART_RECORDS) if mid == mod_id]
    elif isinstance(function_names, str):
        target_list = [function_names]
    else:
        target_list = list(function_names)

    restored = {}
    restart_counts = getattr(module, "__rsibox_restart_counts__", None)
    for target_name in target_list:
        rec = _RESTART_RECORDS.pop((mod_id, target_name), None)
        if rec is None:
            continue
        parent_obj = rec["parent_obj"]
        leaf_name = rec["leaf_name"]
        wrapper = rec["wrapper"]
        orig_raw_attr = rec["orig_raw_attr"]
        orig_callable = rec["orig_callable"]
        descriptor_kind = rec["descriptor_kind"]
        module_name = rec["module_name"]

        try:
            current = inspect.getattr_static(parent_obj, leaf_name)
        except AttributeError:
            current = getattr(parent_obj, leaf_name, _MISSING)

        should_restore = (
            current is wrapper
            or (descriptor_kind in ("staticmethod", "classmethod")
                and getattr(current, "__func__", None) is wrapper)
            or (descriptor_kind == "property" and getattr(current, "fget", None) is wrapper)
        )
        if should_restore:
            setattr(parent_obj, leaf_name, orig_raw_attr)

        if descriptor_kind is None and parent_obj is module:
            for loaded_mod in list(sys.modules.values()):
                loaded_dict = getattr(loaded_mod, "__dict__", None)
                if isinstance(loaded_dict, dict):
                    for attr_key, attr_val in list(loaded_dict.items()):
                        if attr_val is wrapper:
                            loaded_dict[attr_key] = orig_callable

        if isinstance(restart_counts, dict):
            restart_counts.pop(target_name, None)

        remaining_by_name = [other for other in _RESTART_BY_NAME.get(target_name, ())
                             if other is not wrapper]
        if remaining_by_name:
            _RESTART_BY_NAME[target_name] = remaining_by_name
        else:
            _RESTART_BY_NAME.pop(target_name, None)

        if _RESTART_INSTRUMENTED_TARGETS.get(f"{module_name}:{target_name}") is wrapper:
            _RESTART_INSTRUMENTED_TARGETS.pop(f"{module_name}:{target_name}", None)
        if _RESTART_INSTRUMENTED_TARGETS.get(target_name) is wrapper:
            if remaining_by_name:
                _RESTART_INSTRUMENTED_TARGETS[target_name] = remaining_by_name[-1]
            else:
                _RESTART_INSTRUMENTED_TARGETS.pop(target_name, None)

        restored[target_name] = orig_raw_attr

    still_instrumented = any(mid == mod_id for (mid, _) in _RESTART_RECORDS)
    if not still_instrumented:
        meta = _RESTART_MODULE_META.pop(mod_id, None)
        if meta is not None:
            if not meta["had_counts"] and isinstance(restart_counts, dict) and not restart_counts:
                if hasattr(module, "__rsibox_restart_counts__"):
                    delattr(module, "__rsibox_restart_counts__")
            current_hit = getattr(module, HIT, _MISSING)
            if current_hit is meta["module_hit"]:
                if meta["had_hit"]:
                    setattr(module, HIT, meta["prev_hit"])
                else:
                    delattr(module, HIT)

    return restored


def collect_restart_witness(targets):
    """Collect per-target execution call counts from instrumented code-restart targets."""
    if isinstance(targets, types.ModuleType):
        return dict(getattr(targets, "__rsibox_restart_counts__", {}))

    if isinstance(targets, dict):
        collected = {}
        for key, val in targets.items():
            if isinstance(val, int):
                collected[key] = val
            elif isinstance(val, dict):
                for inner_key, inner_val in collect_restart_witness(val).items():
                    collected[inner_key] = collected.get(inner_key, 0) + int(inner_val)
            else:
                unwrapped = (
                    val.__func__
                    if isinstance(val, (staticmethod, classmethod))
                    else val
                )
                counts_map = getattr(unwrapped, "__rsibox_counts__", None)
                if isinstance(counts_map, dict) and key in counts_map:
                    collected[key] = int(counts_map[key])
                else:
                    collected[key] = int(getattr(unwrapped, "__rsibox_calls__", 0))
        return collected

    collected = {}
    for item in targets:
        if isinstance(item, types.ModuleType):
            for key, count in getattr(item, "__rsibox_restart_counts__", {}).items():
                collected[key] = collected.get(key, 0) + int(count)
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(
            item[0], types.ModuleType
        ):
            mod_obj, target_name = item
            counts_map = getattr(mod_obj, "__rsibox_restart_counts__", {})
            collected[target_name] = int(counts_map.get(target_name, 0))
        elif isinstance(item, str):
            wrappers = _RESTART_BY_NAME.get(item) or [
                one for one in (_RESTART_INSTRUMENTED_TARGETS.get(item),) if one is not None]
            counts = []
            for wrapper in wrappers:
                target_name = getattr(wrapper, "__rsibox_target__", item)
                counts_map = getattr(wrapper, "__rsibox_counts__", {})
                counts.append(int(
                    counts_map.get(target_name, getattr(wrapper, "__rsibox_calls__", 0))))
            collected[item] = min(counts) if counts else 0
        else:
            unwrapped = (
                item.__func__
                if isinstance(item, (staticmethod, classmethod))
                else item
            )
            target_name = getattr(
                unwrapped,
                "__rsibox_target__",
                getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", repr(unwrapped))),
            )
            counts_map = getattr(unwrapped, "__rsibox_counts__", None)
            if isinstance(counts_map, dict) and target_name in counts_map:
                collected[target_name] = int(counts_map[target_name])
            else:
                collected[target_name] = int(getattr(unwrapped, "__rsibox_calls__", 0))
    return collected


def summarize_restart_witness(counts_by_target):
    """Whether every edited target executed on every rank. Several ranks (a list, or a dict
    of per-rank dicts): `witness` is the per-rank minimum (a target proven only where it ran
    on each rank), `per_rank` the counts as given; `proven: True` only when every target has
    at least one call on every rank, else `finding: "zero_witness"`. One rank: its counts.
    A rank whose counts cannot be read (an exception) fails closed: not proven, the error in
    `errors`."""
    per_rank, errors = {}, {}

    def read(label, entry):
        try:
            per_rank[label] = {k: int(v) for k, v in collect_restart_witness(entry).items()}
        except Exception as e:          # fail closed
            errors[label] = f"{type(e).__name__}: {e}"
    if isinstance(counts_by_target, (list, tuple)):
        for i, entry in enumerate(counts_by_target):
            read(f"rank{i}", entry)
    elif isinstance(counts_by_target, dict) and counts_by_target and all(
            isinstance(v, dict) for v in counts_by_target.values()):
        for label, entry in counts_by_target.items():
            read(str(label), entry)
    else:
        read("rank0", counts_by_target)
    keys = sorted({k for c in per_rank.values() for k in c})
    aggregated = {k: min(c.get(k, 0) for c in per_rank.values()) for k in keys}
    proven = bool(aggregated) and not errors and all(n > 0 for n in aggregated.values())
    out = {"witness": aggregated, "proven": proven,
           "finding": None if proven else "zero_witness"}
    if len(per_rank) > 1 or errors:
        out["per_rank"] = per_rank
    if errors:
        out["errors"] = errors
    return out
