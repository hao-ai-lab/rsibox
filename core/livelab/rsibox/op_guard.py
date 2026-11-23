"""Torch custom-op re-definition guard for live module refresh and reload paths.

Imports ``torch`` lazily inside :func:`op_redefine_guard` so importing ``rsibox`` or
``rsibox.op_guard`` never pulls GPU runtimes into control-plane processes.
"""
import contextlib
import inspect
import re


def _norm_schema(schema):
    """Arguments and returns of a schema, without the op name and with opaque class types
    (dotted names, which the C++ schema shows as PyObject) unified."""
    body = schema.split("(", 1)[-1]
    return re.sub(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+", "PyObject", body).replace(" ", "")


@contextlib.contextmanager
def op_redefine_guard(report=None):
    """While re-importing refreshed modules: ``Library.define`` of an op that already exists
    with the same schema is skipped (its implementation is still replaced by ``Library.impl``,
    which overrides, and its fake impl by ``Library._register_fake`` with allow_override); a
    changed schema raises with the op's name, since a running process cannot change an op's
    signature. ``report(event)`` gets one dict per skipped define, replaced implementation,
    replaced fake impl and re-registered opaque type.

    No-ops transparently when ``torch`` / ``torch.library`` is not installed.
    """
    try:
        from torch.library import Library
        import torch
    except ImportError:
        yield
        return

    orig = Library.define
    newly_defined = set()

    def _emit(event):
        if report is not None:
            report(event)

    def _strip_ns(ns, text):
        stripped = text.lstrip()
        prefix = f"{ns}::"
        if stripped.startswith(prefix):
            return stripped[len(prefix):]
        return stripped

    def _parse_op(ns, raw_name):
        name = _strip_ns(ns, raw_name).strip()
        op, _, overload = name.partition(".")
        return op, overload

    def _schema_exists(ns, op, overload):
        try:
            torch._C._get_schema(f"{ns}::{op}", overload)
            return True
        except RuntimeError:
            return False

    def define(self, schema, *args, **kw):
        unqual_schema = _strip_ns(self.ns, schema)
        name = unqual_schema.split("(", 1)[0].strip()
        op, _, overload = name.partition(".")
        qual = f"{self.ns}::{op}"
        try:
            old = str(torch._C._get_schema(qual, overload))
        except RuntimeError:
            res = orig(self, schema, *args, **kw)
            newly_defined.add((self.ns, op, overload))
            return res
        try:   # torch's own printing, so defaults compare alike ("0.0" prints as "0.")
            new = str(torch._C.parse_schema(f"{self.ns}::{unqual_schema}"))
        except RuntimeError:
            new = unqual_schema
        if _norm_schema(old) != _norm_schema(new):
            raise RuntimeError(
                f"refresh: op {qual} changed its schema ({old} -> {schema}); "
                "a process cannot change an op's signature: restart the zygote"
            )
        _emit({"event": "zygote_op_redefined",
               "op": qual + (f".{overload}" if overload else ""),
               "schema": old})
        return name

    orig_impl = Library.impl
    orig_fake = getattr(Library, "_register_fake", None)
    orig_opaque = getattr(torch._C, "_register_opaque_type", None)
    can_override = "allow_override" in inspect.signature(orig_impl).parameters
    can_override_fake = (
        orig_fake is not None and "allow_override" in inspect.signature(orig_fake).parameters
    )

    def _should_override(ns, op_name):
        if not isinstance(op_name, str):
            return False, None
        op, overload = _parse_op(ns, op_name)
        qual_op = f"{ns}::{op}" + (f".{overload}" if overload else "")
        if (ns, op, overload) in newly_defined:
            return False, qual_op
        return _schema_exists(ns, op, overload), qual_op

    def impl(self, op_name, *args, **kw):
        override, qual_op = _should_override(self.ns, op_name)
        if override:
            if can_override:
                kw["allow_override"] = True
            _emit({"event": "zygote_op_impl_replaced", "op": qual_op,
                   "dispatch_key": kw.get("dispatch_key", args[1] if len(args) > 1 else "")})
        return orig_impl(self, op_name, *args, **kw)

    def register_fake(self, op_name, fn, _stacklevel=1, **kw):
        override, qual_op = _should_override(self.ns, op_name)
        if override:
            if can_override_fake:
                kw["allow_override"] = True
            else:
                simple_reg = getattr(getattr(torch, "_library", None), "simple_registry", None)
                if simple_reg is not None:
                    entry = simple_reg.singleton.find(qual_op)
                    if entry.fake_impl.kernel is not None:
                        entry.fake_impl.kernel = None
            _emit({"event": "zygote_op_fake_replaced", "op": qual_op})
        return orig_fake(self, op_name, fn, _stacklevel + 1, **kw)

    def register_opaque(name, *args, **kw):
        try:
            return orig_opaque(name, *args, **kw)
        except RuntimeError as e:
            if "already registered" not in str(e):
                raise
            _emit({"event": "zygote_opaque_reregistered", "type": name})

    patches = [(Library, "define", orig, define), (Library, "impl", orig_impl, impl)]
    if orig_fake is not None:
        patches.append((Library, "_register_fake", orig_fake, register_fake))
    if orig_opaque is not None:
        patches.append((torch._C, "_register_opaque_type", orig_opaque, register_opaque))
    for obj, attr, _old, new in patches:
        setattr(obj, attr, new)
    try:
        yield
    finally:
        for obj, attr, old, _new in patches:
            setattr(obj, attr, old)


__all__ = ["op_redefine_guard"]
