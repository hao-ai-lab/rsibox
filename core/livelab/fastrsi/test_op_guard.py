"""Tests for shared `fastrsi.op_guard` (M-H16): lazy torch import and op-redefine protection."""
import os
import subprocess
import sys
import types
import unittest

import fastrsi
from fastrsi.op_guard import _norm_schema, op_redefine_guard


def _make_fake_torch(schemas, *, fake_allow_override=True):
    """Build a minimal fake `torch` + `torch.library.Library` module tree for CPU tests."""
    calls = {"define": [], "impl": [], "register_fake": [], "opaque": []}

    class FakeLibrary:
        def __init__(self, ns):
            self.ns = ns

        def define(self, schema, *args, **kw):
            calls["define"].append((self.ns, schema, args, kw))
            name = schema.split("(", 1)[0].strip()
            op, _, overload = name.partition(".")
            schemas[(f"{self.ns}::{op}", overload)] = f"{self.ns}::{schema}"
            return name

        def impl(self, op_name, fn, dispatch_key="", *, allow_override=False):
            calls["impl"].append((self.ns, op_name, fn, dispatch_key, allow_override))
            return fn

        if fake_allow_override:
            def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=False):
                calls["register_fake"].append((self.ns, op_name, fn, _stacklevel, allow_override))
                return fn
        else:
            def _register_fake(self, op_name, fn, _stacklevel=1):
                calls["register_fake"].append((self.ns, op_name, fn, _stacklevel, None))
                return fn

    fake_entries = {}

    class FakeEntry:
        def __init__(self):
            self.fake_impl = types.SimpleNamespace(kernel=lambda: "old")

    def find_entry(qual):
        return fake_entries.setdefault(qual, FakeEntry())

    def get_schema(qual, overload):
        key = (qual, overload)
        if key not in schemas:
            raise RuntimeError(f"schema {qual}.{overload} not found")
        return schemas[key]

    def parse_schema(full_schema):
        if "INVALID" in full_schema:
            raise RuntimeError("bad schema")
        return full_schema

    opaque_error = {"mode": "already"}

    def register_opaque_type(name, *args, **kw):
        calls["opaque"].append((name, args, kw))
        if opaque_error["mode"] == "already":
            raise RuntimeError(f"opaque type {name} is already registered")
        if opaque_error["mode"] == "other":
            raise RuntimeError("unexpected opaque failure")
        return name

    fake_torch = types.ModuleType("torch")
    fake_torch._C = types.SimpleNamespace(
        _get_schema=get_schema,
        parse_schema=parse_schema,
        _register_opaque_type=register_opaque_type,
    )
    fake_torch._library = types.SimpleNamespace(
        simple_registry=types.SimpleNamespace(singleton=types.SimpleNamespace(find=find_entry))
    )
    fake_torch_lib = types.ModuleType("torch.library")
    fake_torch_lib.Library = FakeLibrary
    fake_torch.library = fake_torch_lib
    return fake_torch, fake_torch_lib, FakeLibrary, fake_entries, opaque_error, calls


class OpGuardTest(unittest.TestCase):
    def test_reexport_and_zero_eager_torch_import(self):
        self.assertIs(fastrsi.op_redefine_guard, op_redefine_guard)
        self.assertIs(fastrsi.op_guard.op_redefine_guard, op_redefine_guard)
        code = (
            "import sys, fastrsi, fastrsi.op_guard; "
            "bad = [m for m in ('torch', 'triton', 'cuda') if m in sys.modules]; "
            "assert not bad, bad"
        )
        env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_import_error_when_torch_missing_noops_cleanly(self):
        missing = object()
        saved = {k: sys.modules.get(k, missing) for k in ("torch", "torch.library")}
        sys.modules["torch"] = None
        sys.modules["torch.library"] = None
        try:
            events = []
            with op_redefine_guard(events.append):
                events.append("ran")
            self.assertEqual(events, ["ran"])
        finally:
            for k, orig in saved.items():
                if orig is missing:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = orig

    def test_norm_schema_unifies_opaque_types_and_spaces(self):
        self.assertEqual(
            _norm_schema("ns::op(my.mod.Token tok, Tensor x) -> Tensor"),
            _norm_schema("op(PyObject tok,Tensor x)->Tensor"),
        )

    def test_guard_skips_same_schema_replaces_impls_and_restores_patches(self):
        schemas = {("myns::myop", "over"): "myns::myop.over(Tensor x) -> Tensor"}
        fake_torch, fake_lib, Library, _, _, calls = _make_fake_torch(schemas, fake_allow_override=True)
        orig_define, orig_impl = Library.define, Library.impl
        orig_fake, orig_opaque = Library._register_fake, fake_torch._C._register_opaque_type

        saved = {k: sys.modules.get(k) for k in ("torch", "torch.library")}
        sys.modules["torch"] = fake_torch
        sys.modules["torch.library"] = fake_lib
        events = []
        try:
            with op_redefine_guard(events.append):
                lib = Library("myns")
                # Same op/schema -> skipped + impl/fake replaced with allow_override
                self.assertEqual(lib.define("myop.over(Tensor x) -> Tensor"), "myop.over")
                lib.impl("myop.over", lambda x: x, "CUDA")
                lib._register_fake("myop.over", lambda x: x)
                # Brand new op -> delegates to orig define; impl doesn't force allow_override
                self.assertEqual(lib.define("brand_new(Tensor x) -> Tensor"), "brand_new")
                lib.impl("brand_new", lambda x: x, "CUDA")
                # Already-registered opaque type -> event without raising
                fake_torch._C._register_opaque_type("OpaqueToken")

            # Patched hooks restored cleanly after exit
            self.assertIs(Library.define, orig_define)
            self.assertIs(Library.impl, orig_impl)
            self.assertIs(Library._register_fake, orig_fake)
            self.assertIs(fake_torch._C._register_opaque_type, orig_opaque)

            self.assertEqual([e["event"] for e in events], [
                "zygote_op_redefined",
                "zygote_op_impl_replaced",
                "zygote_op_fake_replaced",
                "zygote_opaque_reregistered",
            ])
            self.assertEqual(events[0]["op"], "myns::myop.over")
            self.assertEqual(calls["define"][0][1], "brand_new(Tensor x) -> Tensor")
            self.assertTrue(calls["impl"][0][4])   # skipped op got allow_override=True
            self.assertFalse(calls["impl"][1][4])  # brand_new did not
            self.assertTrue(calls["register_fake"][0][4])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_changed_schema_raises_naming_op_and_restores_patches(self):
        schemas = {("myns::myop", ""): "myns::myop(Tensor x) -> Tensor"}
        fake_torch, fake_lib, Library, _, _, _ = _make_fake_torch(schemas)
        orig_define = Library.define
        saved = {k: sys.modules.get(k) for k in ("torch", "torch.library")}
        sys.modules["torch"] = fake_torch
        sys.modules["torch.library"] = fake_lib
        try:
            with self.assertRaises(RuntimeError) as ctx:
                with op_redefine_guard():
                    Library("myns").define("myop(Tensor x, int k) -> Tensor")
            self.assertIn("myns::myop", str(ctx.exception))
            self.assertIs(Library.define, orig_define)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_legacy_fake_fallback_and_non_duplicate_opaque_error(self):
        schemas = {("myns::myop", ""): "myns::myop(Tensor x) -> Tensor"}
        fake_torch, fake_lib, Library, fake_entries, opaque_error, _ = _make_fake_torch(
            schemas, fake_allow_override=False
        )
        saved = {k: sys.modules.get(k) for k in ("torch", "torch.library")}
        sys.modules["torch"] = fake_torch
        sys.modules["torch.library"] = fake_lib
        try:
            opaque_error["mode"] = "other"
            events = []
            with op_redefine_guard(events.append):
                lib = Library("myns")
                lib.define("myop(Tensor x) -> Tensor")
                lib._register_fake("myop", lambda x: x)
                self.assertIsNone(fake_entries["myns::myop"].fake_impl.kernel)
                with self.assertRaises(RuntimeError):
                    fake_torch._C._register_opaque_type("OpaqueToken")
            self.assertEqual([e["event"] for e in events], [
                "zygote_op_redefined",
                "zygote_op_fake_replaced",
            ])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
