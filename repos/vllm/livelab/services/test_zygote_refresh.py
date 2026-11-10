"""Golden cases for refreshing edited modules in a process forked from the zygote.

Each case imports a small real package in the parent (the zygote), edits a file, forks, and
checks in the child what a server would see. They run on CPU; the vLLM-sized numbers come from
the GPU receipt (repos/vllm/kb/costs.md).
"""
import json, os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zygote, zygote_refresh as zr


def _pkg(tmp, files):
    root = tmp / "zr_pkgroot"
    for name, src in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    sys.path.insert(0, str(root))
    return root


def _in_child(fn):
    """Run fn() in a forked child; return what it returned (JSON) via a pipe."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = fn()
        except BaseException as e:
            out = {"error": repr(e)}
        os.write(w, json.dumps(out).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    os.waitpid(pid, 0)
    return json.loads(data)


def _edit(path, old, new):
    s = path.read_text()
    path.write_text(s.replace(old, new))
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # mtime moves even within 1 s


def test_edit_reaches_importers_and_spares_the_rest(tmp_path):
    """The pain point: `from a import X` copies X into b, so dropping only a leaves b on the old
    value. A refresh drops a and its importers (b, and d through b), and leaves c alone."""
    root = _pkg(tmp_path, {
        "zr1/__init__.py": "",
        "zr1/a.py": "X = 1\n",
        "zr1/b.py": "from zr1.a import X\ndef get():\n    return X\n",
        "zr1/d.py": "from . import b\ndef get():\n    return b.get()\n",
        "zr1/c.py": "Y = 'unrelated'\n",
    })
    import zr1.d, zr1.c
    snap = zygote._sources()
    c_before = id(sys.modules["zr1.c"])
    graph = zr.import_graph(("zr1",))
    _edit(root / "zr1/a.py", "X = 1", "X = 2")

    def child():
        p = zr.plan(zygote._changed(snap), graph)
        zr.refresh(p["modules"])
        import zr1.d as d2, zr1.c as c2
        return {"plan": p, "value": d2.get(), "c_same": id(c2) == c_before}
    out = _in_child(child)
    assert out["plan"]["changed"] == ["zr1.a"]
    assert out["plan"]["modules"] == ["zr1.a", "zr1.b", "zr1.d"]
    assert out["plan"]["blocked"] == []
    assert out["value"] == 2           # without the refresh the child returns 1
    assert out["c_same"] is True       # unrelated modules keep the zygote's copy
    assert zr1.d.get() == 1            # the zygote itself still holds the old code


def test_extension_edit_is_blocked_not_silently_kept(tmp_path):
    """A changed file that is not Python source (here standing in for a rebuilt .so) cannot be
    unloaded; the plan must name it so the zygote reports a finding instead of forking."""
    p = zr.plan(["/x/_C.abi3.so"], {}, modules={"m": type(sys)("m")})
    assert p["modules"] == [] and p["blocked"][0]["file"] == "/x/_C.abi3.so"
    assert "cannot be unloaded" in p["blocked"][0]["why"]


def test_reimported_op_with_same_schema_takes_the_new_body():
    """A refreshed module registers its torch op again. torch refuses a second define; the guard
    lets an identical schema through so the new implementation replaces the old one, and refuses
    a changed schema with the op's name."""
    torch = __import__("pytest").importorskip("torch")
    from torch.library import Library
    lib = Library("zrtest", "FRAGMENT")
    lib.define("scale(Tensor x, int k) -> Tensor")
    lib.impl("scale", lambda x, k: x * k, "CPU")
    seen = []
    with zr.guard_for(["zrtest_module"], seen.append):
        lib.define("scale(Tensor x, int k) -> Tensor")
        lib.impl("scale", lambda x, k: x + k, "CPU")
        assert torch.ops.zrtest.scale(torch.ones(2), 3).tolist() == [4.0, 4.0]
        try:
            lib.define("scale(Tensor x) -> Tensor")
            raise AssertionError("a changed schema must raise")
        except RuntimeError as e:
            assert "zrtest::scale changed its schema" in str(e)
    assert [e["event"] for e in seen] == ["zygote_op_redefined", "zygote_op_impl_replaced"]
    assert seen[0]["op"] == "zrtest::scale"


def test_fork_without_refresh_keeps_torch_duplicate_errors():
    """The guard must not outlive its purpose: a fork that refreshed nothing (and the engine
    core and workers forked from it) must still fail on a duplicate op definition, as torch
    does, instead of silently skipping it."""
    torch = __import__("pytest").importorskip("torch")
    from torch.library import Library
    lib = Library("zrtest2", "FRAGMENT")
    lib.define("twice(Tensor x) -> Tensor")
    seen = []
    with zr.guard_for([], seen.append):
        try:
            lib.define("twice(Tensor x) -> Tensor")
            raise AssertionError("a duplicate define must raise when nothing was refreshed")
        except RuntimeError as e:
            assert "refresh:" not in str(e)        # torch's own error, not the guard's
    assert seen == []


def test_refreshed_layer_registers_its_name_again(tmp_path, monkeypatch):
    """Upstream 61323 edits layernorm.py, whose `@CustomOp.register("rms_norm")` asserts the
    name is new in a registry that lives in a module the refresh keeps. Re-importing the edited
    module then dies with "Duplicate op name: rms_norm" and the forked server never starts.
    Here a small package copies that registration path: the refreshed module registers its name
    again with the new class, and a real duplicate (another module's name) still fails."""
    root = _pkg(tmp_path, {
        "zr2/__init__.py": "",
        "zr2/custom_op.py": """
            op_registry = {}
            def register(name):
                def deco(cls):
                    assert name not in op_registry, f"Duplicate op name: {name}"
                    op_registry[name] = cls
                    return cls
                return deco
        """,
        "zr2/layers.py": """
            from zr2.custom_op import register
            @register("rms_norm")
            class RMSNorm:
                eps = 1
        """,
        "zr2/other.py": """
            from zr2.custom_op import register
            @register("silu")
            class Silu:
                pass
        """,
    })
    import zr2.layers, zr2.other
    monkeypatch.setattr(zr, "NAME_REGISTRIES", {"zr2.custom_op": ("op_registry",)},
                        raising=False)
    graph = zr.import_graph(("zr2",))
    _edit(root / "zr2/layers.py", "eps = 1", "eps = 2")

    def child():
        p = zr.plan([str(root / "zr2/layers.py")], graph)
        zr.refresh(p["modules"])
        seen = []
        with zr.guard_for(p["modules"], seen.append):
            import zr2.layers as layers
            from zr2.custom_op import op_registry, register
            try:                       # a name the refreshed module does not own stays taken
                register("silu")(type("Dup", (), {}))
                dup = "registered"
            except AssertionError as e:
                dup = str(e)
        return {"refreshed": p["modules"], "eps": op_registry["rms_norm"].eps,
                "is_new": op_registry["rms_norm"] is layers.RMSNorm, "dup": dup,
                "events": [(e["event"], e.get("name")) for e in seen]}
    out = _in_child(child)
    assert "error" not in out, out["error"]   # without the fix: "Duplicate op name: rms_norm"
    assert out["refreshed"] == ["zr2.layers"]
    assert out["eps"] == 2 and out["is_new"] is True
    assert out["dup"] == "Duplicate op name: silu"
    assert out["events"] == [["zygote_registry_forgot", "rms_norm"]]


VLLM_PY = os.environ.get("FASTRSI_VLLM_PYTHON")
_VLLM_REFRESH = r'''
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import zygote_refresh as zr
import vllm.model_executor.layers.layernorm as ln
from vllm.model_executor.custom_op import op_registry
from vllm.utils.torch_utils import direct_register_custom_op
from torch.library import Library
out = {"n_layernorm_ops": sum(1 for c in op_registry.values()
                             if getattr(c, "__module__", "") == ln.__name__)}
p = zr.plan([ln.__file__], zr.import_graph(("vllm",)))
zr.refresh(p["modules"])
seen = []
with zr.guard_for(p["modules"], seen.append):
    import vllm.model_executor.layers.layernorm as ln2
out["new_class"] = ln2 is not ln and op_registry["rms_norm"] is ln2.RMSNorm
# vLLM's other registration path: direct_register_custom_op -> Library.define, impl, fake
lib = Library("zrvllm", "FRAGMENT")
def double(x: torch.Tensor) -> torch.Tensor:
    return x * 2
def fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
direct_register_custom_op("probe", double, fake_impl=fake, target_lib=lib, dispatch_key="CPU")
with zr.guard_for(["zr_probe_module"], seen.append):
    def triple(x: torch.Tensor) -> torch.Tensor:
        return x * 3
    direct_register_custom_op("probe", triple, fake_impl=fake, target_lib=lib,
                              dispatch_key="CPU")
    out["same_schema_value"] = torch.ops.zrvllm.probe(torch.ones(1)).tolist()
    def changed(x: torch.Tensor, k: int) -> torch.Tensor:
        return x * k
    try:
        direct_register_custom_op("probe", changed, target_lib=lib, dispatch_key="CPU")
        out["changed_schema"] = "registered"
    except RuntimeError as e:
        out["changed_schema"] = str(e)
out["events"] = [e["event"] for e in seen]
print("RESULT " + json.dumps(out))
'''


@__import__("pytest").mark.skipif(not VLLM_PY, reason="needs vLLM: set FASTRSI_VLLM_PYTHON to a "
                                  "vLLM venv's python (CPU CI has no vLLM)")
def test_vllm_layernorm_refresh_reimports_in_real_vllm(tmp_path):
    """The 61323 failure on real vLLM, on CPU: import vLLM's layernorm, refresh it and import it
    again. Then vLLM's direct_register_custom_op path: the same schema takes the new body, a
    changed schema refuses naming the op. Proves the import-time registrations pass; it does
    not start an engine (the forked-server run is in repos/vllm/kb/costs.md, Zygote)."""
    import subprocess
    script = tmp_path / "vllm_refresh.py"
    script.write_text(_VLLM_REFRESH)
    r = subprocess.run([VLLM_PY, str(script), os.path.dirname(os.path.abspath(__file__))],
                       capture_output=True, text=True, timeout=600)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, r.stderr[-3000:]     # without the fix: AssertionError: Duplicate op name
    out = json.loads(lines[-1][len("RESULT "):])
    assert out["new_class"] is True
    assert out["same_schema_value"] == [3.0]
    assert "zrvllm::probe changed its schema" in out["changed_schema"]
    # one "forgot" per CustomOp layernorm.py registers (3 in vLLM 0.15.1, 4 in later ones)
    n = out["n_layernorm_ops"]
    assert n >= 1 and out["events"][:n] == ["zygote_registry_forgot"] * n, out
    assert "zygote_op_redefined" in out["events"]
    # direct_register_custom_op's fake impl is replaced too (before: "already has an fake impl
    # registered" from Library._register_fake)
    assert "zygote_op_fake_replaced" in out["events"]
