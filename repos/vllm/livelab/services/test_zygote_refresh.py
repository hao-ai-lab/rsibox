"""Golden cases for refreshing edited modules in a process forked from the zygote.

Each case imports a small real package in the parent (the zygote), edits a file, forks, and
checks in the child what a server would see. They run on CPU; the vLLM-sized numbers come from
the GPU receipt (repos/vllm/kb/costs.md).
"""
import json, os, sys, textwrap, time
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
    with zr.op_redefine_guard(seen):
        lib.define("scale(Tensor x, int k) -> Tensor")
        lib.impl("scale", lambda x, k: x + k, "CPU")
        assert torch.ops.zrtest.scale(torch.ones(2), 3).tolist() == [4.0, 4.0]
        try:
            lib.define("scale(Tensor x) -> Tensor")
            raise AssertionError("a changed schema must raise")
        except RuntimeError as e:
            assert "zrtest::scale changed its schema" in str(e)
    assert seen == ["zrtest::scale"]
