"""CPU unit tests for `adapter.native` C++/CUDA side-library hot-reload (`NATIVE`).

Tests run without CUDA/vLLM using synthetic `nvcc` and `torch.ops`/`torch.library` stand-ins.
Run from repo root:
    PYTHONPATH=repos/vllm/livelab:core/livelab python3 -W error::ResourceWarning \
        -m unittest discover -s repos/vllm/livelab/adapter -p 'test_native.py' -t repos/vllm/livelab -v
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from fastrsi import lab as lab_base
from kernel import coordinator as C
from kernel import locate as L
from kernel import toy

from . import native as N
from . import recapture as RC
from . import side as S
from .test_side import worker as make_graph_worker

ADAPTER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = os.path.abspath(os.path.join(ADAPTER_ROOT, "..", "..", "..", "core", "livelab"))

OP_NAME = "fused_add_rms_norm"
OP_SCHEMA = "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()"


def _make_fake_torch():
    """Minimal pure-Python stand-in for `torch` supporting `_C`, `_Cv<key>`, `Library('IMPL')`,
    and `cuda.get_device_capability()`."""
    torch_mod = types.ModuleType("torch")
    fake_pkg = os.path.join(tempfile.gettempdir(), "fake_torch_root")
    torch_mod.__file__ = os.path.join(fake_pkg, "__init__.py")

    def base_kernel(x, delta=1):
        return x + delta

    base_ns = types.SimpleNamespace(fused_add_rms_norm=base_kernel)
    active = {"impl": None, "impl_history": [], "destroy_calls": 0}

    class _DispatcherOp:
        def __call__(self, *args, **kwargs):
            fn = active["impl"] if active["impl"] is not None else getattr(base_ns, OP_NAME)
            return fn(*args, **kwargs)

    c_ns = types.SimpleNamespace(**{OP_NAME: _DispatcherOp()})
    ops_ns = types.SimpleNamespace(_C=c_ns)

    class FakeLibrary:
        def __init__(self, ns, kind):
            self.ns = ns
            self.kind = kind
            self._destroyed = False

        def impl(self, name, fn, key):
            assert self.ns == "_C" and self.kind == "IMPL" and name == OP_NAME and key == "CUDA"
            active["impl"] = fn
            active["impl_history"].append(fn)

        def _destroy(self):
            if not self._destroyed:
                self._destroyed = True
                active["impl"] = None
                active["destroy_calls"] += 1

    torch_mod.ops = ops_ns
    torch_mod.library = types.SimpleNamespace(Library=FakeLibrary)
    torch_mod.cuda = types.SimpleNamespace(
        get_device_capability=lambda: (10, 0),
        is_initialized=lambda: True,
        is_current_stream_capturing=lambda: False,
    )
    return torch_mod, ops_ns, active


def _make_csrc(tmpdir, filename="libtorch_stable/layernorm_kernels.cu", body="// base kernel\n"):
    csrc = os.path.join(tmpdir, "csrc")
    os.makedirs(os.path.join(csrc, os.path.dirname(filename)), exist_ok=True)
    with open(os.path.join(csrc, filename), "w", encoding="utf-8") as fh:
        fh.write(body)
    return csrc


def _native_spec(csrc_root, build_root, *, edits=None, sources=None, rebind=None, filename="libtorch_stable/layernorm_kernels.cu"):
    spec = {
        "csrc_root": csrc_root,
        "build_root": build_root,
        "files": [filename],
        "ops": {
            OP_NAME: {
                "fn": OP_NAME,
                "schema": OP_SCHEMA,
                **({"rebind": list(rebind)} if rebind else {}),
            }
        },
        "arch": "100",
    }
    if edits is not None:
        spec["edits"] = edits
    if sources is not None:
        spec["sources"] = sources
    return spec


def _fake_nvcc_factory(calls, *, delta=10, delay_s=0.0):
    """Simulate `nvcc -c` and `nvcc -shared -o <tmp.so>`."""
    def fake_run(cmd, cwd=None, **_kw):
        calls.append(list(cmd))
        if delay_s and "-c" in cmd:
            time.sleep(delay_s)
        if "-shared" in cmd:
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as fh:
                fh.write(f"ELF:{delta}".encode("ascii"))
        elif "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            with open(os.path.join(cwd, cmd[out_idx]), "wb") as fh:
                fh.write(b"OBJ")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


def _fake_cdll_factory(ops_ns, versioned_deltas, dlclose_calls):
    class FakeCDLL:
        def __init__(self, so_path, mode=0):
            self.so_path = so_path
            self.mode = mode
            base = os.path.basename(so_path)
            key = base.removeprefix("libfastrsi_").removesuffix(".so")
            delta = versioned_deltas.get(key, 100)
            setattr(ops_ns, f"_Cv{key}", types.SimpleNamespace(**{OP_NAME: lambda x, d=delta: x + d}))

    def fake_cdll(path, mode=0):
        return FakeCDLL(path, mode=mode)

    return fake_cdll


class NativeUnitTests(unittest.TestCase):
    def setUp(self):
        N._LOADED_LIBS.clear()

    def tearDown(self):
        N._LOADED_LIBS.clear()

    def test_zero_gpu_import(self):
        """Importing `adapter.native` and `adapter.side` leaves `torch` and GPU modules unloaded."""
        probe = (
            "import sys\n"
            "import adapter.native, adapter.side\n"
            "bad = sorted(m for m in ('torch', 'cuda', 'triton', 'vllm') if m in sys.modules)\n"
            "assert not bad, bad\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([ADAPTER_ROOT, CORE_ROOT])}
        proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_registration_codegen_and_spec_key_disk_invalidation(self):
        """Checks `STABLE_TORCH_LIBRARY` and `TORCH_LIBRARY` `reg.cu` emitters plus disk content hashing."""
        ops = {OP_NAME: {"fn": OP_NAME, "schema": OP_SCHEMA}}
        stable_cu = N.generate_registration("abc123", ops, stable=True)
        self.assertIn("STABLE_TORCH_LIBRARY(_Cvabc123, ops)", stable_cu)
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL(_Cvabc123, CUDA, ops)", stable_cu)
        self.assertIn(f'ops.impl("{OP_NAME}", TORCH_BOX(&{OP_NAME}))', stable_cu)

        plain_cu = N.generate_registration("abc123", ops, stable=False)
        self.assertIn("TORCH_LIBRARY(_Cvabc123, ops)", plain_cu)
        self.assertIn(f'ops.impl("{OP_NAME}", &{OP_NAME})', plain_cu)

        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// version A\n")
            spec = _native_spec(csrc, tmp)
            k1 = N.spec_key(spec)
            k2 = N.spec_key(spec)
            self.assertEqual(k1, k2)
            with open(os.path.join(csrc, "libtorch_stable/layernorm_kernels.cu"), "w", encoding="utf-8") as fh:
                fh.write("// version B\n")
            self.assertNotEqual(k1, N.spec_key(spec))

    def test_compiler_cache_and_flock(self):
        """Second identical build hits cache (`build_s == 0.0`, `cached=True`), and concurrent builds
        under `flock` invoke `nvcc` only once."""
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\nvoid x() {}\n")
            build_root = os.path.join(tmp, "build")
            spec = _native_spec(csrc, build_root, edits=[["libtorch_stable/layernorm_kernels.cu", "void x() {}", "void x() { /*fast*/ }"]])
            calls = []
            results = []
            errors = []

            def worker():
                try:
                    results.append(N.build_native(spec))
                except BaseException as exc:
                    errors.append(exc)

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls, delay_s=0.05)),
            ):
                t1 = threading.Thread(target=worker)
                t2 = threading.Thread(target=worker)
                t1.start()
                t2.start()
                t1.join(10)
                t2.join(10)
            self.assertFalse(errors, errors)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["key"], results[1]["key"])
            self.assertTrue(os.path.isfile(results[0]["so"]))
            self.assertEqual(sorted(r["cached"] for r in results), [False, True])
            self.assertEqual(sum(1 for cmd in calls if "-shared" in cmd), 1)
            self.assertEqual(sum(1 for cmd in calls if "-c" in cmd), 2)

    def test_2pc_lifecycle_commit_witness_and_sub_ms_undo(self):
        """Full 2PC native lifecycle: prepare compiles & loads without dlclose, commit binds via
        `Library('_C', 'IMPL')`, live calls increment `witness`, `proof()` reports hash, and `<1ms`
        undo restores the original op via `lib._destroy()` with `at_base: ok`."""
        fake_torch, ops_ns, active = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\nvoid f() {}\n")
            spec = _native_spec(csrc, os.path.join(tmp, "build"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// new"]])
            key = N.spec_key(spec)
            calls = []
            worker_side = S.Side("worker", rank=0)
            core_side = S.Side("core")
            co = lab_base.RecaptureCoordinator(
                [
                    C.CallParticipant("worker", worker_side.dispatch),
                    C.CallParticipant("core", core_side.dispatch),
                ],
                gap=lambda fn: fn(),
                phase2_deadline_s=5.0,
            )
            dlclose_calls = []
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls)),
                mock.patch.object(N.ctypes, "CDLL", side_effect=_fake_cdll_factory(ops_ns, {key: 50}, dlclose_calls)),
            ):
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(5), 6)
                res = co.run({"id": "txn-nat1", "native": spec})
                self.assertEqual(res["outcome"], "committed", res)
                self.assertTrue(res["receipt"]["participants"]["core"]["prepare"]["abstain"])
                prep_native = res["receipt"]["participants"]["worker"]["prepare"]["native"]
                self.assertEqual(prep_native["key"], key)
                wkey = f"native:_C.{OP_NAME}:{key}"
                self.assertEqual(res["receipt"]["commit_hashes"]["participants"]["worker"][wkey], [key])

                # Live calls go through wrapper and increment witness.
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(5), 55)
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(7), 57)
                proof = co.prove("txn-nat1")
                self.assertTrue(proof["ok"], proof["receipt"]["findings"])
                self.assertEqual(proof["receipt"]["witness"]["worker"][wkey], 2)
                self.assertEqual(proof["receipt"].get("abstained"), ["core"])

                t0 = time.perf_counter()
                undone = co.undo("txn-nat1")
                undo_elapsed = time.perf_counter() - t0
                self.assertEqual(undone["status"], "undone", undone)
                self.assertLess(undo_elapsed, 0.5)
                self.assertEqual(active["destroy_calls"], 1)
                self.assertEqual(dlclose_calls, [])
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(5), 6)
                self.assertTrue(worker_side.dispatch("at_base")["ok"])
                self.assertEqual(worker_side.native.hashes("txn-nat1"), {})

    def test_graph_safety_refuses_without_r1_and_allows_with_r1(self):
        """Worker with active CUDA graphs refuses native edit without `rebuild: ['R1']` before
        invoking `nvcc`, and accepts with `rebuild: ['R1']` (capturing graphs under witness and
        restoring both graphs and native op on undo)."""
        fake_torch, ops_ns, active = _make_fake_torch()
        captured_graphs = {"FULL": {8, 1}}
        captures = []

        def fake_capture(mode, desc):
            captures.append((mode, desc, fake_torch.ops._C.fused_add_rms_norm(1)))

        eng = RC.Engine(lambda: [], captured_graphs, fake_capture)
        w = make_graph_worker(cudagraph="FULL_DECODE_ONLY", compile_mode="NONE")

        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\n")
            spec = _native_spec(csrc, os.path.join(tmp, "build"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v2"]])
            key = N.spec_key(spec)
            calls = []
            worker_side = S.Side("worker", scope={"worker": w, "recapture_engine": eng}, rank=0)
            core_side = S.Side("core")
            co = lab_base.RecaptureCoordinator(
                [
                    C.CallParticipant("worker", worker_side.dispatch),
                    C.CallParticipant("core", core_side.dispatch),
                ],
                gap=lambda fn: fn(),
                phase2_deadline_s=5.0,
                rebuild=lambda txn, rspec: worker_side.dispatch("recapture", {"txn": txn, **rspec}),
            )

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls)),
                mock.patch.object(N.ctypes, "CDLL", side_effect=_fake_cdll_factory(ops_ns, {key: 20}, [])),
            ):
                refused = co.run({"id": "txn-no-r1", "native": spec})
                self.assertEqual(refused["outcome"], "aborted", refused)
                self.assertEqual(calls, [])  # nvcc must NOT run when graph check blocks
                findings = refused["receipt"]["findings"]
                self.assertEqual([f["check"] for f in findings], ["captured_target"])
                self.assertEqual(findings[0]["route"], "R1")

                ok_res = co.run({"id": "txn-with-r1", "native": spec, "rebuild": ["R1"]})
                self.assertEqual(ok_res["outcome"], "committed", ok_res)
                self.assertEqual(ok_res["receipt"]["rebuild"]["routes"], ["R1"])
                self.assertEqual([(m, d, v) for m, d, v in captures], [("FULL", 8, 21), ("FULL", 1, 21)])

                wkey = f"native:_C.{OP_NAME}:{key}"
                self.assertEqual(worker_side.lp.witness.capture_counts.get(wkey), 2)
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(2), 22)
                proof = co.prove("txn-with-r1")
                self.assertTrue(proof["ok"], proof["receipt"]["findings"])
                self.assertEqual(proof["receipt"]["witness"]["worker"][wkey], 1)
                self.assertEqual(proof["receipt"]["witness_capture"]["worker"][wkey], 2)

                undone = co.undo("txn-with-r1")
                self.assertEqual(undone["status"], "undone")
                self.assertEqual(active["destroy_calls"], 1)
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(2), 3)
                self.assertTrue(worker_side.dispatch("at_base")["ok"])

    def test_zero_witness_on_native_in_graph_mode_routes_to_r1(self):
        """A committed native edit in CUDA-graph mode that was only captured (zero live calls)
        reports `zero_witness` with `route == 'R1'`."""
        fake_torch, ops_ns, _ = _make_fake_torch()
        eng = RC.Engine(lambda: [], {"FULL": {4}}, lambda _m, _d: fake_torch.ops._C.fused_add_rms_norm(1))
        w = make_graph_worker(cudagraph="FULL_DECODE_ONLY", compile_mode="NONE")
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\n")
            spec = _native_spec(csrc, os.path.join(tmp, "build"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v2"]])
            key = N.spec_key(spec)
            worker_side = S.Side("worker", scope={"worker": w, "recapture_engine": eng}, rank=0)
            co = lab_base.RecaptureCoordinator(
                [C.CallParticipant("worker", worker_side.dispatch)],
                gap=lambda fn: fn(),
                phase2_deadline_s=5.0,
                rebuild=lambda txn, rspec: worker_side.dispatch("recapture", {"txn": txn, **rspec}),
            )
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory([])),
                mock.patch.object(N.ctypes, "CDLL", side_effect=_fake_cdll_factory(ops_ns, {key: 20}, [])),
            ):
                res = co.run({"id": "txn-zw", "native": spec, "rebuild": ["R1"]})
                self.assertEqual(res["outcome"], "committed", res)
                proof = co.prove("txn-zw")
                self.assertFalse(proof["ok"])
                zw = [f for f in proof["receipt"]["findings"] if f["check"] == "zero_witness"]
                self.assertEqual(len(zw), 1)
                self.assertEqual(zw[0]["route"], "R1")
                self.assertIn("native CUDA kernel", zw[0]["observed"])
                self.assertEqual(co.undo("txn-zw")["status"], "undone")

    def test_hybrid_python_and_native_plan_and_abort_cleanup(self):
        """Commit fault rolls back both Python swaps and native bindings; failed native prepare
        leaves no staged state; optional `rebind` restores module attributes via journal."""
        fake_torch, ops_ns, active = _make_fake_torch()
        pkg = "nat_hyb"
        toy.load_copy(pkg)
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")])
        ops_src = toy.source("ops")
        patched_ops = ops_src.replace("    return x + 1", "    return x + 9")

        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\n")
            spec = _native_spec(
                csrc,
                os.path.join(tmp, "build"),
                edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v3"]],
                rebind=[f"{pkg}.ops:activate"],
            )
            key = N.spec_key(spec)
            worker_side = S.Side("worker", rank=0, alias={"toy": pkg})
            core_side = S.Side("core", alias={"toy": pkg})
            co = lab_base.RecaptureCoordinator(
                [
                    C.CallParticipant("worker", worker_side.dispatch),
                    C.CallParticipant("core", core_side.dispatch),
                ],
                gap=lambda fn: fn(),
                phase2_deadline_s=5.0,
            )

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory([])),
                mock.patch.object(N.ctypes, "CDLL", side_effect=_fake_cdll_factory(ops_ns, {key: 30}, [])),
            ):
                # Fault during commit rolls back native staged state cleanly.
                faulty = co.run({
                    "id": "txn-fault",
                    "modules": {"toy.ops": {"source": patched_ops}},
                    "native": spec,
                    "faults": {"core": "commit"},
                })
                self.assertEqual(faulty["outcome"], "rolled_back", faulty)
                self.assertEqual(active["destroy_calls"], 1)
                self.assertFalse(worker_side.native.has_staged("txn-fault"))
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(4), 5)
                self.assertEqual(sys.modules[f"{pkg}.ops"].activate(4), 5)

                # Bad edit snippet fails prepare cleanly without staging.
                bad_spec = _native_spec(csrc, os.path.join(tmp, "build"), edits=[["libtorch_stable/layernorm_kernels.cu", "NONEXISTENT", "x"]])
                bad_res = co.run({"id": "txn-bad", "native": bad_spec})
                self.assertEqual(bad_res["outcome"], "aborted", bad_res)
                self.assertFalse(worker_side.native.has_staged("txn-bad"))

                # Succeeding hybrid plan swaps both core+worker modules and native+rebind.
                ok_hyb = co.run({
                    "id": "txn-hyb",
                    "modules": {"toy.ops": {"source": patched_ops, "on": ["core"]}},
                    "native": spec,
                })
                self.assertEqual(ok_hyb["outcome"], "committed", ok_hyb)
                self.assertEqual(sys.modules[f"{pkg}.ops"].activate(4), 34)
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(4), 34)
                self.assertEqual(co.undo("txn-hyb")["status"], "undone")
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(4), 5)
                self.assertEqual(sys.modules[f"{pkg}.ops"].activate(4), 5)


if __name__ == "__main__":
    unittest.main()
