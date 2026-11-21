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
                prep_timings = worker_side.prepare({"id": "txn-timings", "native": spec})
                self.assertTrue(prep_timings["ok"], prep_timings)
                self.assertEqual(
                    set(prep_timings["native"]["timings"]),
                    {"stage_s", "pch_s", "pch_cached", "compile_s", "link_s", "tu_s"},
                )
                worker_side.local("abort", {"txn": "txn-timings"})
                self.assertEqual(sys.modules[f"{pkg}.ops"].activate(4), 34)
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(4), 34)
                self.assertEqual(co.undo("txn-hyb")["status"], "undone")
                self.assertEqual(fake_torch.ops._C.fused_add_rms_norm(4), 5)
                self.assertEqual(sys.modules[f"{pkg}.ops"].activate(4), 5)

    def test_spec_key_tracks_env_arch_and_rejects_path_traversal(self):
        """`FASTRSI_NATIVE_ARCH` participates in `spec_key`, and `/`/`.`/`..` escapes raise ValueError."""
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\nvoid f() {}\n")
            build_root = os.path.join(tmp, "build")
            spec_auto = _native_spec(csrc, build_root)
            spec_auto["arch"] = "auto"

            with mock.patch.dict(os.environ, {"FASTRSI_NATIVE_ARCH": "90a"}):
                k_90a = N.spec_key(spec_auto)
            with mock.patch.dict(os.environ, {"FASTRSI_NATIVE_ARCH": "100a"}):
                k_100a = N.spec_key(spec_auto)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FASTRSI_NATIVE_ARCH", None)
                k_none = N.spec_key(spec_auto)
            self.assertEqual(len({k_90a, k_100a, k_none}), 3)

            stage_parent = os.path.join(tmp, "sandbox")
            csrc_stage = os.path.join(stage_parent, "csrc")
            os.makedirs(csrc_stage, exist_ok=True)
            escape_name = os.path.join(stage_parent, "escape.cu")
            for bad_rel in ("../escape.cu", "sub/../../escape.cu", "/etc/passwd", "a//b.cu", "a\\b.cu"):
                with self.assertRaises(ValueError):
                    N._apply_source_changes(csrc_stage, {"sources": {bad_rel: "// pwn\n"}})
                with self.assertRaises(ValueError):
                    N._apply_source_changes(csrc_stage, {"edits": [[bad_rel, "a", "b"]]})
                bad_file_spec = _native_spec(csrc, build_root, filename=bad_rel)
                with (
                    mock.patch.dict(sys.modules, {"torch": fake_torch}),
                    mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                    mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory([])),
                ):
                    with self.assertRaises(ValueError):
                        N.build_native(bad_file_spec)
                    bad_src_spec = _native_spec(csrc, build_root, sources={bad_rel: "// pwn\n"})
                    with self.assertRaises(ValueError):
                        N.build_native(bad_src_spec)
                    bad_edit_spec = _native_spec(csrc, build_root, edits=[[bad_rel, "a", "b"]])
                    with self.assertRaises(ValueError):
                        N.build_native(bad_edit_spec)
            self.assertFalse(os.path.exists(escape_name))

    def test_parallel_gxx_registration_and_nvcc_fallback(self):
        """Compiles `reg.cpp` with `g++` in parallel with `nvcc` + links with `g++ -shared` when
        adjacent CUDA libs exist, and falls back cleanly to `nvcc` for both when `g++` is absent."""
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\nvoid f() {}\n")
            cuda_root = os.path.join(tmp, "cuda")
            os.makedirs(os.path.join(cuda_root, "bin"), exist_ok=True)
            os.makedirs(os.path.join(cuda_root, "include"), exist_ok=True)
            os.makedirs(os.path.join(cuda_root, "lib64"), exist_ok=True)
            fake_nvcc = os.path.join(cuda_root, "bin", "nvcc")
            for p in (fake_nvcc, os.path.join(cuda_root, "include", "cuda_runtime.h"), os.path.join(cuda_root, "lib64", "libcudart.so")):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write("stub\n")
            os.chmod(fake_nvcc, 0o755)

            spec_gxx = _native_spec(csrc, os.path.join(tmp, "b1"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// gxx"]])
            calls_gxx = []
            t0 = time.perf_counter()
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value=fake_nvcc),
                mock.patch.object(N, "find_gxx", return_value="/usr/bin/g++"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls_gxx, delay_s=0.12)),
            ):
                built_gxx = N.build_native(spec_gxx)
            elapsed = time.perf_counter() - t0
            self.assertTrue(os.path.isfile(built_gxx["so"]))
            self.assertLess(elapsed, 0.22, f"expected parallel compile (<0.22s), took {elapsed:.3f}s")
            compiles = [c for c in calls_gxx if "-c" in c]
            links = [c for c in calls_gxx if "-shared" in c]
            self.assertEqual(len(compiles), 2)
            self.assertEqual(len(links), 1)
            cu_cmd = next(c for c in compiles if c[0] == fake_nvcc)
            reg_cmd = next(c for c in compiles if c[0] == "/usr/bin/g++")
            self.assertIn("-split-compile=16", cu_cmd)
            self.assertIn("--threads=16", cu_cmd)
            self.assertIn("reg.cpp", reg_cmd)
            self.assertIn(f"-I{os.path.join(cuda_root, 'include')}", reg_cmd)
            self.assertEqual(links[0][0], "/usr/bin/g++")
            self.assertIn("-lcudart", links[0])

            spec_nvcc = _native_spec(csrc, os.path.join(tmp, "b2"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// nvcc"]])
            calls_nvcc = []
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value=fake_nvcc),
                mock.patch.object(N, "find_gxx", return_value=None),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls_nvcc)),
            ):
                built_nvcc = N.build_native(spec_nvcc)
            self.assertTrue(os.path.isfile(built_nvcc["so"]))
            compiles2 = [c for c in calls_nvcc if "-c" in c]
            links2 = [c for c in calls_nvcc if "-shared" in c]
            self.assertTrue(all(c[0] == fake_nvcc for c in compiles2 + links2))
            self.assertTrue(any("reg.cu" in c for c in compiles2))

    def test_stage_csrc_overlay_nested_edits_and_build_native(self):
        """`_stage_csrc_overlay` and `build_native` symlink untouched siblings/subtrees, materialize
        real copies for edits, leave `csrc_root` bytes/mtimes untouched, and reject `../` escapes."""
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc_root = os.path.join(tmp, "csrc")
            tree = {
                "cuda_compat.h": "// compat header\n",
                "attention/attention_kernels.cu": "// untouched attention kernel\n",
                "quantization/fp8/common.cu": "// untouched fp8\n",
                "libtorch_stable/ops.h": "// stable ops header\n",
                "libtorch_stable/detail/helper.cuh": "// nested detail header\n",
                "libtorch_stable/pos_encoding_kernels.cu": "// pos_encoding base\nvoid rotary_embedding() {}\n",
                "libtorch_stable/activation_kernels.cu": "// activation base\nvoid silu_and_mul() {}\n",
                "libtorch_stable/replaced_kernels.cu": "// old replaced file\n",
            }
            for rel, body in tree.items():
                path = os.path.join(csrc_root, *rel.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(body)
            # Snapshot bytes and mtime_ns of every file in csrc_root.
            before_snap = {}
            for rel in tree:
                p = os.path.join(csrc_root, *rel.split("/"))
                st = os.stat(p)
                with open(p, "rb") as fh:
                    before_snap[rel] = (fh.read(), st.st_mtime_ns)

            # Direct `_stage_csrc_overlay` verification.
            direct_stage = os.path.join(tmp, "direct_stage", "csrc")
            pos_rel = "libtorch_stable/pos_encoding_kernels.cu"
            act_rel = "libtorch_stable/activation_kernels.cu"
            rep_rel = "libtorch_stable/replaced_kernels.cu"
            new_rel = "new_dir/custom_kernel.cu"
            N._stage_csrc_overlay(
                csrc_root,
                direct_stage,
                [pos_rel, act_rel, rep_rel, new_rel],
                source_rels={rep_rel, new_rel},
            )
            self.assertTrue(os.path.isdir(direct_stage) and not os.path.islink(direct_stage))
            stable_dir = os.path.join(direct_stage, "libtorch_stable")
            self.assertTrue(os.path.isdir(stable_dir) and not os.path.islink(stable_dir))
            new_dir = os.path.join(direct_stage, "new_dir")
            self.assertTrue(os.path.isdir(new_dir) and not os.path.islink(new_dir))

            # All directories under csrc_root are real directories (never symlinks) so
            # `#include "../..."` from any subdir stays inside `direct_stage`.
            for sub_dir in (
                "attention",
                "quantization",
                "quantization/fp8",
                "libtorch_stable",
                "libtorch_stable/detail",
            ):
                staged_d = os.path.join(direct_stage, *sub_dir.split("/"))
                self.assertTrue(os.path.isdir(staged_d) and not os.path.islink(staged_d), f"expected real dir: {sub_dir}")

            # Only untouched regular files are symlinks pointing to csrc_root.
            for untouched in (
                "cuda_compat.h",
                "attention/attention_kernels.cu",
                "quantization/fp8/common.cu",
                "libtorch_stable/ops.h",
                "libtorch_stable/detail/helper.cuh",
            ):
                staged_p = os.path.join(direct_stage, *untouched.split("/"))
                self.assertTrue(os.path.islink(staged_p), f"expected symlink: {untouched}")
                self.assertEqual(
                    os.readlink(staged_p),
                    os.path.realpath(os.path.join(csrc_root, *untouched.split("/"))),
                )

            # Edited targets are materialized regular files; `sources` targets are not created yet.
            for edited in (pos_rel, act_rel):
                staged_p = os.path.join(direct_stage, *edited.split("/"))
                self.assertTrue(os.path.isfile(staged_p) and not os.path.islink(staged_p))
            for src_rel in (rep_rel, new_rel):
                staged_p = os.path.join(direct_stage, *src_rel.split("/"))
                self.assertFalse(os.path.lexists(staged_p))

            # `_apply_source_changes` refuses to mutate untouched symlinked files in `csrc_root`.
            with self.assertRaises(ValueError):
                N._apply_source_changes(direct_stage, {"edits": [["libtorch_stable/ops.h", "stable", "hacked"]]})
            with self.assertRaises(ValueError):
                N._apply_source_changes(direct_stage, {"sources": {"attention/attention_kernels.cu": "// pwn\n"}})

            # Traversal in `_stage_csrc_overlay` raises ValueError.
            for bad_rel in ("../escape.cu", "libtorch_stable/../../escape.cu", "/etc/passwd", "a\\b.cu"):
                with self.assertRaises(ValueError):
                    N._stage_csrc_overlay(csrc_root, os.path.join(tmp, "bad_stage"), [bad_rel])

            # Full `build_native` with nested edits + source override + untouched compiled TU.
            stage_observations = []
            obs_lock = threading.Lock()
            base_fake_run = _fake_nvcc_factory([])

            def inspecting_run(cmd, cwd=None, **kw):
                if "-c" in cmd:
                    with obs_lock:
                        take_snap = not stage_observations
                        if take_snap:
                            stage_observations.append(None)
                    if take_snap:
                        csrc_s = os.path.join(cwd, "csrc")
                        contents = {}
                        for rel in (pos_rel, act_rel, rep_rel):
                            with open(os.path.join(csrc_s, *rel.split("/")), encoding="utf-8") as fh:
                                contents[rel] = fh.read()
                        stage_observations[0] = {
                            "real_dirs": {
                                rel: (
                                    os.path.isdir(os.path.join(csrc_s, *rel.split("/")))
                                    and not os.path.islink(os.path.join(csrc_s, *rel.split("/")))
                                )
                                for rel in ("attention", "quantization", "quantization/fp8", "libtorch_stable/detail")
                            },
                            "symlinks": {
                                rel: os.path.islink(os.path.join(csrc_s, *rel.split("/")))
                                for rel in (
                                    "cuda_compat.h",
                                    "attention/attention_kernels.cu",
                                    "quantization/fp8/common.cu",
                                    "libtorch_stable/ops.h",
                                    "libtorch_stable/detail/helper.cuh",
                                )
                            },
                            "real_files": {
                                rel: (
                                    os.path.isfile(os.path.join(csrc_s, *rel.split("/")))
                                    and not os.path.islink(os.path.join(csrc_s, *rel.split("/")))
                                )
                                for rel in (pos_rel, act_rel, rep_rel)
                            },
                            "contents": contents,
                        }
                return base_fake_run(cmd, cwd=cwd, **kw)

            spec = {
                "csrc_root": csrc_root,
                "build_root": os.path.join(tmp, "build_overlay"),
                "files": [pos_rel, act_rel, "attention/attention_kernels.cu"],
                "edits": [
                    [pos_rel, "// pos_encoding base", "// pos_encoding fast"],
                    [act_rel, "// activation base", "// activation fast"],
                ],
                "sources": {rep_rel: "// fresh replacement\n"},
                "ops": {OP_NAME: {"fn": OP_NAME, "schema": OP_SCHEMA}},
                "arch": "100",
            }
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N, "find_gxx", return_value=None),
                mock.patch.object(N.subprocess, "run", side_effect=inspecting_run),
            ):
                built = N.build_native(spec)
                self.assertTrue(os.path.isfile(built["so"]))

                # Unmodified-only build (no edits, no sources) also succeeds cleanly over symlinks.
                untouched_spec = _native_spec(csrc_root, os.path.join(tmp, "build_untouched"), filename=pos_rel)
                built_untouched = N.build_native(untouched_spec)
                self.assertTrue(os.path.isfile(built_untouched["so"]))

                # Traversal through build_native still raises ValueError.
                with self.assertRaises(ValueError):
                    N.build_native(_native_spec(csrc_root, os.path.join(tmp, "b_err"), edits=[["../escape.cu", "a", "b"]]))

            self.assertEqual(len(stage_observations), 1)
            obs = stage_observations[0]
            self.assertTrue(all(obs["real_dirs"].values()), obs["real_dirs"])
            self.assertTrue(all(obs["symlinks"].values()), obs["symlinks"])
            self.assertTrue(all(obs["real_files"].values()), obs["real_files"])
            self.assertIn("// pos_encoding fast", obs["contents"][pos_rel])
            self.assertIn("// activation fast", obs["contents"][act_rel])
            self.assertEqual(obs["contents"][rep_rel], "// fresh replacement\n")

            # Every file in `csrc_root` retains identical bytes and mtime_ns after build + cleanup.
            for rel, (orig_bytes, orig_mtime) in before_snap.items():
                p = os.path.join(csrc_root, *rel.split("/"))
                self.assertTrue(os.path.isfile(p) and not os.path.islink(p), f"missing/corrupted {rel}")
                with open(p, "rb") as fh:
                    self.assertEqual(fh.read(), orig_bytes, f"bytes changed for {rel}")
                self.assertEqual(os.stat(p).st_mtime_ns, orig_mtime, f"mtime changed for {rel}")


    def test_tu_threads_scaling(self):
        self.assertEqual(N._tu_threads(16, 1), 16)
        self.assertEqual(N._tu_threads(16, 2), 8)
        self.assertEqual(N._tu_threads(16, 4), 4)
        self.assertEqual(N._tu_threads(0, 2), 0)
        self.assertEqual(N._tu_threads(1, 4), 1)

    def test_multi_tu_threads_and_phase_timings(self):
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, filename="libtorch_stable/a.cu", body="// a\n")
            with open(os.path.join(csrc, "libtorch_stable/b.cu"), "w", encoding="utf-8") as fh:
                fh.write("// b\n")
            build_root = os.path.join(tmp, "build")
            spec = {
                "csrc_root": csrc,
                "build_root": build_root,
                "files": ["libtorch_stable/a.cu", "libtorch_stable/b.cu"],
                "ops": {OP_NAME: {"fn": OP_NAME, "schema": OP_SCHEMA}},
                "arch": "100",
                "threads": 16,
                "edits": [["libtorch_stable/a.cu", "// a", "// a v2"]],
            }
            calls = []
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N, "find_gxx", return_value="/usr/bin/g++"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls)),
            ):
                built = N.build_native(spec)
                cached = N.build_native(spec)

            self.assertFalse(built["cached"])
            nvcc_compiles = [c for c in calls if "-c" in c and c[0] == "/usr/local/cuda/bin/nvcc"]
            self.assertEqual(len(nvcc_compiles), 2)
            for cmd in nvcc_compiles:
                self.assertIn("--threads=8", cmd)

            timings = built["timings"]
            self.assertIn("stage_s", timings)
            self.assertIn("pch_s", timings)
            self.assertIn("pch_cached", timings)
            self.assertIn("compile_s", timings)
            self.assertIn("link_s", timings)
            self.assertEqual(
                set(timings["tu_s"]),
                {"csrc/libtorch_stable/a.cu", "csrc/libtorch_stable/b.cu", "reg.cpp"},
            )

            self.assertTrue(cached["cached"])
            self.assertEqual(
                cached["timings"],
                {"stage_s": 0.0, "pch_s": 0.0, "pch_cached": False, "compile_s": 0.0, "link_s": 0.0, "tu_s": {}},
            )

    def test_reg_pch_cache_hit_miss_and_silent_fallback(self):
        fake_torch, _, _ = _make_fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            csrc = _make_csrc(tmp, body="// base\nvoid f() {}\n")
            build_root = os.path.join(tmp, "build")
            spec1 = _native_spec(csrc, build_root, edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v1"]])
            spec2 = _native_spec(csrc, build_root, edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v2"]])
            calls = []
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N, "find_gxx", return_value="/usr/bin/g++"),
                mock.patch.object(N.subprocess, "run", side_effect=_fake_nvcc_factory(calls)),
            ):
                b1 = N.build_native(spec1)
                b2 = N.build_native(spec2)

            self.assertFalse(b1["timings"]["pch_cached"])
            self.assertTrue(b2["timings"]["pch_cached"])
            pch_builds = [c for c in calls if "c++-header" in c]
            self.assertEqual(len(pch_builds), 1)
            reg_compiles = [c for c in calls if "reg.cpp" in c]
            self.assertEqual(len(reg_compiles), 2)
            for cmd in reg_compiles:
                self.assertIn("-include", cmd)
                self.assertTrue(os.path.isfile(cmd[cmd.index("-include") + 1] + ".gch"))

            # When PCH compile fails, build_native falls back cleanly without -include.
            raw_calls = []
            raw_delegate = _fake_nvcc_factory(raw_calls)

            def fail_pch_only(cmd, cwd=None, **kw):
                if "c++-header" in cmd:
                    raw_calls.append(list(cmd))
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="pch fail")
                return raw_delegate(cmd, cwd=cwd, **kw)

            spec3 = _native_spec(csrc, os.path.join(tmp, "build-fail"), edits=[["libtorch_stable/layernorm_kernels.cu", "// base", "// v3"]])
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
                mock.patch.object(N, "find_gxx", return_value="/usr/bin/g++"),
                mock.patch.object(N.subprocess, "run", side_effect=fail_pch_only),
            ):
                b3 = N.build_native(spec3)

            self.assertTrue(os.path.isfile(b3["so"]))
            self.assertFalse(b3["timings"]["pch_cached"])
            reg_fail_cmd = next(c for c in raw_calls if "reg.cpp" in c)
            self.assertNotIn("-include", reg_fail_cmd)

    def test_stage_csrc_overlay_parent_header_visibility_from_subdir(self):
        """Edited parent headers are visible via `#include "../..."` from untouched subdir files."""
        with tempfile.TemporaryDirectory() as tmp:
            csrc_root = os.path.join(tmp, "csrc")
            tree = {
                "cuda_compat.h": "#define PARENT_VAL 1\n",
                "attention/attention_kernels.cu": '#include "../cuda_compat.h"\nint v = PARENT_VAL;\n',
                "libtorch_stable/ops.h": "#define OPS_VAL 10\n",
                "libtorch_stable/detail/helper.cuh": '#include "../ops.h"\nint w = OPS_VAL;\n',
            }
            for rel, body in tree.items():
                p = os.path.join(csrc_root, *rel.split("/"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(body)

            csrc_stage = os.path.join(tmp, "stage", "csrc")
            spec = {
                "edits": [
                    ["cuda_compat.h", "PARENT_VAL 1", "PARENT_VAL 2"],
                    ["libtorch_stable/ops.h", "OPS_VAL 10", "OPS_VAL 20"],
                ]
            }
            N._stage_csrc_overlay(csrc_root, csrc_stage, ["cuda_compat.h", "libtorch_stable/ops.h"])
            N._apply_source_changes(csrc_stage, spec)

            # No directory inside csrc_stage may be a symlink.
            for dirpath, dirnames, _ in os.walk(csrc_stage):
                self.assertFalse(os.path.islink(dirpath), dirpath)
                for d in dirnames:
                    self.assertFalse(os.path.islink(os.path.join(dirpath, d)), d)

            # Direct filesystem resolution of `../` from untouched subdirs reads edited copies.
            for sub_file, rel_hdr, expected, orig in (
                ("attention/attention_kernels.cu", "../cuda_compat.h", "PARENT_VAL 2", "PARENT_VAL 1"),
                ("libtorch_stable/detail/helper.cuh", "../ops.h", "OPS_VAL 20", "OPS_VAL 10"),
            ):
                staged_sub = os.path.join(csrc_stage, *sub_file.split("/"))
                self.assertTrue(os.path.islink(staged_sub))
                resolved = os.path.join(os.path.dirname(staged_sub), rel_hdr)
                with open(resolved, encoding="utf-8") as fh:
                    self.assertIn(expected, fh.read())
                with open(os.path.join(csrc_root, os.path.relpath(os.path.normpath(resolved), csrc_stage)), encoding="utf-8") as fh:
                    self.assertIn(orig, fh.read())

            # When host `g++` is present, verify real preprocessor `#include "../..."` output.
            gxx = N.find_gxx()
            if gxx:
                for sub_file, expected_line in (
                    ("attention/attention_kernels.cu", "int v = 2;"),
                    ("libtorch_stable/detail/helper.cuh", "int w = 20;"),
                ):
                    proc = subprocess.run(
                        [gxx, "-E", "-P", "-x", "c++", os.path.join(csrc_stage, *sub_file.split("/"))],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertIn(expected_line, proc.stdout)

    def test_reg_pch_thread_safe_tmp_and_torch_header_invalidation(self):
        """`_ensure_reg_pch` isolates per-thread `tmp_gch` under `flock` and invalidates on
        torch header stat or `torch.__version__` changes."""
        with tempfile.TemporaryDirectory() as tmp:
            torch_root = os.path.join(tmp, "stage-torch")
            torch_inc = os.path.join(torch_root, "include")
            hdr_path = os.path.join(torch_inc, "torch", "csrc", "stable", "library.h")
            os.makedirs(os.path.dirname(hdr_path), exist_ok=True)
            with open(hdr_path, "w", encoding="utf-8") as fh:
                fh.write("// torch stable library v1\n")

            fake_gxx = os.path.join(tmp, "g++")
            with open(fake_gxx, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
            os.chmod(fake_gxx, 0o755)

            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "2.11.0"
            build_root = os.path.join(tmp, "build")
            stage_dir = os.path.join(build_root, "k1", "stage-999")
            gxx_flags = ["-std=c++20", f"-I{torch_inc}", "-I.", "-Icsrc", f"-I{stage_dir}/csrc"]

            # 1) Thread A fails (TimeoutExpired) while Thread B waits on flock; Thread A's
            # cleanup runs inside flock on its own tid-tagged tmp file and never deletes Thread B's.
            tmp_outputs = []
            tmp_lock = threading.Lock()
            t_a_entered = threading.Event()
            release_t_a = threading.Event()

            def race_run(cmd, cwd=None, **kw):
                out_path = cmd[cmd.index("-o") + 1]
                with tmp_lock:
                    idx = len(tmp_outputs)
                    tmp_outputs.append(out_path)
                if idx == 0:
                    with open(out_path, "wb") as fh:
                        fh.write(b"partial-a")
                    t_a_entered.set()
                    release_t_a.wait(timeout=5)
                    raise subprocess.TimeoutExpired(cmd, 120)
                # Thread B runs after Thread A exits its flock critical section: Thread A's tmp must be gone.
                self.assertFalse(os.path.exists(tmp_outputs[0]), "Thread A failed to unlink tmp_gch before releasing flock")
                with open(out_path, "wb") as fh:
                    fh.write(b"gch-ok-b")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            results = [None, None]
            errors = []

            def run_slot(slot_idx):
                try:
                    results[slot_idx] = N._ensure_reg_pch(
                        build_root, fake_gxx, gxx_flags, stable=True, stage_dir=stage_dir
                    )
                except BaseException as exc:
                    errors.append(exc)

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N.subprocess, "run", side_effect=race_run),
            ):
                ta = threading.Thread(target=run_slot, args=(0,))
                tb = threading.Thread(target=run_slot, args=(1,))
                ta.start()
                self.assertTrue(t_a_entered.wait(timeout=5))
                tb.start()
                time.sleep(0.05)
                release_t_a.set()
                ta.join(10)
                tb.join(10)

            self.assertFalse(errors, errors)
            self.assertIsNone(results[0][0])
            self.assertIsNotNone(results[1][0])
            self.assertTrue(os.path.isfile(results[1][0] + ".gch"))
            self.assertEqual(len(tmp_outputs), 2)
            self.assertNotEqual(tmp_outputs[0], tmp_outputs[1])

            # 2) Same torch header + version hits cache; modifying header or __version__ invalidates.
            def ok_run(cmd, cwd=None, **kw):
                out_path = cmd[cmd.index("-o") + 1]
                with open(out_path, "wb") as fh:
                    fh.write(b"gch-fresh")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(N.subprocess, "run", side_effect=ok_run),
            ):
                hpp_hit, _, cached_hit = N._ensure_reg_pch(
                    build_root, fake_gxx, gxx_flags, stable=True, stage_dir=stage_dir
                )
                self.assertTrue(cached_hit)
                self.assertEqual(hpp_hit, results[1][0])

                # In-place header upgrade (different size + mtime_ns) produces a new PCH key.
                with open(hdr_path, "w", encoding="utf-8") as fh:
                    fh.write("// torch stable library v2 (upgraded header)\n")
                hpp_hdr, _, cached_hdr = N._ensure_reg_pch(
                    build_root, fake_gxx, gxx_flags, stable=True, stage_dir=stage_dir
                )
                self.assertFalse(cached_hdr)
                self.assertNotEqual(hpp_hdr, hpp_hit)

                # Changing torch.__version__ produces another distinct PCH key.
                fake_torch.__version__ = "2.12.0"
                hpp_ver, _, cached_ver = N._ensure_reg_pch(
                    build_root, fake_gxx, gxx_flags, stable=True, stage_dir=stage_dir
                )
                self.assertFalse(cached_ver)
                self.assertNotIn(hpp_ver, {hpp_hit, hpp_hdr})

    def test_pch_portable_flag_filtering(self):
        """`_filter_pch_portable_flags` handles paired `-I`/`-isystem` tokens and compares
        resolved paths against `stage_dir` instead of matching `'/stage-'` substrings."""
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = os.path.join(tmp, "build", "stage-123")
            stage_csrc = os.path.join(stage_dir, "csrc")
            external_stage_inc = os.path.join(tmp, "opt", "stage-prod", "torch", "include")
            external_sys_inc = os.path.join(tmp, "opt", "stage-prod", "cuda", "include")
            os.makedirs(stage_csrc, exist_ok=True)
            os.makedirs(external_stage_inc, exist_ok=True)
            os.makedirs(external_sys_inc, exist_ok=True)

            raw_flags = [
                "-std=c++20",
                "-I.",
                "-Icsrc",
                "-Icsrc/libtorch_stable",
                "-I", ".",
                "-I", "csrc",
                "-isystem", "csrc/sub",
                f"-I{stage_dir}",
                "-I", stage_csrc,
                "-isystem", stage_csrc,
                f"-isystem{stage_csrc}",
                f"-I{external_stage_inc}",
                "-I", external_stage_inc,
                "-isystem", external_sys_inc,
                f"-isystem{external_sys_inc}",
                "-O1",
            ]
            filtered = N._filter_pch_portable_flags(raw_flags, stage_dir=stage_dir)
            self.assertEqual(
                filtered,
                [
                    "-std=c++20",
                    f"-I{external_stage_inc}",
                    "-I", external_stage_inc,
                    "-isystem", external_sys_inc,
                    f"-isystem{external_sys_inc}",
                    "-O1",
                ],
            )


if __name__ == "__main__":
    unittest.main()
