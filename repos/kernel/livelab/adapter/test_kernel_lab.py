"""Golden unit tests, Triton/CUDA `.cu` live-reload tests, and 11/11 conformance for `KernelLab` / `ToyKernelLab`."""
from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import adapter  # noqa: F401 - puts core/livelab on sys.path
from adapter import kernel_ops
from adapter.lab import CPU_MODULES, KernelLab, ToyKernelLab, _PATCH_SPECS, _enter_critical_region
from adapter.worker import KernelWorker, require_kernel_deps, validate_positive_int
from conformance import cases
from rsibox import harness


def _gpu_backend_available(backend: str = "triton") -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        return False
    try:
        info = require_kernel_deps(backend, "cuda:0")
        return bool(info.get("device"))
    except Exception:
        return False


class KernelLabConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases (`c0`–`c10`) on `ToyKernelLab`."""

    def test_full_conformance_c0_to_c10_cpu(self) -> None:
        failures = []
        for name in cases.CASES:
            result = cases.run(name, ToyKernelLab)
            if result["result"] != "pass":
                failures.append((name, result["result"], result["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class KernelLabGoldenTests(unittest.TestCase):
    """Standalone kernel hot-swap, Triton/CUDA `.cu` reload, KernelService hook, and fail-closed tests."""

    def setUp(self) -> None:
        self.lab = ToyKernelLab()

    def tearDown(self) -> None:
        self.lab.close()

    def test_zero_eager_gpu_or_triton_imports(self) -> None:
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.abspath(os.path.join(livelab_dir, "..", "..", "..", "core", "livelab"))
        probe = (
            "import importlib, sys\n"
            "for m in ('adapter.kernel_ops', 'adapter.worker', 'adapter.lab'):\n"
            "    importlib.import_module(m)\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'triton', 'cuda', 'cupy', 'pynvml', 'nvidia')]\n"
            "assert not bad, f'eager imports: {bad}'\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([livelab_dir, core_dir]), "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)

    def test_multi_shape_parity_and_critical_region_duck_typing(self) -> None:
        events: list[str] = []

        class _FakeKernelService:
            @contextlib.contextmanager
            def critical_region(self):
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")

        svc_lab = ToyKernelLab(kernel_service=_FakeKernelService())
        try:
            c_small = svc_lab.check({"numel": 128, "iters": 2})
            c_large = svc_lab.check({"numel": 512, "iters": 2})
            self.assertTrue(c_small["ok"] and c_large["ok"])
            self.assertEqual(c_small["output"]["numel"], 128)
            self.assertEqual(c_large["output"]["numel"], 512)
            self.assertEqual(c_small["output"]["max_abs_diff"], 0.0)
            self.assertEqual(c_large["output"]["max_abs_diff"], 0.0)
            self.assertIn("latency_ms", c_small)
            self.assertNotIn("latency_ms", c_small["output"])
            self.assertGreaterEqual(events.count("enter"), 2)
            self.assertEqual(events.count("enter"), events.count("exit"))

            km = CPU_MODULES[0]
            edited = harness.edit_once(svc_lab._base_source(km), "kernel_gain = 1.0 * KERNEL_SCALE", "kernel_gain = 2.0 * KERNEL_SCALE")
            ins = svc_lab.install({"files": {km: edited}})
            self.assertTrue(ins["ok"], ins)
            self.assertGreaterEqual(events.count("enter"), 3)
            self.assertEqual(events.count("enter"), events.count("exit"))
            chk = svc_lab.check({"numel": 512, "iters": 2})
            self.assertIs(chk["proven"], True)
            self.assertGreater(chk["output"]["max_abs_diff"], 0.1)
            fresh = svc_lab.fresh_check({"files": {km: edited}}, {"numel": 512, "iters": 2})
            self.assertTrue(fresh["ok"], fresh)
            self.assertEqual(fresh["output"], chk["output"])
        finally:
            svc_lab.close()

    def test_cuda_compile_and_load_mock_and_sticky_cuda_poison_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_nvcc = os.path.join(tmp, "fake_nvcc")
            with open(fake_nvcc, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\nout=''\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then out=\"$2\"; shift 2; continue; fi\n  shift\ndone\ntouch \"$out\"\n")
            os.chmod(fake_nvcc, 0o755)

            class _FakeLib:
                def __init__(self, path: str, mode: int = 0) -> None:
                    self.path = path
                    self.mode = mode
                    self.launch_fused_affine_silu = mock.MagicMock()

            with mock.patch.dict(os.environ, {"RSIBOX_NVCC": fake_nvcc}, clear=False), \
                 mock.patch("adapter.kernel_ops.ctypes.CDLL", side_effect=_FakeLib):
                kernel_ops._CU_LIBS.clear()
                b1 = kernel_ops.compile_and_load_cuda(kernel_ops.cuda_kernel_source(), build_root=tmp, arch="100", use_cache=True)
                self.assertFalse(b1["cached"])
                self.assertTrue(os.path.isfile(b1["so"]))
                b2 = kernel_ops.compile_and_load_cuda(kernel_ops.cuda_kernel_source(), build_root=tmp, arch="100", use_cache=True)
                self.assertTrue(b2["cached"])
                self.assertEqual(b1["key"], b2["key"])
                kernel_ops._CU_LIBS.clear()

        base_out = self.lab.check()["output"]
        self.assertTrue(self.lab.inject("sticky_cuda")["ok"])
        c_fail = self.lab.check()
        self.assertFalse(c_fail["ok"])
        self.assertEqual(self.lab.status()["state"], "poisoned")
        refused = self.lab.install(self.lab.patches()["visible"])
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["codes"], ["session_poisoned"])
        rs = self.lab.reset("respawn")
        self.assertTrue(rs["ok"], rs)
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], base_out)

    def test_refuses_constructor_edit_and_invalid_configs_fail_closed(self) -> None:
        km = CPU_MODULES[0]
        ctor_patch = harness.edit_once(self.lab._base_source(km), "str(backend), init_kernel_gain()", "str(backend), 2.0")
        res = self.lab.install({"files": {km: ctor_patch}})
        self.assertFalse(res["ok"])
        self.assertIn("constructor_edit", res["codes"])

        for bad in (0, -4, True, 1.5, "64"):
            with self.assertRaises(ValueError):
                validate_positive_int(bad, "numel")
            with self.assertRaises(ValueError):
                KernelLab({"backend": "cpu", "numel": bad}, auto_spawn=False)

        env_vars = KernelLab.launch_environment(["--backend", "triton", "--numel", "16384", "--block-size", "256"])
        self.assertEqual(env_vars["RSIBOX_KERNEL_BACKEND"], "triton")
        self.assertEqual(env_vars["RSIBOX_KERNEL_NUMEL"], "16384")
        self.assertEqual(env_vars["RSIBOX_KERNEL_BLOCK_SIZE"], "256")

        if not _gpu_backend_available("triton"):
            with self.assertRaisesRegex(RuntimeError, "requires torch|CUDA requested|requires triton"):
                KernelLab({"backend": "triton", "numel": 256}, auto_spawn=True)

    def test_cuda_auto_without_triton_device_routing_arch_hash_and_context_manager(self) -> None:
        orig_find_spec = importlib.util.find_spec

        def _no_triton_find_spec(name: str, *args, **kwargs):
            return None if name == "triton" else orig_find_spec(name, *args, **kwargs)

        with mock.patch("importlib.util.find_spec", side_effect=_no_triton_find_spec):
            for backend in ("cuda", "auto"):
                lab = KernelLab({"backend": backend, "numel": 256}, auto_spawn=False)
                self.assertEqual(lab._resolved_backend, "cuda")
                self.assertEqual(lab._init_modules, list(CPU_MODULES))
                self.assertNotIn("adapter.triton_kernels", lab._init_modules)
                p = lab.patches()
                self.assertEqual(set(p["visible"]["files"]), {CPU_MODULES[0]})
                self.assertIn(_PATCH_SPECS["cuda"][2], p["visible"]["files"][CPU_MODULES[0]])

        calls: list[tuple[str, object]] = []

        class _FakeTensor:
            def __init__(self, n: int = 8) -> None: self._n = n
            def __mul__(self, o): return self
            def __sub__(self, o): return self
            def __add__(self, o): return self
            def data_ptr(self) -> int: return 1234
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [0.0] * self._n
            def zero_(self): return None

        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda d: (calls.append(("name", d)), f"MockGPU-{d}")[1],
            set_device=lambda d: calls.append(("set_device", d)),
            manual_seed=lambda s: calls.append(("manual_seed", s)),
            get_rng_state=lambda d: ("rng", d),
            set_rng_state=lambda st, d: None,
            current_stream=lambda d: types.SimpleNamespace(cuda_stream=99),
            current_device=lambda: 1,
            get_device_capability=lambda d=0: (calls.append(("cap", d)), (9, 0) if d in (0, "cuda:0") else (10, 0))[1],
            device=lambda d: contextlib.nullcontext((calls.append(("ctx", d)), d)[1]),
            synchronize=lambda d: calls.append(("sync", d)),
        )
        fake_torch = types.SimpleNamespace(
            __version__="2.14.0-mock", cuda=fake_cuda, float32="fp32",
            Generator=lambda device=None: types.SimpleNamespace(manual_seed=lambda s: None),
            rand=lambda n, **kw: _FakeTensor(n), empty_like=lambda t: _FakeTensor(t._n),
            sigmoid=lambda t: t, manual_seed=lambda s: None, get_rng_state=lambda: "cpu_rng",
            set_rng_state=lambda s: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            fake_nvcc = os.path.join(tmp, "fake_nvcc")
            with open(fake_nvcc, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\nout=''\nwhile [ $# -gt 0 ]; do [ \"$1\" = \"-o\" ] && out=\"$2\" && shift 2 && continue; shift; done\ntouch \"$out\"\n")
            os.chmod(fake_nvcc, 0o755)
            fake_lib = types.SimpleNamespace(launch_fused_affine_silu=mock.MagicMock(return_value=0))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                 mock.patch.dict(os.environ, {"RSIBOX_NVCC": fake_nvcc}, clear=False), \
                 mock.patch("importlib.util.find_spec", side_effect=lambda n, *a, **kw: object() if n == "torch" else orig_find_spec(n, *a, **kw)), \
                 mock.patch("adapter.kernel_ops.ctypes.CDLL", return_value=fake_lib):
                kernel_ops._CU_LIBS.clear()
                deps = require_kernel_deps("cuda", "cuda:1")
                self.assertEqual(deps["device"], "MockGPU-cuda:1")
                b90 = kernel_ops.compile_and_load_cuda(kernel_ops.cuda_kernel_source(), build_root=tmp, device="cuda:0", use_cache=True)
                b100 = kernel_ops.compile_and_load_cuda(kernel_ops.cuda_kernel_source(), build_root=tmp, device="cuda:1", use_cache=True)
                self.assertNotEqual(b90["key"], b100["key"])
                kw = KernelWorker({"backend": "cuda", "numel": 8, "device": "cuda:1", "build_root": tmp, "iters": 1})
                kw.run_kernel({"numel": 8, "iters": 1})
                self.assertIn(("set_device", "cuda:1"), calls)
                self.assertIn(("ctx", "cuda:1"), calls)
                self.assertIn(("cap", "cuda:1"), calls)
                kernel_ops._CU_LIBS.clear()

        cm_events: list[str] = []

        @contextlib.contextmanager
        def _cm():
            cm_events.append("in")
            try: yield
            finally: cm_events.append("out")

        class _CallableCM:
            def __call__(self): raise AssertionError("must enter __enter__, not __call__")
            def __enter__(self): cm_events.append("c_in"); return self
            def __exit__(self, *a): cm_events.append("c_out")

        with _enter_critical_region(_cm()): cm_events.append("body")
        with _enter_critical_region(_CallableCM()): cm_events.append("c_body")
        self.assertEqual(cm_events, ["in", "body", "out", "c_in", "c_body", "c_out"])


if __name__ == "__main__":
    unittest.main()
