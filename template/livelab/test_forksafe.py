"""Golden cases for forksafe.py. CPU only: a stand-in torch.cuda whose original functions
initialize "CUDA" the way the real ones do, a stand-in NVML with fixed capabilities, and a
stand-in Triton driver; no GPU is touched.

The pain point: an engine zygote imports without CUDA so the processes it forks can use the GPU,
and some imports initialize CUDA only to ask about it. At TP>1 vLLM's API server imports
FlashInfer, which asks for the GPU's capability at import; the real
torch.cuda.get_device_capability initializes CUDA, and vLLM then spawns (instead of forks) the
engine core and every worker, each importing vLLM again (33 s and 121 s in two of our TP2
starts). Megatron's zygote imports read device properties, FastVideo's read Triton's target.

Run from the repo root (standard library only; the torch case skips without torch):
    python3 -m unittest discover -s template/livelab -p 'test_forksafe.py' -v
(or python3 -m pytest -v template/livelab/test_forksafe.py)
"""
import os, subprocess, sys, types, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import forksafe  # noqa: E402


def fake_cuda():
    """torch.cuda as the zygote sees it: CUDA not initialized until a real query runs."""
    c = types.SimpleNamespace(initialized=False)

    def get_device_capability(device=None):
        c.initialized = True          # the real one creates a CUDA context
        return (8, 0)

    def current_device():
        c.initialized = True
        return 0

    def get_device_properties(device=None):
        c.initialized = True
        return "original properties"
    c.is_initialized = lambda: c.initialized
    c.get_device_capability, c.current_device = get_device_capability, current_device
    c.get_device_properties = get_device_properties
    return c


def fake_nvml(caps):
    """NVML with one capability per physical GPU index."""
    return types.SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(caps),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetCudaComputeCapability=lambda h: caps[h])


def fake_triton():
    """sys.modules entries for Triton's GPUTarget and CudaDriver; the driver's own
    get_current_target initializes "CUDA" (records the call)."""
    calls = []

    class GPUTarget(tuple):
        def __new__(cls, backend, arch, warp_size):
            return tuple.__new__(cls, (backend, arch, warp_size))

    class CudaDriver:
        def get_current_target(self):
            calls.append("driver")
            return GPUTarget("cuda", 80, 32)
    compiler = types.ModuleType("triton.backends.compiler")
    compiler.GPUTarget = GPUTarget
    driver = types.ModuleType("triton.backends.nvidia.driver")
    driver.CudaDriver = CudaDriver
    mods = {"triton": types.ModuleType("triton"),
            "triton.backends": types.ModuleType("triton.backends"),
            "triton.backends.compiler": compiler,
            "triton.backends.nvidia": types.ModuleType("triton.backends.nvidia"),
            "triton.backends.nvidia.driver": driver}
    return mods, CudaDriver, calls


class Capability(unittest.TestCase):
    def test_capability_comes_from_nvml_and_cuda_stays_uninitialized(self):
        # Input: GPUs 2,3 visible, both 10.0 (GB200). Action: FlashInfer's import-time queries.
        # Expected: 10.0 for each form of device, CUDA never initialized. Without install() the
        # first call initializes CUDA (and the stand-in answers 8.0): vLLM would then spawn.
        nvml = fake_nvml([(9, 0), (9, 0), (10, 0), (10, 0)])
        self.assertEqual(forksafe.nvml_capability("2,3", nvml), (10, 0))
        cuda = fake_cuda()
        self.assertEqual(
            forksafe.install(cuda, capability=lambda: forksafe.nvml_visible("2,3", nvml)),
            {"installed": True})
        dev = types.SimpleNamespace(type="cuda", index=1)
        for d in (None, 0, 1, "cuda", "cuda:1", dev):
            self.assertEqual(cuda.get_device_capability(d), (10, 0), d)
        self.assertEqual(cuda.current_device(), 0)
        self.assertIs(cuda.is_initialized(), False)
        # Proves the answers match NVML for the visible GPUs; not that CUDA would say the same
        # (the GPU receipt shows the forked TP2 server serving the same greedy output).
        plain = fake_cuda()
        plain.get_device_capability(0)
        self.assertIs(plain.is_initialized(), True)

    def test_no_guess_when_visible_gpus_differ_or_device_is_not_cuda(self):
        # Input: GPUs 0,2 visible with 9.0 and 10.0. Expected: no single answer exists, so the
        # original runs (CUDA initializes, as without forksafe) and the fallback is counted.
        nvml = fake_nvml([(9, 0), (9, 0), (10, 0), (10, 0)])
        self.assertIsNone(forksafe.nvml_capability("0,2", nvml))
        self.assertIsNone(forksafe.nvml_capability("GPU-uuid-1", nvml))   # UUIDs: left to CUDA
        cuda = fake_cuda()
        forksafe.install(cuda, capability=lambda: forksafe.nvml_visible("0,2", nvml))
        before = forksafe.COUNTS["fallback"]
        self.assertEqual(cuda.get_device_capability(0), (8, 0))
        self.assertIs(cuda.is_initialized(), True)
        self.assertEqual(forksafe.COUNTS["fallback"], before + 1)
        # A non-cuda device is never answered from NVML either.
        cuda2 = fake_cuda()
        forksafe.install(cuda2, capability=lambda: ((10, 0), 2))
        self.assertEqual(cuda2.get_device_capability("cpu"), (8, 0))

    def test_an_index_past_the_visible_gpus_reaches_the_original(self):
        # Input: GPUs 2,3 visible (2 GPUs, both 10.0); get_device_capability(2), (999), "cuda:5".
        # Expected: each reaches the original (here it initializes CUDA and answers 8.0; the real
        # one raises "Invalid device id"), counted as a fallback. Before: NVML's 10.0 for any
        # index >= 0, so a bad device index got an answer where CUDA would have raised.
        nvml = fake_nvml([(9, 0), (9, 0), (10, 0), (10, 0)])
        self.assertEqual(forksafe.nvml_visible("2,3", nvml), ((10, 0), 2))
        for d in (2, 999, "cuda:5"):
            cuda = fake_cuda()
            forksafe.install(cuda, capability=lambda: forksafe.nvml_visible("2,3", nvml))
            before = forksafe.COUNTS["fallback"]
            self.assertEqual(cuda.get_device_capability(d), (8, 0), d)
            self.assertIs(cuda.is_initialized(), True, d)
            self.assertEqual(forksafe.COUNTS["fallback"], before + 1, d)

    def test_after_cuda_initializes_the_originals_answer(self):
        # In a worker CUDA is initialized (set_device): the real functions must run there.
        cuda = fake_cuda()
        forksafe.install(cuda, capability=lambda: ((10, 0), 1))
        cuda.initialized = True
        self.assertEqual(cuda.get_device_capability(0), (8, 0))
        self.assertIs(forksafe.install(cuda)["installed"], False)         # once only

    def test_real_torch_module_is_patched_without_touching_the_gpu(self):
        # Input: the real torch, the capability given (no NVML, no GPU touched). Expected: after
        # install, torch.cuda.get_device_capability and a name imported from torch.cuda later (as
        # FlashInfer does) answer 10.0 and torch.cuda.is_initialized() stays False.
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import torch, forksafe\n"
            "print(forksafe.install(torch.cuda, capability=lambda: ((10, 0), 1)))\n"
            "from torch.cuda import get_device_capability\n"
            "assert torch.cuda.get_device_capability(0) == (10, 0)\n"
            "assert get_device_capability() == (10, 0) and torch.cuda.current_device() == 0\n"
            "assert not torch.cuda.is_initialized()\n"
            "print('ok')\n" % str(HERE))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(r.stdout.strip().endswith("ok"), r.stdout)


class DeviceProperties(unittest.TestCase):
    def test_properties_answer_from_the_probe_until_cuda_is_initialized(self):
        # Input: a probe reporting 2 devices (152 and 148 SMs). Expected: None, 0, "cuda:1" and a
        # cuda device answer 152, 152, 148, 148 with CUDA still uninitialized; index 2, a cpu
        # device and any call after CUDA initializes run the original. Without install the
        # first call initializes CUDA (flashinfer's gdn kernels read multi_processor_count).
        cuda = fake_cuda()
        r = forksafe.install_properties(cuda, probe=lambda: [
            {"multi_processor_count": 152, "major": 10}, {"multi_processor_count": 148, "major": 10}])
        self.assertTrue(r["installed"])
        self.assertEqual(r["devices"], 2)
        dev = types.SimpleNamespace(type="cuda", index=1)
        self.assertEqual([cuda.get_device_properties(d).multi_processor_count
                          for d in (None, 0, "cuda:1", dev)], [152, 152, 148, 148])
        self.assertIsInstance(cuda.get_device_properties(0), forksafe.Properties)
        self.assertIs(cuda.is_initialized(), False)
        self.assertEqual(cuda.get_device_properties("cpu"), "original properties")
        cuda.initialized = False
        self.assertEqual(cuda.get_device_properties(2), "original properties")
        self.assertIs(cuda.is_initialized(), True)
        self.assertEqual(cuda.get_device_properties(0), "original properties")

    def test_a_failing_probe_installs_nothing_and_install_runs_once(self):
        # Input: a probe that raises (no GPU, or torch fails in the helper). Expected: nothing
        # patched and the reason returned; after a good install a second one is refused.
        cuda = fake_cuda()
        orig = cuda.get_device_properties

        def broken():
            raise RuntimeError("no device")
        r = forksafe.install_properties(cuda, probe=broken)
        self.assertEqual(r["installed"], False)
        self.assertIn("no device", r["why"])
        self.assertIs(cuda.get_device_properties, orig)
        forksafe.install_properties(cuda, probe=lambda: [{"multi_processor_count": 1}])
        self.assertEqual(forksafe.install_properties(cuda, probe=lambda: [])["why"],
                         "already installed")


class Unpatch(unittest.TestCase):
    def test_unpatch_puts_back_every_patched_function_and_only_those(self):
        # Input: a fake torch.cuda with install and install_properties applied, plus a function
        # someone else wrapped. Expected: the three patched names come back as the originals
        # (calling them initializes CUDA again, as in a forked child that sets its device); the
        # foreign wrapper stays; a second unpatch finds nothing.
        cuda = fake_cuda()
        origs = {n: getattr(cuda, n) for n in forksafe.PATCHED}

        def foreign(device=None):
            return "theirs"
        foreign.__wrapped__ = lambda device=None: "inner"
        cuda.synchronize = foreign
        forksafe.install(cuda, capability=lambda: ((10, 0), 1))
        forksafe.install_properties(cuda, probe=lambda: [{"multi_processor_count": 1}])
        self.assertTrue(all(getattr(cuda, n) is not origs[n] for n in forksafe.PATCHED))
        self.assertEqual(sorted(forksafe.unpatch(cuda)), sorted(forksafe.PATCHED))
        self.assertTrue(all(getattr(cuda, n) is origs[n] for n in forksafe.PATCHED))
        self.assertIs(cuda.synchronize, foreign)
        self.assertEqual(forksafe.unpatch(cuda), [])
        self.assertEqual(cuda.get_device_capability(0), (8, 0))
        self.assertIs(cuda.is_initialized(), True)

    def test_unpatch_clears_the_nvml_check_variable_only_when_asked(self):
        # PYTORCH_NVML_BASED_CUDA_CHECK keeps torch.cuda.is_available() off the CUDA runtime in
        # the zygote; a child drops it only if the zygote set it (clear_env).
        with mock.patch.dict(os.environ, {"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}):
            forksafe.unpatch(fake_cuda())
            self.assertEqual(os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"), "1")
            forksafe.unpatch(fake_cuda(), clear_env=True)
            self.assertNotIn("PYTORCH_NVML_BASED_CUDA_CHECK", os.environ)

    def test_unpatch_without_torch_loaded_changes_nothing(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            self.assertEqual(forksafe.unpatch(), [])


class TritonTarget(unittest.TestCase):
    def test_target_is_pinned_while_importing_and_restored_after(self):
        # Input: a stand-in Triton whose driver query would initialize CUDA; capability (10, 0).
        # Expected: get_current_target answers ("cuda", 100, 32) without the driver, restore()
        # puts the driver's function back. Without the pin the query reaches the driver.
        mods, CudaDriver, calls = fake_triton()
        with mock.patch.dict(sys.modules, mods):
            restore = forksafe.pin_triton_target((10, 0))
            self.assertEqual(tuple(CudaDriver().get_current_target()), ("cuda", 100, 32))
            self.assertEqual(calls, [])
            restore()
            self.assertEqual(tuple(CudaDriver().get_current_target()), ("cuda", 80, 32))
            self.assertEqual(calls, ["driver"])

    def test_capability_from_env_then_nvml(self):
        # FASTRSI_ZYGOTE_CC="9.0" wins over NVML; without it the visible GPUs' shared capability;
        # when they differ, the first visible GPU's (CUDA device 0 in the child).
        nvml = fake_nvml([(10, 3), (10, 3), (9, 0)])
        mods, CudaDriver, _ = fake_triton()
        with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, {"FASTRSI_ZYGOTE_CC": "9.0"}):
            restore = forksafe.pin_triton_target(nvml=nvml)
            self.assertEqual(tuple(CudaDriver().get_current_target()), ("cuda", 90, 32))
            restore()
        env = {k: v for k, v in os.environ.items()
               if k not in ("FASTRSI_ZYGOTE_CC", "CUDA_VISIBLE_DEVICES")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(forksafe.triton_capability(nvml=fake_nvml([(10, 0), (10, 0)])), (10, 0))
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
            self.assertEqual(forksafe.triton_capability(nvml=nvml), (10, 3))
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,0"
            self.assertEqual(forksafe.triton_capability(nvml=nvml), (9, 0))
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            self.assertEqual(forksafe.triton_capability(nvml=nvml), (9, 0))

    def test_without_triton_nothing_is_patched(self):
        # No Triton importable: restore() is a no-op and NVML is never asked.
        with mock.patch.dict(sys.modules, {"triton": None, "triton.backends": None,
                                           "triton.backends.compiler": None,
                                           "triton.backends.nvidia": None,
                                           "triton.backends.nvidia.driver": None}):
            restore = forksafe.pin_triton_target(nvml=object())
            self.assertIsNone(restore())


if __name__ == "__main__":
    unittest.main()
