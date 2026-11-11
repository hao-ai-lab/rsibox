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
    python3 -m unittest discover -s core/livelab -p 'test_forksafe.py' -v
(or python3 -m pytest -v core/livelab/test_forksafe.py)
"""
import gc, json, os, shutil, subprocess, sys, tempfile, types, unittest
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


class CacheAndDeviceIndexFixes(unittest.TestCase):
    def test_cap_of_cache_keys_on_pid_and_visible_devices(self):
        """Input: default `install(cuda)` (using `cap_of`'s internal `lru_cache`) with a stand-in
        NVML where GPU 0 is (9, 0) and GPU 1 is (10, 0). Action: query under
        CUDA_VISIBLE_DEVICES=0, then change CUDA_VISIBLE_DEVICES=1 (as a forked worker does) and
        query again, and fork a real child with a different visibility. Expected: changing
        CUDA_VISIBLE_DEVICES or PID re-queries NVML and returns (10, 0) instead of a stale parent
        (9, 0). Pain point (#117): a 0-arg lru_cache froze the parent's visibility tuple across
        forks and env updates."""
        nvml_standin = fake_nvml([(9, 0), (10, 0)])
        cuda_standin = fake_cuda()
        with mock.patch.object(forksafe, "_nvml", return_value=nvml_standin):
            forksafe.install(cuda_standin)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                self.assertEqual(cuda_standin.get_device_capability(0), (9, 0))
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                self.assertEqual(cuda_standin.get_device_capability(0), (10, 0))
            read_fd, write_fd = os.pipe()
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                cap = cuda_standin.get_device_capability(0)
                os.write(write_fd, json.dumps(list(cap)).encode())
                os._exit(0)
            os.close(write_fd)
            child_payload = os.read(read_fd, 64)
            os.close(read_fd)
            os.waitpid(child_pid, 0)
            self.assertEqual(tuple(json.loads(child_payload)), (10, 0))

    def test_malformed_cuda_device_strings_return_none_and_fall_back(self):
        """Input: `_index` and `get_device_capability` called with malformed device strings
        `"cuda:abc"` and `"cuda:0:extra"`. Expected: `_index` returns None (never raises
        ValueError) and `get_device_capability` falls back cleanly to the original function.
        Pain point (#117): `int(device.split(':')[1])` raised ValueError on `"cuda:abc"` and
        silently ignored trailing segments on `"cuda:0:extra"`."""
        self.assertIsNone(forksafe._index("cuda:abc"))
        self.assertIsNone(forksafe._index("cuda:0:extra"))
        self.assertEqual(forksafe._index("cuda:0"), 0)
        self.assertEqual(forksafe._index("cuda:-1"), -1)
        cuda_standin = fake_cuda()
        forksafe.install(cuda_standin, capability=lambda: ((10, 0), 2))
        for bad_device in ("cuda:abc", "cuda:0:extra"):
            cuda_standin.initialized = False
            before_fallbacks = forksafe.COUNTS["fallback"]
            self.assertEqual(cuda_standin.get_device_capability(bad_device), (8, 0))
            self.assertTrue(cuda_standin.is_initialized())
            self.assertEqual(forksafe.COUNTS["fallback"], before_fallbacks + 1)


def fake_nvml_uuids(caps, uuids, cpus=None):
    """fake_nvml plus GPUs named by UUID (uuids[i] is GPU i's) and, when `cpus` is given,
    each GPU's CPU affinity mask (cpus[i]: the CPU ids local to GPU i)."""
    nvml = fake_nvml(caps)
    nvml.nvmlDeviceGetHandleByUUID = lambda uuid: uuids.index(uuid)
    if cpus is not None:
        nvml.nvmlDeviceGetCpuAffinity = lambda handle, words: [
            sum(1 << c for c in cpus[handle] if c // 64 == w) for w in range(words)]
    return nvml


class VisibleDevicesAfterFork(unittest.TestCase):
    """A forked child sets CUDA_VISIBLE_DEVICES before CUDA initializes; Slurm names GPUs by
    UUID."""

    def test_properties_follow_the_childs_visible_devices(self):
        """Input: properties probed with CUDA_VISIBLE_DEVICES="0,1" (152 and 148 SMs); the
        child then sets "1", then "", then a UUID. Expected: get_device_properties(0) answers
        148 for "1"; for "" and the UUID the original runs (it raises or initializes CUDA as
        without forksafe). Pain point: the child read physical GPU 0's properties for its
        device 0 (it is physical GPU 1). Not: a probe run under UUIDs."""
        cuda = fake_cuda()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            forksafe.install_properties(cuda, probe=lambda: [
                {"multi_processor_count": 152}, {"multi_processor_count": 148}])
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self.assertEqual(cuda.get_device_properties(0).multi_processor_count, 148)
            self.assertIs(cuda.is_initialized(), False)
            for hidden in ("", "GPU-aaaa"):
                cuda.initialized = False
                os.environ["CUDA_VISIBLE_DEVICES"] = hidden
                self.assertEqual(cuda.get_device_properties(0), "original properties", hidden)

    def test_triton_capability_with_no_visible_gpu_uuids_and_cc_forms(self):
        """Input: CUDA_VISIBLE_DEVICES="" (every GPU hidden); "GPU-b,GPU-a" on mixed GPUs
        (GPU 0 is 9.0, GPU 1 is 10.0, GPU-b is GPU 1); FASTRSI_ZYGOTE_CC "100", "sm_90a",
        "10.3" and "ten". Expected: RuntimeError for ""; (10, 0) for the UUIDs (the first
        visible GPU); (10, 0), (9, 0), (10, 3); ValueError naming the variable for "ten".
        Pain point: "" answered GPU 0's capability for a process with no GPU, a UUID list
        answered physical GPU 0's, and "100" raised an unpacking error. Not: MIG names."""
        nvml = fake_nvml_uuids([(9, 0), (10, 0)], ["GPU-a", "GPU-b"])
        env = {k: v for k, v in os.environ.items()
               if k not in ("FASTRSI_ZYGOTE_CC", "CUDA_VISIBLE_DEVICES")}
        with mock.patch.dict(os.environ, env, clear=True):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            with self.assertRaises(RuntimeError):
                forksafe.triton_capability(nvml=nvml)
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-b,GPU-a"
            self.assertEqual(forksafe.triton_capability(nvml=nvml), (10, 0))
            for text, cap in (("100", (10, 0)), ("sm_90a", (9, 0)), ("10.3", (10, 3))):
                os.environ["FASTRSI_ZYGOTE_CC"] = text
                self.assertEqual(forksafe.triton_capability(nvml=nvml), cap, text)
            os.environ["FASTRSI_ZYGOTE_CC"] = "ten"
            with self.assertRaisesRegex(ValueError, "FASTRSI_ZYGOTE_CC"):
                forksafe.triton_capability(nvml=nvml)

    def test_affinity_of_a_gpu_named_by_uuid(self):
        """Input: NVML with 2 GPUs (GPU-a local to CPUs 0-1, GPU-b to CPUs 2-3) and
        CUDA_VISIBLE_DEVICES="GPU-b" (Slurm's form); then the same without NVML, over a fake
        sysfs. Expected: device 0 resolves to GPU-b's CPUs [2, 3] through NVML; without NVML
        no CPUs (source "none"), never GPU 0's. Pain point: UUIDs were skipped by an
        isdigit() filter, so every rank bound to GPU 0's NUMA node. Not: real NVML or sysfs
        (owed on a GPU node)."""
        nvml = fake_nvml_uuids([(10, 0)] * 2, ["GPU-a", "GPU-b"], cpus=[[0, 1], [2, 3]])
        sysfs = tempfile.mkdtemp(prefix="forksafe-uuid-")
        self.addCleanup(shutil.rmtree, sysfs, True)
        for addr, cpus in (("0000:01:00.0", "0-1"), ("0000:02:00.0", "2-3")):
            os.makedirs(os.path.join(sysfs, addr))
            for name, text in (("local_cpulist", cpus), ("numa_node", "0"), ("vendor", "0x10de")):
                with open(os.path.join(sysfs, addr, name), "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-b"}):
            with mock.patch.object(forksafe, "_nvml", return_value=nvml), \
                    mock.patch.object(forksafe.os.path, "isdir", return_value=False):
                self.assertEqual(forksafe.detect_gpu_cpu_affinity(0)["cpus"], [2, 3])
            with mock.patch.object(forksafe, "_nvml", return_value=None):
                self.assertEqual(forksafe.detect_gpu_cpu_affinity(0, sysfs_root=sysfs),
                                 {"cpus": [], "numa_node": None, "source": "none"})

    def test_bind_uses_only_cpus_this_process_may_run_on(self):
        """Input: a cpuset allowing CPUs {2, 3} (stand-in sched_getaffinity / sched_setaffinity
        that, like Linux under a Slurm cpuset, raise EINVAL for a set with no allowed CPU);
        the GPU's CPUs are [0, 1, 2], then [0, 1]. Expected: bound True on [2]; then bound
        False, no CPUs, the affinity unchanged ({2, 3}) and a reason. Pain point: Slurm's
        cpuset made sched_setaffinity fail, the error was swallowed and the reply still
        listed bound_cpus. Not: a real Slurm allocation (owed on a GPU node)."""
        allowed, current = {2, 3}, [{2, 3}]

        def setaffinity(pid, cpus):
            if not set(cpus) & allowed:
                raise OSError(22, "Invalid argument")
            current[0] = set(cpus) & allowed
        with mock.patch.object(forksafe.os, "sched_getaffinity", lambda pid: set(current[0]),
                               create=True), \
                mock.patch.object(forksafe.os, "sched_setaffinity", setaffinity, create=True):
            r = forksafe.bind_rank_numa_and_cpus(0, explicit_cpus=[0, 1, 2])
            self.assertEqual((r["bound"], r["bound_cpus"], current[0]), (True, [2], {2}))
            current[0] = {2, 3}
            r = forksafe.bind_rank_numa_and_cpus(0, explicit_cpus=[0, 1])
            self.assertEqual((r["bound"], r["bound_cpus"], current[0]), (False, [], {2, 3}))
            self.assertTrue(r["why"])


class NumaThreadsAndCowHygiene(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="forksafe-sysfs-")
        self.addCleanup(shutil.rmtree, self.temp_dir, True)

    def _write_pci_entry(self, pci_addr, cpulist_text, numa_node_text, vendor="0x10de",
                         device_class="0x030200"):
        pci_dir = os.path.join(self.temp_dir, pci_addr)
        os.makedirs(pci_dir, exist_ok=True)
        with open(os.path.join(pci_dir, "local_cpulist"), "w", encoding="utf-8") as handle:
            handle.write(cpulist_text + "\n")
        with open(os.path.join(pci_dir, "numa_node"), "w", encoding="utf-8") as handle:
            handle.write(str(numa_node_text) + "\n")
        if vendor is not None:
            with open(os.path.join(pci_dir, "vendor"), "w", encoding="utf-8") as handle:
                handle.write(vendor + "\n")
        if device_class is not None:
            with open(os.path.join(pci_dir, "class"), "w", encoding="utf-8") as handle:
                handle.write(device_class + "\n")
        return pci_dir

    def test_parse_cpulist_expands_ranges_and_singletons(self):
        """Input: Linux sysfs cpulist strings (`"0-3,8,10-11"`, `""`, `"4"`). Expected: sorted
        integer CPU lists (`[0, 1, 2, 3, 8, 10, 11]`, `[]`, `[4]`). Proves: sysfs `local_cpulist`
        ranges are parsed accurately without external tools."""
        self.assertEqual(forksafe.parse_cpulist("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(forksafe.parse_cpulist(""), [])
        self.assertEqual(forksafe.parse_cpulist("4"), [4])

    def test_detect_gpu_cpu_affinity_reads_fake_sysfs_without_cuda(self):
        """Input: a fake sysfs root with a non-GPU PCI device (`0000:00:01.0`, vendor `0x8086`)
        and two NVIDIA GPUs (`0000:01:00.0` on NUMA 0 with CPUs `0-3,8`, `0000:02:00.0` on NUMA 1
        with CPUs `4-7`). Expected: device 0 resolves to `[0, 1, 2, 3, 8]` on NUMA 0 and device 1
        to `[4, 5, 6, 7]` on NUMA 1, honoring `CUDA_VISIBLE_DEVICES=1`. Proves: sysfs topology
        discovery maps logical rank to physical GPU NUMA node without touching CUDA."""
        self._write_pci_entry("0000:00:01.0", "0-15", 0, vendor="0x8086", device_class="0x060400")
        self._write_pci_entry("0000:01:00.0", "0-3,8", 0)
        self._write_pci_entry("0000:02:00.0", "4-7", 1)
        clean_env = {key: val for key, val in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            aff0 = forksafe.detect_gpu_cpu_affinity(0, sysfs_root=self.temp_dir)
            aff1 = forksafe.detect_gpu_cpu_affinity(1, sysfs_root=self.temp_dir)
            self.assertEqual(aff0, {"cpus": [0, 1, 2, 3, 8], "numa_node": 0, "source": "sysfs"})
            self.assertEqual(aff1, {"cpus": [4, 5, 6, 7], "numa_node": 1, "source": "sysfs"})
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            remapped = forksafe.detect_gpu_cpu_affinity(0, sysfs_root=self.temp_dir)
            self.assertEqual(remapped, {"cpus": [4, 5, 6, 7], "numa_node": 1, "source": "sysfs"})

    def test_bind_rank_numa_and_cpus_sets_process_affinity(self):
        """Input: `bind_rank_numa_and_cpus` called with `explicit_cpus` and with a fake sysfs root
        pointing to an online CPU of the current machine. Expected: `os.sched_setaffinity` binds
        the process to that CPU and returns `bound` True, `bound_cpus`, `numa_node`, `source` and
        `why` None (where the platform has no sched_setaffinity: bound False and the reason)."""
        if hasattr(os, "sched_getaffinity"):
            original_affinity = os.sched_getaffinity(0)
            self.addCleanup(os.sched_setaffinity, 0, original_affinity)
            first_online_cpu = sorted(original_affinity)[0]
        else:
            first_online_cpu = 0
        self._write_pci_entry("0000:01:00.0", str(first_online_cpu), 0)
        clean_env = {key: val for key, val in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            can_bind = hasattr(os, "sched_setaffinity")
            done = {"bound": True, "bound_cpus": [first_online_cpu], "why": None} if can_bind \
                else {"bound": False, "bound_cpus": [], "why": "this platform has no sched_setaffinity"}
            bound = forksafe.bind_rank_numa_and_cpus(0, sysfs_root=self.temp_dir)
            self.assertEqual(bound, {**done, "numa_node": 0, "source": "sysfs"})
            if hasattr(os, "sched_getaffinity"):
                self.assertEqual(os.sched_getaffinity(0), {first_online_cpu})
            explicit = forksafe.bind_rank_numa_and_cpus(
                0, explicit_cpus=[first_online_cpu], sysfs_root=self.temp_dir
            )
            self.assertEqual(explicit, {**done, "numa_node": None, "source": "explicit"})

    def test_clamp_thread_pools_updates_env_and_loaded_torch(self):
        """Input: `clamp_thread_pools(num_threads=2, clamp_compile_threads=True)` with a stand-in
        `torch` module in `sys.modules` exposing `set_num_threads`. Expected: `OMP_NUM_THREADS`,
        `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `TORCHINDUCTOR_COMPILE_THREADS` are set to
        `"2"`, and `torch.set_num_threads(2)` is called."""
        recorded_threads = []
        fake_torch = types.SimpleNamespace(set_num_threads=recorded_threads.append)
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.dict(
            sys.modules, {"torch": fake_torch}
        ):
            summary = forksafe.clamp_thread_pools(num_threads=2, clamp_compile_threads=True)
            self.assertEqual(summary["num_threads"], 2)
            self.assertTrue(summary["torch_threads_set"])
            self.assertEqual(recorded_threads, [2])
            for env_var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS",
            ):
                self.assertEqual(os.environ.get(env_var), "2")

    def test_freeze_gc_before_fork_collects_and_freezes_heap(self):
        """Input: cyclic garbage on the heap before fork. Action: `freeze_gc_before_fork()`.
        Expected: full collection runs and `gc.get_freeze_count() > 0`."""
        self.addCleanup(gc.unfreeze)
        cycle_a, cycle_b = [], []
        cycle_a.append(cycle_b)
        cycle_b.append(cycle_a)
        del cycle_a, cycle_b
        result = forksafe.freeze_gc_before_fork()
        self.assertGreaterEqual(result["collected"], 2)
        self.assertGreater(result["frozen_count"], 0)

    def test_audit_system_environment_reports_all_four_hazards_and_clean_pass(self):
        """Input: a fake sysfs where GPU 0 is on CPUs `0-3`, THP file set to `[always] madvise`,
        `OMP_NUM_THREADS=64` across `local_world_size=8`, `bound_cpus=[4, 5]`, `gc_frozen=False`,
        and `require_imex=True` with a missing IMEX path. Expected: `audit_system_environment`
        returns findings for `numa_mismatch`, `thread_oversubscription`, `thp_unfrozen_gc`, and
        `imex_channel_missing`; when aligned, clamped, frozen, and IMEX present, it returns `[]`."""
        self._write_pci_entry("0000:01:00.0", "0-3", 0)
        thp_file = os.path.join(self.temp_dir, "thp_enabled")
        with open(thp_file, "w", encoding="utf-8") as handle:
            handle.write("[always] madvise never\n")
        missing_imex = os.path.join(self.temp_dir, "no-imex")
        clean_env = {key: val for key, val in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
        clean_env["OMP_NUM_THREADS"] = "64"
        with mock.patch.dict(os.environ, clean_env, clear=True):
            findings_list = forksafe.audit_system_environment(
                local_rank=0,
                local_world_size=8,
                bound_cpus=[4, 5],
                gc_frozen=False,
                sysfs_root=self.temp_dir,
                thp_enabled_path=thp_file,
                imex_path=missing_imex,
                require_imex=True,
            )
            self.assertEqual(
                [item["check"] for item in findings_list],
                [
                    "numa_mismatch",
                    "thread_oversubscription",
                    "thp_unfrozen_gc",
                    "imex_channel_missing",
                ],
            )
            os.environ["OMP_NUM_THREADS"] = "1"
            present_imex = os.path.join(self.temp_dir, "imex-ok")
            os.makedirs(present_imex, exist_ok=True)
            clean_findings = forksafe.audit_system_environment(
                local_rank=0,
                local_world_size=1,
                bound_cpus=[0, 1],
                gc_frozen=True,
                sysfs_root=self.temp_dir,
                thp_enabled_path=thp_file,
                imex_path=present_imex,
                require_imex=True,
            )
            self.assertEqual(clean_findings, [])


if __name__ == "__main__":
    unittest.main()
