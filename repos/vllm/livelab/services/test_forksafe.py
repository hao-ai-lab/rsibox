"""Golden cases for forksafe.py. CPU only: a stand-in torch.cuda whose original functions
initialize "CUDA" the way the real ones do, and a stand-in NVML with fixed capabilities.

The pain point: at TP>1 the API server imports FlashInfer, which asks for the GPU's capability
at import; the real torch.cuda.get_device_capability initializes CUDA, and vLLM then spawns
(instead of forks) the engine core and every worker, each importing vLLM again (33 s and
121 s in two of our TP2 starts).

Run: python3 -m pytest repos/vllm/livelab/services/test_forksafe.py -v
"""
import os, subprocess, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import forksafe  # noqa: E402


def fake_cuda():
    """torch.cuda as the API server sees it: CUDA not initialized until a real query runs."""
    c = types.SimpleNamespace(initialized=False)

    def get_device_capability(device=None):
        c.initialized = True          # the real one creates a CUDA context
        return (8, 0)

    def current_device():
        c.initialized = True
        return 0
    c.is_initialized = lambda: c.initialized
    c.get_device_capability, c.current_device = get_device_capability, current_device
    return c


def fake_nvml(caps):
    """NVML with one capability per physical GPU index."""
    return types.SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(caps),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetCudaComputeCapability=lambda h: caps[h])


def test_capability_comes_from_nvml_and_cuda_stays_uninitialized():
    # Input: GPUs 2,3 visible, both 10.0 (GB200). Action: FlashInfer's import-time queries.
    # Expected: 10.0 for each form of device, CUDA never initialized. Without install() the
    # first call initializes CUDA (and the stand-in answers 8.0): vLLM would then spawn.
    nvml = fake_nvml([(9, 0), (9, 0), (10, 0), (10, 0)])
    assert forksafe.nvml_capability("2,3", nvml) == (10, 0)
    cuda = fake_cuda()
    assert forksafe.install(cuda, capability=lambda: forksafe.nvml_capability("2,3", nvml)) \
        == {"installed": True}
    dev = types.SimpleNamespace(type="cuda", index=1)
    for d in (None, 0, 1, "cuda", "cuda:1", dev):
        assert cuda.get_device_capability(d) == (10, 0)
    assert cuda.current_device() == 0
    assert cuda.is_initialized() is False
    # Proves the answers match NVML for the visible GPUs; not that CUDA would say the same
    # (the GPU receipt shows the forked TP2 server serving the same greedy output).
    plain = fake_cuda()
    plain.get_device_capability(0)
    assert plain.is_initialized() is True


def test_no_guess_when_visible_gpus_differ_or_device_is_not_cuda():
    # Input: GPUs 0,2 visible with 9.0 and 10.0. Expected: no single answer exists, so the
    # original runs (CUDA initializes, as without forksafe) and the fallback is counted.
    nvml = fake_nvml([(9, 0), (9, 0), (10, 0), (10, 0)])
    assert forksafe.nvml_capability("0,2", nvml) is None
    assert forksafe.nvml_capability("GPU-uuid-1", nvml) is None     # UUIDs: left to CUDA
    cuda = fake_cuda()
    forksafe.install(cuda, capability=lambda: forksafe.nvml_capability("0,2", nvml))
    before = forksafe.COUNTS["fallback"]
    assert cuda.get_device_capability(0) == (8, 0)
    assert cuda.is_initialized() is True
    assert forksafe.COUNTS["fallback"] == before + 1
    # A non-cuda device is never answered from NVML either.
    cuda2 = fake_cuda()
    forksafe.install(cuda2, capability=lambda: (10, 0))
    assert cuda2.get_device_capability("cpu") == (8, 0)


def test_after_cuda_initializes_the_originals_answer():
    # In a worker CUDA is initialized (set_device): the real functions must run there.
    cuda = fake_cuda()
    forksafe.install(cuda, capability=lambda: (10, 0))
    cuda.initialized = True
    assert cuda.get_device_capability(0) == (8, 0)
    assert forksafe.install(cuda)["installed"] is False                 # once only


def test_real_torch_module_is_patched_without_touching_the_gpu():
    # Input: the real torch, the capability given (no NVML, no GPU touched). Expected: after
    # install, torch.cuda.get_device_capability and a name imported from torch.cuda later (as
    # FlashInfer does) answer 10.0 and torch.cuda.is_initialized() stays False.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import torch, forksafe\n"
        "print(forksafe.install(torch.cuda, capability=lambda: (10, 0)))\n"
        "from torch.cuda import get_device_capability\n"
        "assert torch.cuda.get_device_capability(0) == (10, 0)\n"
        "assert get_device_capability() == (10, 0) and torch.cuda.current_device() == 0\n"
        "assert not torch.cuda.is_initialized()\n"
        "print('ok')\n" % str(HERE))
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            import pytest
            pytest.skip("torch not installed")
    except ImportError:
        return
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok")
