"""Golden tests of the FastVideo worker zygote's refresh (CPU, a real fork).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v
"""
import importlib
import json
import os
import sys
import tempfile
import time
import unittest

from services import zygote as Z


def fork_and_read(fn):
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


class ZygoteRefreshTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="zygote-")
        os.makedirs(os.path.join(self.dir, "zpkg"))
        self.files = {"__init__.py": "", "consts.py": "VALUE = 1\n",
                      "user.py": "from zpkg.consts import VALUE\n\n\ndef get():\n    return VALUE\n",
                      "other.py": "def get():\n    return 'other'\n"}
        for n, t in self.files.items():
            with open(os.path.join(self.dir, "zpkg", n), "w") as f:
                f.write(t)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()
        import zpkg.user, zpkg.other  # noqa: E401,F401  (the zygote's preload)
        Z.snapshot(prefixes=("zpkg",))

    def tearDown(self):
        for n in [n for n in sys.modules if n.split(".")[0] == "zpkg"]:
            del sys.modules[n]
        sys.path.remove(self.dir)

    def child(self):
        rep = Z.refresh_stale(prefixes=("zpkg",))
        import zpkg.user
        return {**rep, "value": zpkg.user.get()}

    def test_edit_after_preload_is_seen_by_the_forked_worker(self):
        """Input: the zygote imported consts (VALUE = 1) and user (which copied VALUE); then
        consts.py is edited to VALUE = 2. Expect: a forked child refreshes consts and user (not
        other) and returns 2; the zygote itself still has 1. Proves: a worker forked after an
        edit runs the new code, importers included. Not: a C extension edit (a finding)."""
        time.sleep(0.01)
        with open(os.path.join(self.dir, "zpkg", "consts.py"), "w") as f:
            f.write("VALUE = 2\n")
        out = fork_and_read(self.child)
        self.assertEqual(out["value"], 2, out)
        self.assertEqual(out["stale"], ["zpkg.consts"])
        self.assertEqual(out["dropped"], ["zpkg.consts", "zpkg.user"])
        self.assertEqual(sys.modules["zpkg.user"].get(), 1)

    def test_no_edit_drops_nothing(self):
        """Input: no edit. Expect: the child drops nothing and keeps the preloaded modules.
        Proves: a respawn without an edit pays no re-import (the zygote's point). Not: a GPU
        worker (the zygote respawn receipt covers that)."""
        out = fork_and_read(self.child)
        self.assertEqual((out["stale"], out["dropped"], out["value"]), ([], [], 1))


FORKSERVER_DRIVER = """
import importlib, json, multiprocessing as mp, os, sys, time
from services import zygote as Z

Z.enable(preload=["fastvideo.user"])
ctx = mp.get_context("forkserver")


def spawn_once():
    import fastvideo.consts as consts_module
    import fastvideo.user as user_module
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=user_module.work, args=(consts_module.Arg(7), writer))
    process.start()
    writer.close()
    result = reader.recv()
    reader.close()
    process.join(10)
    return result


first = spawn_once()
time.sleep(0.02)
with open(sys.argv[1], "w") as handle:
    handle.write("VALUE = 2\\n\\nclass Arg:\\n    def __init__(self, x):\\n        self.x = x\\n")
importlib.invalidate_caches()
second = spawn_once()
print(json.dumps({"first": first, "second": second}))
"""


class ForkserverPrepareHookTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="zygote-fs-")
        package_dir = os.path.join(self.dir, "fastvideo")
        os.makedirs(package_dir)
        self.consts = os.path.join(package_dir, "consts.py")
        files = {
            "__init__.py": "",
            "consts.py": "VALUE = 1\n\nclass Arg:\n    def __init__(self, x):\n        self.x = x\n",
            "user.py": (
                "from fastvideo.consts import Arg, VALUE\n\n"
                "def work(arg, writer):\n"
                "    writer.send({'value': VALUE, 'is_arg': isinstance(arg, Arg), 'x': arg.x})\n"
                "    writer.close()\n"
            ),
        }
        for name, text in files.items():
            with open(os.path.join(package_dir, name), "w") as handle:
                handle.write(text)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_forkserver_prepare_hook_refreshes_child_before_unpickling_target(self):
        """Input: `zygote.enable` preloads `fastvideo.user` (VALUE = 1, class Arg) in a real
        multiprocessing forkserver; worker 1 runs; `consts.py` is edited on disk to VALUE = 2;
        worker 2 is forked from the same server with an `Arg(7)` argument and never calls
        `refresh_stale` itself. Expect: worker 1 sees VALUE = 1; worker 2's `spawn.prepare`
        hook refreshes `fastvideo.consts` and `fastvideo.user` before unpickling the process
        object, so worker 2 sees VALUE = 2 and `isinstance(arg, Arg)` is True. Proves: the
        forkserver `spawn.prepare` hook wired by `_serve` refreshes forked workers end to end."""
        import subprocess
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([self.dir, livelab_dir])}
        completed = subprocess.run([sys.executable, "-c", FORKSERVER_DRIVER, self.consts],
                                   capture_output=True, text=True, check=True, env=env, timeout=30)
        output = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(output["first"], {"value": 1, "is_arg": True, "x": 7})
        self.assertEqual(output["second"], {"value": 2, "is_arg": True, "x": 7}, completed.stderr)
        self.assertIn("[fastrsi-zygote] pid ", completed.stderr)
        self.assertIn("refreshed stale=['fastvideo.consts']", completed.stderr)


class ForkGuardTest(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.get(k) for k in ("CUTE_DSL_ARCH", "FASTRSI_ZYGOTE_CC")}

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cute_dsl_arch_is_pinned_before_the_preload(self):
        """Input: CUTE_DSL_ARCH unset, FASTRSI_ZYGOTE_CC=10.0 (what NVML answers on B200/GB200);
        then CUTE_DSL_ARCH preset to sm_90a. Expect: pin_cute_dsl_arch sets sm_100a, then keeps
        the preset value. Proves: the zygote gives CuTe DSL its arch, so importing flashinfer
        -> cutlass in the server does not call cuInit to find it. Not: that the DSL reads it
        (the GPU check of the zygote respawn covers that)."""
        os.environ.pop("CUTE_DSL_ARCH", None)
        os.environ["FASTRSI_ZYGOTE_CC"] = "10.0"
        self.assertEqual(Z.pin_cute_dsl_arch(), "sm_100a")
        self.assertEqual(os.environ["CUTE_DSL_ARCH"], "sm_100a")
        os.environ["CUTE_DSL_ARCH"] = "sm_90a"
        self.assertEqual(Z.pin_cute_dsl_arch(), "sm_90a")

    def test_a_poisoned_zygote_refuses_to_start_workers(self):
        """Input: the worker hook installed with a poison reason (CUDA initialized in the server).
        Expect: the forked child's `spawn.prepare` raises RuntimeError naming the reason, before
        any worker code runs. Proves: a zygote whose forks cannot use CUDA fails the worker
        start loudly instead of handing out workers that fail later. Not: the detection itself
        (driver_initialized needs libcuda: GPU check)."""
        import multiprocessing.spawn as SP
        orig = SP.prepare
        try:
            Z._install_worker_hook(poison=["CUDA driver initialized in the zygote"])
            with self.assertRaisesRegex(RuntimeError, "CUDA driver initialized"):
                SP.prepare({})
        finally:
            SP.prepare = orig

    def test_no_libcuda_means_no_driver_poison(self):
        """Input: this CPU host (no libcuda.so.1). Expect: poison_reasons() is empty. Proves:
        the driver probe never blocks a CPU zygote. Not: a GPU host (the probe there is
        cuDeviceGetCount, checked on the GPU node)."""
        if os.path.exists("/proc/driver/nvidia"):
            self.skipTest("a GPU host")
        self.assertEqual(Z.poison_reasons(), [])


class ServeTest(unittest.TestCase):
    """`_serve` in this process, as the forkserver runs it, driven into its failure branches."""

    def setUp(self):
        import multiprocessing.spawn as SP
        self.SP, self.prepare = SP, SP.prepare
        self.saved = (Z.LIBCUDA, Z.driver_initialized, dict(Z._SNAPSHOT))
        self.env = {k: os.environ.get(k) for k in ("FASTRSI_ZYGOTE_PRELOAD", "FASTRSI_ZYGOTE_GC_FREEZE",
                                                   "CUTE_DSL_ARCH", "FASTRSI_ZYGOTE_CC")}
        os.environ["FASTRSI_ZYGOTE_PRELOAD"] = "fastrsi_no_such_module"
        os.environ["FASTRSI_ZYGOTE_GC_FREEZE"] = "0"

    def tearDown(self):
        self.SP.prepare = self.prepare
        Z.LIBCUDA, Z.driver_initialized = self.saved[0], self.saved[1]
        Z._SNAPSHOT.clear()
        Z._SNAPSHOT.update(self.saved[2])
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_a_preload_that_raises_after_cuinit_still_refuses_workers(self):
        """Input: the preload list names a module that does not import (ImportError), and the
        driver reports itself initialized (the cuInit happened before the failure). Expect:
        `_serve` returns the ImportError and the driver poison, and a forked child's prepare
        raises. Proves: the poison check runs however the preload ends. Not: the real driver
        call (stubbed here: no libcuda on a CPU host)."""
        Z.driver_initialized = lambda: True
        rep = Z._serve()
        self.assertIn("ModuleNotFoundError", rep["preload_error"])
        self.assertEqual(rep["poison"][0], "CUDA driver initialized in the zygote (cuInit)")
        with self.assertRaisesRegex(RuntimeError, "cannot use CUDA"):
            self.SP.prepare({})

    def test_a_libcuda_that_cannot_answer_is_poison(self):
        """Input: LIBCUDA names a library that loads but has no cuDeviceGetCount (this host's
        libc). Expect: the reason says the driver state is unreadable, and a forked child's
        prepare raises. Proves: an unknown driver state fails closed, through the real
        `driver_initialized` ctypes path."""
        import ctypes.util
        Z.LIBCUDA = ctypes.util.find_library("c") or "libc.so.6"
        rep = Z._serve()
        self.assertTrue(rep["poison"][0].startswith("CUDA driver state unreadable (AttributeError"), rep)
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            self.SP.prepare({})


def _child_poisoned(env):
    import subprocess
    code = ("import sys; sys.path.insert(0, '.'); import torch; torch.cuda.is_available(); "
            "from services import zygote as Z; print(int(Z.fork_poisoned()))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         env={**os.environ, **env})
    return out.stdout.strip().splitlines()[-1] == "1"


def _have_cuda():
    try:
        import torch
        return torch.cuda.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_have_cuda(), "needs torch and a CUDA device")
class ForkPoisonTest(unittest.TestCase):
    def test_runtime_check_poisons_forks_nvml_check_does_not(self):
        """Input: a process calls torch.cuda.is_available() (as FastVideo does at import), then
        forks. Expect: through the CUDA runtime the child is "in bad fork" (cannot use CUDA);
        with PYTORCH_NVML_BASED_CUDA_CHECK=1 (the setting `enable` makes) it is not. Proves: the
        environment `enable` sets keeps a forked child able to use CUDA after the parent asked
        torch whether CUDA is available. Not: an import of FastVideo or `enable` itself (the GPU
        receipt of the zygote respawn covers those), or other CUDA calls."""
        self.assertTrue(_child_poisoned({"PYTORCH_NVML_BASED_CUDA_CHECK": "0"}))
        self.assertFalse(_child_poisoned({"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}))


if __name__ == "__main__":
    unittest.main()
