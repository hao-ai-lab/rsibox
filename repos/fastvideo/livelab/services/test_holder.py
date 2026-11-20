"""Golden tests of the weights holder's protocol and its stale refusal (CPU tensors, real
processes over a unix socket; the GPU path is the same code with CUDA IPC handles).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v
"""
import importlib
import os
import sys
import tempfile
import time
import unittest

try:
    import torch
except ImportError:                 # CI without torch: these cases skip
    torch = None

from services import holder as H

MODEL = '''import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)
'''


@unittest.skipIf(torch is None, "needs torch")
class HolderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="holder-")
        os.makedirs(os.path.join(self.dir, "hpkg"))
        for n, t in (("__init__.py", ""), ("net.py", MODEL)):
            with open(os.path.join(self.dir, "hpkg", n), "w") as f:
                f.write(t)
        sys.path.insert(0, self.dir)
        os.environ["PYTHONPATH"] = os.pathsep.join([self.dir, os.environ.get("PYTHONPATH", "")])
        importlib.invalidate_caches()
        import hpkg.net
        torch.manual_seed(0)
        self.net = hpkg.net.Net()
        self.sock = os.path.join(self.dir, "h.sock")
        self.proc = H.start(self.sock)

    def tearDown(self):
        try:
            H.call(self.sock, {"op": "stop"}, timeout=10)
        except Exception:
            pass
        self.proc.join(10)
        for n in [n for n in sys.modules if n.split(".")[0] == "hpkg"]:
            del sys.modules[n]
        sys.path.remove(self.dir)

    def deposit(self):
        files = H.stamp_files(self.net, prefixes=("hpkg.loader",))
        self.assertIn("hpkg.net", files)
        r = H.call(self.sock, {"op": "put", "key": "rank0:transformer", "module": self.net,
                                "files": files, "stamp": H.stamp(files)})
        self.assertTrue(r["ok"], r)
        return files

    def test_served_module_equals_the_deposit(self):
        """Input: a module deposited, then asked for with an unchanged stamp. Expect: the same
        class and bit-equal weights, in another tensor (the holder owns a copy). Proves: the
        holder's put/get round trip returns the deposited module as a copy. Not: the loader
        hook (covered below) or a respawned GPU worker (covered by the zygote-with-holder
        receipt)."""
        files = self.deposit()
        r = H.call(self.sock, {"op": "get", "key": "rank0:transformer", "stamp": H.stamp(files)})
        self.assertTrue(r["ok"], r)
        m = r["module"]
        self.assertEqual(type(m).__name__, "Net")
        self.assertTrue(torch.equal(m.proj.weight, self.net.proj.weight))
        self.assertNotEqual(m.proj.weight.data_ptr(), self.net.proj.weight.data_ptr())

    def test_an_edit_of_a_stamped_file_is_refused_and_never_served_again(self):
        """Input: deposit, then the model's source file changes (as an edit of a constructor or
        a weight layout would), then two gets. Expect: `holder_stale` naming the file, and the
        entry is gone (`holder_empty` next). Proves: the holder never serves weights loaded by
        code that no longer exists. Not: an edit in an unstamped module."""
        files = self.deposit()
        with open(os.path.join(self.dir, "hpkg", "net.py"), "a") as f:
            f.write("\n# an edit\n")
        r = H.call(self.sock, {"op": "get", "key": "rank0:transformer", "stamp": H.stamp(files)})
        self.assertFalse(r["ok"])
        self.assertEqual(r["findings"][0]["check"], "holder_stale")
        self.assertIn("files:hpkg.net", r["findings"][0]["observed"])
        r2 = H.call(self.sock, {"op": "get", "key": "rank0:transformer", "stamp": H.stamp(files)})
        self.assertEqual(r2["findings"][0]["check"], "holder_empty")

    def test_a_stalled_client_does_not_block_the_next_one(self):
        """Input: one client connects and never sends a request; then another asks for status.
        Expect: the status reply arrives (after at most the holder's request timeout). Proves:
        one wedged rank cannot freeze the holder for every other rank."""
        from multiprocessing.connection import Client
        stalled = Client(self.sock, family="AF_UNIX")
        try:
            t0 = time.perf_counter()
            r = H.call(self.sock, {"op": "status"}, timeout=H.REQUEST_TIMEOUT_S + 10)
            self.assertTrue(r["ok"], r)
            self.assertLess(time.perf_counter() - t0, H.REQUEST_TIMEOUT_S + 10)
        finally:
            stalled.close()

    def test_loader_hook_deposits_then_serves_and_reloads_when_stale(self):
        """Input: `install_loader_hook` wraps `PipelineComponentLoader.load_module` (each raw
        disk load increments a counter and seeds new weights); load `transformer` twice, then
        edit `hpkg/net.py` and load a third time, plus load an unhooked `scheduler`. Expect:
        load 1 runs the disk loader (count 1) and deposits; load 2 is served by the holder
        without calling the disk loader (count stays 1, weights bit-equal to load 1); after the
        source edit, load 3 gets `holder_stale`, runs the disk loader (count 2, new weights)
        and re-deposits; `scheduler` bypasses the holder (count 3). Proves: `install_loader_hook`
        redirects component loads through the holder and falls back to disk on a miss or stale
        stamp. Not: CUDA IPC tensor sharing (covered by the GPU receipt)."""
        loader_dir = os.path.join(self.dir, "fastvideo", "models", "loader")
        os.makedirs(loader_dir)
        for rel in (("fastvideo", "__init__.py"),
                    ("fastvideo", "models", "__init__.py"),
                    ("fastvideo", "models", "loader", "__init__.py")):
            with open(os.path.join(self.dir, *rel), "w") as f:
                f.write("")
        with open(os.path.join(loader_dir, "component_loader.py"), "w") as f:
            f.write(
                "import torch\n"
                "from hpkg.net import Net\n\n"
                "class PipelineComponentLoader:\n"
                "    disk_loads = 0\n\n"
                "    @staticmethod\n"
                "    def load_module(module_name, component_model_path, transformers_or_diffusers, fastvideo_args):\n"
                "        PipelineComponentLoader.disk_loads += 1\n"
                "        torch.manual_seed(100 + PipelineComponentLoader.disk_loads)\n"
                "        return Net()\n"
            )
        importlib.invalidate_caches()
        from fastvideo.models.loader import component_loader
        try:
            report = H.install_loader_hook(self.sock)
            first_model = component_loader.PipelineComponentLoader.load_module("transformer", self.dir, "diffusers", None)
            second_model = component_loader.PipelineComponentLoader.load_module("transformer", self.dir, "diffusers", None)
            self.assertEqual(component_loader.PipelineComponentLoader.disk_loads, 1)
            self.assertTrue(torch.equal(second_model.proj.weight, first_model.proj.weight))
            self.assertEqual([entry["served"] for entry in report], [False, True])
            self.assertTrue(report[0]["deposited"])

            with open(os.path.join(self.dir, "hpkg", "net.py"), "a") as handle:
                handle.write("\n# an edit\n")
            third_model = component_loader.PipelineComponentLoader.load_module("transformer", self.dir, "diffusers", None)
            self.assertEqual(component_loader.PipelineComponentLoader.disk_loads, 2)
            self.assertFalse(report[-1]["served"])
            self.assertEqual(report[-1]["findings"][0]["check"], "holder_stale")
            self.assertFalse(torch.equal(third_model.proj.weight, first_model.proj.weight))

            component_loader.PipelineComponentLoader.load_module("scheduler", self.dir, "diffusers", None)
            self.assertEqual(component_loader.PipelineComponentLoader.disk_loads, 3)
            self.assertEqual(len(report), 3)
        finally:
            for module_name in [name for name in sys.modules if name.split(".")[0] == "fastvideo"]:
                del sys.modules[module_name]

    def test_shared_cpu_storages_survive_depositor_shm_manager_kill_and_keep_tied_views(self):
        """Input: a separate depositor process using `file_system` sharing deposits a module with
        shared CPU tensors (tied views with distinct offsets/strides + nested dict/list tensors),
        then its `torch_shm_manager` descendant is killed before a separate getter fetches it.
        Expect: `_own` clones shared CPU storages into holder-private memory (`not is_shared()`),
        `get` succeeds with zero `Broken pipe`, bit-exact values, and preserved view aliasing."""
        import json
        import signal
        import subprocess
        from fastrsi import proc as proc_util

        mod_path = os.path.join(self.dir, "hpkg", "shared_mod.py")
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(
                "import torch\n"
                "from torch import nn\n\n"
                "class TiedCpuNet(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        base = torch.arange(16, dtype=torch.float32)\n"
                "        self.empty_p = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)\n"
                "        self.register_buffer('view_a', base[2:10].view(2, 4))\n"
                "        self.register_buffer('view_b', base[6:14].view(4, 2))\n"
                "        self.tables = {'nested': [torch.linspace(0.0, 1.0, 5)]}\n"
            )
        importlib.invalidate_caches()
        import hpkg.shared_mod as sm
        local = sm.TiedCpuNet()
        local.view_a.share_memory_()
        local.tables["nested"][0].share_memory_()
        self.assertTrue(local.view_a.untyped_storage().is_shared())
        self.assertTrue(local.view_b.untyped_storage().is_shared())
        owned = H._own(local)
        self.assertFalse(owned.view_a.untyped_storage().is_shared())
        self.assertFalse(owned.view_b.untyped_storage().is_shared())
        self.assertFalse(owned.tables["nested"][0].untyped_storage().is_shared())
        self.assertEqual(owned.view_a.untyped_storage().data_ptr(),
                         owned.view_b.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(owned.view_a, local.view_a))
        self.assertTrue(torch.equal(owned.view_b, local.view_b))

        livelab = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_root = os.path.abspath(os.path.join(livelab, "..", "..", "..", "core", "livelab"))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([self.dir, livelab, core_root])}

        depositor_code = (
            "import json, os, sys, time, torch\n"
            "import torch.multiprocessing as tmp\n"
            "tmp.set_sharing_strategy('file_system')\n"
            "from services import holder as H\n"
            "import hpkg.shared_mod as sm\n"
            "m = sm.TiedCpuNet()\n"
            "files = H.stamp_files(m, prefixes=())\n"
            "r = H.call(sys.argv[1], {'op': 'put', 'key': 'rank0:tied', 'module': m,\n"
            "           'files': files, 'stamp': H.stamp(files)})\n"
            "assert r.get('ok'), r\n"
            "print(json.dumps({'pid': os.getpid(), 'files': files}), flush=True)\n"
            "sys.stdin.readline()\n"
        )
        dep = subprocess.Popen(
            [sys.executable, "-c", depositor_code, self.sock],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            line = dep.stdout.readline()
            if not line:
                dep.kill()
                dep.wait(timeout=5)
                self.fail(f"depositor exited early:\n{dep.stderr.read()}")
            info = json.loads(line)
            # Simulate dirty worker teardown killing depositor AND its torch_shm_manager child.
            proc_util.kill_tree(dep.pid, signal.SIGKILL, include_pg=False)
            dep.wait(timeout=10)
        finally:
            for stream in (dep.stdin, dep.stdout, dep.stderr):
                if stream is not None:
                    stream.close()

        getter_code = (
            "import json, sys, torch\n"
            "import torch.multiprocessing as tmp\n"
            "tmp.set_sharing_strategy('file_system')\n"
            "from services import holder as H\n"
            "files = json.loads(sys.argv[2])\n"
            "r = H.call(sys.argv[1], {'op': 'get', 'key': 'rank0:tied', 'stamp': H.stamp(files)})\n"
            "assert r.get('ok'), r\n"
            "m = r['module']\n"
            "base = torch.arange(16, dtype=torch.float32)\n"
            "assert not m.view_a.untyped_storage().is_shared()\n"
            "assert not m.view_b.untyped_storage().is_shared()\n"
            "assert not m.empty_p.untyped_storage().is_shared()\n"
            "assert m.empty_p.shape == (0,)\n"
            "assert torch.equal(m.view_a, base[2:10].view(2, 4))\n"
            "assert torch.equal(m.view_b, base[6:14].view(4, 2))\n"
            "assert torch.equal(m.tables['nested'][0], torch.linspace(0.0, 1.0, 5))\n"
            "assert m.view_a.untyped_storage().data_ptr() == m.view_b.untyped_storage().data_ptr()\n"
            "m.view_a[1, 0] = 999.0\n"
            "assert float(m.view_b[0, 0]) == 999.0\n"
            "print('OK')\n"
        )
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-Wdefault", "-Werror::ResourceWarning", "-Werror::UserWarning",
                 "-c", getter_code, self.sock, json.dumps(info["files"])],
                env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertEqual(out.stdout.strip(), "OK")


class OwnCpuStorageUnitTest(unittest.TestCase):
    def test_own_clones_shared_cpu_storages_preserves_tied_views_and_breaks_cycles(self):
        """Pure-Python unit test of `holder._own` covering memo pre-population, tied CPU views,
        nested dict/list/__slots__ cycles, custom __deepcopy__ fallback, and untouched CUDA tensors."""
        import contextlib
        import copy
        import types
        from unittest import mock

        class FakeStorage:
            def __init__(self, data, shared=False, cdata=None):
                self.buf = bytearray(data) if isinstance(data, (bytes, bytearray)) else bytearray(data)
                self._shared = shared
                self._cdata = id(self) if cdata is None else cdata

            def nbytes(self):
                return len(self.buf)

            def is_shared(self):
                return self._shared

            def copy_(self, other):
                self.buf[:] = other.buf
                return self

            def __deepcopy__(self, memo):
                # Emulate PyTorch keeping shared storages untouched if memo missed.
                t = memo.setdefault("torch", {})
                if self._cdata in t:
                    return t[self._cdata]
                if self._shared:
                    return self
                out = FakeStorage(self.buf, shared=False)
                t[self._cdata] = out
                return out

        class FakeTensor:
            def __init__(self, storage, offset=0, size=(4,), stride=(1,), device="cpu"):
                self._storage = storage
                self._offset, self._size, self._stride = offset, size, stride
                self.device = types.SimpleNamespace(type=device)

            def untyped_storage(self):
                # Return a fresh wrapper sharing _cdata, like PyTorch's untyped_storage().
                return self._storage

            @property
            def data(self):
                return self

            def storage_offset(self):
                return self._offset

            def size(self):
                return self._size

            def stride(self):
                return self._stride

            def set_(self, storage, offset, size, stride):
                self._storage, self._offset, self._size, self._stride = storage, offset, size, stride
                return self

            def __deepcopy__(self, memo):
                if id(self) in memo:
                    return memo[id(self)]
                ns = copy.deepcopy(self._storage, memo)
                out = FakeTensor(ns, self._offset, self._size, self._stride, self.device.type)
                memo[id(self)] = out
                return out

        class BypassTensor(FakeTensor):
            def __deepcopy__(self, memo):
                # Simulates a custom subclass that ignores memo and keeps the shared storage.
                return BypassTensor(self._storage, self._offset, self._size, self._stride, self.device.type)

        class SlottedBox:
            __slots__ = ("tensor", "loop")

            def __init__(self, tensor):
                self.tensor = tensor
                self.loop = self

        shared_cpu = FakeStorage(b"\x01\x02\x03\x04", shared=True, cdata=777)
        bypass_cpu = FakeStorage(b"\x05\x06\x07\x08", shared=True, cdata=888)
        cuda_st = FakeStorage(b"\x09\x0a", shared=True, cdata=999)

        root = types.SimpleNamespace(
            view_a=FakeTensor(shared_cpu, offset=0),
            view_b=FakeTensor(shared_cpu, offset=2),
            nested={"box": [SlottedBox(BypassTensor(bypass_cpu))], "cuda": FakeTensor(cuda_st, device="cuda")},
        )
        root.cycle = root

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.UntypedStorage = lambda n: FakeStorage(n, shared=False)
        fake_torch.no_grad = contextlib.nullcontext
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            owned = H._own(root)

        self.assertIsNot(owned.view_a._storage, shared_cpu)
        self.assertFalse(owned.view_a._storage.is_shared())
        self.assertIs(owned.view_a._storage, owned.view_b._storage)
        self.assertEqual(bytes(owned.view_a._storage.buf), b"\x01\x02\x03\x04")
        self.assertEqual((owned.view_a.storage_offset(), owned.view_b.storage_offset()), (0, 2))
        bypass_owned = owned.nested["box"][0].tensor
        self.assertIsNot(bypass_owned._storage, bypass_cpu)
        self.assertFalse(bypass_owned._storage.is_shared())
        self.assertEqual(bytes(bypass_owned._storage.buf), b"\x05\x06\x07\x08")
        # CUDA storage stays untouched by the CPU-only detachment path
        self.assertIs(owned.nested["cuda"]._storage, cuda_st)
        self.assertIs(owned.cycle, owned)


class StampFilesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="stamp-")
        os.makedirs(os.path.join(self.dir, "spkg", "vaes"))
        files = {"__init__.py": "", "vaes/__init__.py": "",
                 "vaes/vae.py": "class VAE:\n    pass\n",
                 "vaes/decoder_setup.py": "def apply(vae):\n    vae.decode_dtype = 'bf16'\n",
                 "other.py": "X = 1\n"}
        for n, t in files.items():
            with open(os.path.join(self.dir, "spkg", n), "w") as f:
                f.write(t)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()

    def tearDown(self):
        for n in [n for n in sys.modules if n.split(".")[0] == "spkg"]:
            del sys.modules[n]
        sys.path.remove(self.dir)

    def test_a_setup_helper_next_to_the_model_class_is_stamped(self):
        """Input: a VAE whose class lives in spkg.vaes.vae, set up at load by a function of
        spkg.vaes.decoder_setup (no class of it in the VAE: FastH3's NVFP4 decoder setup);
        spkg.other is loaded too. Expect: stamp_files names both vaes modules and not
        spkg.other; with spkg not among the engine's own packages, only the class module. Proves:
        an edit of a load-time helper beside the model class makes the holder entry stale, and
        third-party siblings are not stamped. Not: a helper in another package (not stamped)."""
        import spkg.other  # noqa: F401
        import spkg.vaes.decoder_setup as setup
        import spkg.vaes.vae as vae_module
        vae = vae_module.VAE()
        setup.apply(vae)
        files = H.stamp_files(vae, prefixes=("spkg.loader",), own=("spkg",))
        self.assertEqual(sorted(files), ["spkg.vaes.decoder_setup", "spkg.vaes.vae"])
        # a package not the engine's own (diffusers, transformers): only the class module
        self.assertEqual(sorted(H.stamp_files(vae, prefixes=("spkg.loader",), own=())), ["spkg.vaes.vae"])


EXIT_DRIVER = """
import multiprocessing as mp, os, subprocess, sys
from services import holder


def child(fix):
    if fix:
        holder._private_tracker_fd()
    # what torch's file_system sharing does: fork and exec a daemon that inherits open fds
    p = subprocess.Popen(["sleep", "20"], close_fds=False, start_new_session=True)
    print(p.pid, flush=True)
    p.returncode = 0  # intentionally detached; parent test kills it in finally


if __name__ == "__main__":
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=child, args=(sys.argv[1] == "fix",))
    proc.start()
    proc.join()
"""


class HostExitTest(unittest.TestCase):
    def run_host(self, mode):
        import subprocess
        livelab = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "PYTHONPATH": livelab}
        with tempfile.TemporaryDirectory(prefix="holder-exit-") as tmp:
            script = os.path.join(tmp, "host.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(EXIT_DRIVER)                 # spawn re-imports the main module
            t0 = time.time()
            p = subprocess.Popen([sys.executable, script, mode], env=env,
                                 stdout=subprocess.PIPE, text=True)
            daemon = None
            try:
                line = p.stdout.readline()
                daemon = int(line) if line.strip() else None
                try:
                    p.wait(timeout=10)
                    return round(time.time() - t0, 1)
                except subprocess.TimeoutExpired:
                    return None
            finally:
                if daemon is not None:
                    try:
                        os.kill(daemon, 9)
                    except ProcessLookupError:
                        pass
                if p.stdout is not None:
                    p.stdout.close()
                p.wait()

    def test_a_daemon_a_spawned_child_executes_does_not_hang_the_host_at_exit(self):
        """Input: a host that runs multiprocessing's resource tracker spawns a child (as the lab
        spawns the holder); the child executes a 20 s daemon that keeps inherited fds open (as
        torch's file_system sharing starts torch_shm_manager) and returns. Expect: with
        `_private_tracker_fd` the host exits at once; without it, on Linux with Python 3.12 (CI,
        the GPU nodes' environment), it is still waiting after 10 s: the tracker's finalizer
        waits for the tracker, which waits for its pipe to close (Python 3.14 on macOS does not
        wait). Proves: the holder no longer hangs the lab host at exit. Not: torch itself (the
        GPU run with the holder exits)."""
        self.assertIsNotNone(self.run_host("fix"))
        if sys.platform.startswith("linux") and sys.version_info[:2] == (3, 12):
            self.assertIsNone(self.run_host("nofix"))


if __name__ == "__main__":
    unittest.main()
