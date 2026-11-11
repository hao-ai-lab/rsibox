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


if __name__ == "__main__":
    unittest.main()
