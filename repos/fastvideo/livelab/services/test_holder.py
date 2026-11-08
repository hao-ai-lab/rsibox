"""Golden tests of the weights holder's protocol and its stale refusal (CPU tensors, real
processes over a unix socket; the GPU path is the same code with CUDA IPC handles).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v
"""
import importlib
import os
import sys
import tempfile
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
        class and bit-equal weights, in another tensor (the holder owns a copy). Proves: a
        respawned worker gets the loaded weights without a load."""
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


if __name__ == "__main__":
    unittest.main()
