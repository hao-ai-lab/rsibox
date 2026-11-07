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
        """Input: no edit. Expect: the child drops nothing and keeps the preloaded modules."""
        out = fork_and_read(self.child)
        self.assertEqual((out["stale"], out["dropped"], out["value"]), ([], [], 1))


if __name__ == "__main__":
    unittest.main()


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
        with PYTORCH_NVML_BASED_CUDA_CHECK=1 (what `enable` sets) it is not. Proves: the zygote
        server can import FastVideo without poisoning its workers. Not: other CUDA calls."""
        self.assertTrue(_child_poisoned({"PYTORCH_NVML_BASED_CUDA_CHECK": "0"}))
        self.assertFalse(_child_poisoned({"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}))
