"""Golden tests of the path-stable JIT-cache pinning (CPU).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from services import jitcache as J

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:                 # CI without torch: the torch_key case skips
    HAVE_TORCH = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = ("import json, sys; from services import jitcache as J; r = J.preset_torch_key(sys.argv[1]); "
       "from torch._inductor import codecache; print(json.dumps({**r, 'key': codecache.torch_key().hex()}))")


def run_key(path, fresh=False):
    code = KEY if not fresh else ("import json; from torch._inductor import codecache; "
                                  "print(json.dumps({'key': codecache.torch_key().hex()}))")
    out = subprocess.run([sys.executable, "-c", code, path], cwd=HERE, capture_output=True,
                         text=True, check=True, env={**os.environ, "PYTHONPATH": HERE})
    return json.loads(out.stdout.strip().splitlines()[-1])


class JitCacheTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jitcache-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_moved_root_is_a_finding_before_the_rebuild(self):
        """Input: a cache root used once at A, then copied to B and used there. Expect: no
        finding at A (twice), finding `jit_root_moved` naming A at B. Proves: the copy that
        cost a 159 s FlashInfer rebuild on GPU is caught before the process starts."""
        a = os.path.join(self.dir, "a")
        self.assertEqual(J.check_root(a), [])
        self.assertEqual(J.check_root(a), [])
        b = os.path.join(self.dir, "b")
        shutil.copytree(a, b)
        f = J.check_root(b)
        self.assertEqual([x["check"] for x in f], ["jit_root_moved"])
        self.assertIn(os.path.abspath(a), f[0]["observed"])

    def test_staged_copy_keeps_path_and_mtimes_symlink_is_caught(self):
        """Input: a root used at A (with a file of a known mtime) is copied away, A removed,
        then (1) the copy reached through a symlink at A's place is checked, (2) the copy
        staged back at A. Expect: (1) a finding `jit_root_moved` (FlashInfer resolves the link
        and would rebuild: 155 s on GPU), (2) no finding and the file's mtime kept to the
        nanosecond. Proves: the staged tree keeps what ninja compares. GPU receipt in kb."""
        a = os.path.join(self.dir, "a")
        self.assertEqual(J.check_root(a), [])
        f = os.path.join(a, "x.o")
        open(f, "w").close()
        os.utime(f, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
        away = os.path.join(self.dir, "away")
        shutil.copytree(a, away, copy_function=shutil.copy2)
        shutil.rmtree(a)
        os.symlink(away, a)
        self.assertEqual([x["check"] for x in J.check_root(a)], ["jit_root_moved"])
        os.unlink(a)
        J.stage_root(away, a)
        self.assertEqual(J.check_root(a), [])
        self.assertEqual(os.stat(os.path.join(a, "x.o")).st_mtime_ns, 1_700_000_000_123_456_789)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: compares realpath with abspath; macOS tmp dirs are symlinks (/var -> /private/var)")
    def test_env_pins_every_cache_under_the_root(self):
        """Input: env_for(root). Expect: the four cache roots under root, autotune caching on.
        Proves: the environment mapping points every JIT cache variable under one root. Not:
        that a started process writes its caches there (the warm-restart GPU runs in
        kb/costs.md show that), or that FlashInfer's build files keep relative paths (they
        do not)."""
        env = J.env_for(self.dir)
        for k in J.ROOTS:
            self.assertTrue(env[k].startswith(os.path.abspath(self.dir)), k)
        self.assertEqual(env["TRITON_CACHE_AUTOTUNING"], "1")

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_preset_torch_key_equals_the_computed_key(self):
        """Input: process 1 computes and stores torch_key; process 2 presets it from the file;
        process 3 computes it from scratch. Expect: 1 is a miss, 2 a hit, and all three keys
        equal. Proves: the preset key is the key Inductor would compute (a wrong key would
        miss every FX-graph cache entry, or worse, hit a stale one). Not: a torch edit."""
        path = os.path.join(self.dir, "torch_key.json")
        first, second, ref = run_key(path), run_key(path), run_key(path, fresh=True)
        self.assertFalse(first["hit"])
        self.assertTrue(second["hit"], second)
        self.assertEqual(first["key"], ref["key"])
        self.assertEqual(second["key"], ref["key"])


if __name__ == "__main__":
    unittest.main()
