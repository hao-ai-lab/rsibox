"""Golden case for stage_local.sh on a toy venv and tree (real script, real copies, no GPU).
What the copy buys on a real environment is in kb/costs.md (restart tiers)."""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class StageLocal(unittest.TestCase):
    def test_staged_env_points_at_the_copy(self):
        """Input: a venv whose console script names the venv's python in its shebang, a tree
        with .git, a cache root with inductor/. Expect: the copied script names DEST's python
        (else torchrun's ranks import from the original location again: 54 s vs 29 s at 14B),
        no .git in the copy, the inductor cache seeded, and env.sh putting the copies first
        and the caches under DEST/cache."""
        with tempfile.TemporaryDirectory() as d:
            venv, tree, cf, dest = (os.path.join(d, x) for x in ("venv", "tree", "cf", "out"))
            os.makedirs(os.path.join(venv, "bin"))
            os.makedirs(os.path.join(venv, "lib", "python3.12", "site-packages", "pkg"))
            open(os.path.join(venv, "pyvenv.cfg"), "w").close()
            with open(os.path.join(venv, "bin", "torchrun"), "w") as f:
                f.write(f"#!{venv}/bin/python\nimport sys\n")
            os.makedirs(os.path.join(tree, ".git"))
            os.makedirs(os.path.join(tree, "megatron"))
            os.makedirs(os.path.join(cf, "inductor"))
            open(os.path.join(cf, "inductor", "graph"), "w").close()
            subprocess.run(["bash", os.path.join(HERE, "stage_local.sh"), venv, tree, dest, "2"],
                           check=True, capture_output=True, env=dict(os.environ, CACHE_FROM=cf))
            with open(os.path.join(dest, "venv", "bin", "torchrun")) as f:
                self.assertEqual(f.readline().strip(), f"#!{dest}/venv/bin/python")
            self.assertEqual(sorted(os.listdir(os.path.join(dest, "tree"))), ["megatron"])
            self.assertTrue(os.path.exists(os.path.join(dest, "cache", "inductor", "graph")))
            env = open(os.path.join(dest, "env.sh")).read()
            self.assertIn(f"export PATH={dest}/venv/bin:", env)
            self.assertIn(f"export PYTHONPATH={dest}/tree", env)
            self.assertIn(f"CUDA_CACHE_PATH={dest}/cache/cuda", env)
            self.assertIn(f"TORCHINDUCTOR_CACHE_DIR={dest}/cache/inductor", env)
            self.assertIn("TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-8}", env)


if __name__ == "__main__":
    unittest.main()
