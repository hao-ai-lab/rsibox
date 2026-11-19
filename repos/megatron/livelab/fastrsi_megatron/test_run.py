"""Golden CPU case for run.py's first-step split, against a stand-in `megatron.training.training`
(the five functions pretrain() calls and a train_step that runs 3 microbatches and an optimizer
step). Real run.py, real subprocess; no GPU (the synchronize is a no-op without CUDA)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

TRAINING = """import time
def initialize_megatron(): pass
def setup_model_and_optimizer(): pass
def build_train_valid_test_data_iterators(): pass
def get_model(): pass
def get_megatron_optimizer(): pass
def warmup_training_kernels(): pass
def train_step(forward_step_func, data_iterator, model, optimizer, *rest):
    for i in range(3):
        forward_step_func(i)
        time.sleep(0.02)                     # the microbatch's backward
    optimizer.step()
    return {"lm loss": 1.5}, 0, 0, 0, 0, 2.0
"""
SCRIPT = """import json, time
import megatron.training.training as tr
class Opt:
    def step(self):
        time.sleep(0.05)
opt = Opt()
original = opt.step
opt.step = original
for _ in range(3):
    tr.train_step(lambda i: time.sleep(0.01), None, None, opt)
print(json.dumps({"restored": opt.step is original}))
"""


class StepDetail(unittest.TestCase):
    def test_first_k_steps_are_split_into_microbatches_and_optimizer(self):
        """Input: FASTRSI_STEP_DETAIL=2, three train steps of 3 microbatches (10 ms forward,
        20 ms backward) and a 50 ms optimizer step. Expect: steps 1-2 carry detail.mb with 3
        forward starts about 30 ms apart and opt_done - opt_start about 50 ms; step 3 has no
        detail; the optimizer's own step method is back afterwards. Without the split the
        steps have no "detail" (the first-step cost is one number)."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "megatron", "training"))
            for f in ("megatron/__init__.py", "megatron/training/__init__.py"):
                open(os.path.join(d, f), "w").close()
            with open(os.path.join(d, "megatron", "training", "training.py"), "w") as f:
                f.write(TRAINING)
            if importlib.util.find_spec("torch") is None:
                with open(os.path.join(d, "torch.py"), "w") as f:
                    f.write("class cuda:\n    is_available = staticmethod(lambda: False)\n")
            with open(os.path.join(d, "pretrain.py"), "w") as f:
                f.write(SCRIPT)
            env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + d,
                       FASTRSI_RUN=d, FASTRSI_STEP_DETAIL="2", RANK="0")
            out = subprocess.run([sys.executable, "-m", "fastrsi_megatron.run",
                                  os.path.join(d, "pretrain.py")], env=env, cwd=d,
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertTrue(json.loads(out.stdout.strip().splitlines()[-1])["restored"])
            steps = json.load(open(os.path.join(d, "phases.rank0.json")))["steps"]
            self.assertEqual(len(steps), 3)
            for s in steps[:2]:
                mb, det = s["detail"]["mb"], s["detail"]
                self.assertEqual(len(mb), 3)
                self.assertGreater(mb[2] - mb[0], 0.05)
                self.assertGreater(det["opt_done"] - det["opt_start"], 0.04)
                self.assertGreater(det["opt_start"], mb[2])
            self.assertNotIn("detail", steps[2])
            self.assertEqual(steps[0]["loss"], 1.5)


if __name__ == "__main__":
    unittest.main()
