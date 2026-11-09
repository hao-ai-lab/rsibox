"""Golden CPU cases for the Megatron lab's pieces that need no GPU and no Megatron.

The adapter as a whole (hook.py on a real Megatron job) is proven by the conformance run on
GPU (receipt in ../README.md); these cases pin the three pieces a reader cannot see in it.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# hook.py hosts the live-lab kernel, which still lives in the vLLM artifact (AGENTS.md: shared code moves to
# the framework when a second engine needs it); at run time FASTRSI_KERNEL points at it, here the repo copy
KERNEL = os.environ.get("FASTRSI_KERNEL") or os.path.join(HERE, "..", "..", "..", "vllm", "livelab")
from fastrsi_megatron import client, gap  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class Overlay(unittest.TestCase):
    def test_fresh_process_runs_the_edited_module_under_its_own_file_name(self):
        """A fresh check must run the edit with the same file name and lines as the live swap,
        or witness keys and tracebacks would not line up. Input: pkg/mod.py `f() -> 1`, an
        overlay `pkg.mod.py` with `f() -> 2`. Expect: f() == 2 and co_filename == pkg/mod.py.
        Without the overlay the same process prints 1."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src", "pkg"))
            open(os.path.join(d, "src", "pkg", "__init__.py"), "w").close()
            orig = os.path.join(d, "src", "pkg", "mod.py")
            with open(orig, "w") as f:
                f.write("def f():\n    return 1\n")
            os.makedirs(os.path.join(d, "ov"))
            with open(os.path.join(d, "ov", "pkg.mod.py"), "w") as f:
                f.write("def f():\n    return 2\n")
            code = ("import fastrsi_megatron.overlay, pkg.mod as m, json; "
                    "print(json.dumps([m.f(), m.f.__code__.co_filename]))")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.path.dirname(HERE),
                                                                os.path.join(d, "src")])}
            env.pop("FASTRSI_KERNEL", None)
            on = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env={**env, "FASTRSI_OVERLAY": os.path.join(d, "ov")})
            off = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env=env)
            self.assertEqual(json.loads(on.stdout), [2, orig], on.stderr)
            self.assertEqual(json.loads(off.stdout)[0], 1, off.stderr)


class Mailbox(unittest.TestCase):
    def test_request_and_reply_through_a_directory(self):
        """A client on another host writes req-<id>.json; rank 0's mailbox thread queues it like
        a socket request and the reply comes back as resp-<id>.json, which the client removes.
        Expect: the reply, and no files left but the taken request."""
        with tempfile.TemporaryDirectory() as d:
            import queue
            q = queue.Queue()
            gap.Mailbox(d, q, poll_s=0.005)

            def serve():
                req, reply = q.get(timeout=5)
                reply({"ok": True, "echo": req["verb"], "n": req["n"] + 1})
            t = threading.Thread(target=serve)
            t.start()
            r = client.call_mailbox(d, {"verb": "status", "n": 41}, timeout=5, poll_s=0.005)
            t.join()
            self.assertEqual(r, {"ok": True, "echo": "status", "n": 42})
            left = sorted(os.listdir(d))
            self.assertEqual(len(left), 1)
            self.assertTrue(left[0].startswith("req-") and left[0].endswith(".json.taken"), left)

    def test_no_reply_is_a_timeout_not_a_hang(self):
        """A lab hung in a step never answers: the client must raise, so the caller poisons the
        lab instead of waiting forever."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):
                client.call_mailbox(d, {"verb": "check"}, timeout=0.2, poll_s=0.01)


class _MegatronLikeOptimizer:
    """The attributes state.py reads: fp32 master shards and an inner torch optimizer."""

    def __init__(self, model):
        self.params = [p for p in model.parameters()]
        self.shard_fp32_from_float16_groups = [[p.detach().clone().float() for p in self.params]]
        for m in self.shard_fp32_from_float16_groups[0]:
            m.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.shard_fp32_from_float16_groups[0], lr=0.1)

    def step(self):
        for m, p in zip(self.shard_fp32_from_float16_groups[0], self.params):
            m.grad = p.grad.float()
        self.optimizer.step()
        with torch.no_grad():
            for m, p in zip(self.shard_fp32_from_float16_groups[0], self.params):
                p.copy_(m)


class _Sched:
    def __init__(self):
        self.num_steps = 0

    def step(self, increment):
        self.num_steps += increment


@unittest.skipIf(torch is None, "needs torch")
class StartState(unittest.TestCase):
    def _train(self, model, opt, sched, x):
        model.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        sched.step(1)
        return loss.item()

    def test_every_check_starts_from_the_same_state(self):
        """The lab's start state S: weights as built, Adam moments zero, step counters zero.
        Input: a model, S0 before any step, 2 warm-up steps, restore S0 and zero the Adam state
        the warm-up created, snapshot S. Then 3 steps, restore S, 3 steps again.
        Expect: both runs give the same losses bit for bit, and they equal a fresh optimizer's
        3 steps from the same initial weights (what a fresh process would compute). Without the
        zeroing, the second run starts with the warm-up's moments and differs."""
        from fastrsi_megatron import state as ST
        torch.manual_seed(0)
        x = torch.randn(8, 16)
        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
        init = {k: v.clone() for k, v in model.state_dict().items()}
        opt, sched = _MegatronLikeOptimizer(model), _Sched()
        s0 = ST.Snapshot([model], opt, sched)
        for _ in range(2):
            self._train(model, opt, sched, x)
        s0.restore()
        zeroed, unknown = ST.zero_new_state(opt, set(s0.host))
        self.assertEqual(unknown, [])
        self.assertEqual(len(zeroed), 3 * 4)                 # 4 tensors x (exp_avg, exp_avg_sq, step)
        s = ST.Snapshot([model], opt, sched)
        a = [self._train(model, opt, sched, x) for _ in range(3)]
        s.restore()
        b = [self._train(model, opt, sched, x) for _ in range(3)]
        fresh_model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Tanh(),
                                          torch.nn.Linear(16, 4))
        fresh_model.load_state_dict(init)
        fopt, fsched = _MegatronLikeOptimizer(fresh_model), _Sched()
        c = [self._train(fresh_model, fopt, fsched, x) for _ in range(3)]
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(sched.num_steps, 3)

    def test_unknown_state_is_reported_not_zeroed(self):
        """A state key that is not a fresh Adam's zero (here a master copy kept in the state)
        must come back as unknown and keep its value."""
        from fastrsi_megatron import state as ST
        model = torch.nn.Linear(4, 4)
        opt = _MegatronLikeOptimizer(model)
        model(torch.ones(1, 4)).sum().backward()
        opt.step()
        p = opt.shard_fp32_from_float16_groups[0][0]
        opt.optimizer.state[p]["master_param"] = torch.full((2,), 7.0)
        zeroed, unknown = ST.zero_new_state(opt, set())
        self.assertTrue(any(n.endswith("master_param") for n in unknown), unknown)
        self.assertEqual(opt.optimizer.state[p]["master_param"].tolist(), [7.0, 7.0])

    def test_unknown_state_finding_is_built_not_raised(self):
        """Input: two optimizer state keys the lab cannot reset (master copies, as the precision-aware
        optimizer creates). Expect: a finding in the kernel's shape (check optimizer_state_unknown,
        the keys in `observed`, n_unknown 2) that serializes to JSON, built without the kernel's
        catalog, which does not know this Megatron-only check. Does not run Lab._setup itself."""
        if KERNEL not in sys.path:
            sys.path.insert(0, KERNEL)
        from fastrsi_megatron import hook
        f = hook.unknown_state_finding(["opt0.state.3.master_param", "opt0.state.4.master_param"])
        self.assertEqual(f["check"], "optimizer_state_unknown")
        self.assertEqual(f["observed"],
                         "state keys not reset: ['opt0.state.3.master_param', 'opt0.state.4.master_param']")
        self.assertEqual(f["n_unknown"], 2)
        json.dumps(f)   # findings travel in replies and receipts


if __name__ == "__main__":
    unittest.main()
