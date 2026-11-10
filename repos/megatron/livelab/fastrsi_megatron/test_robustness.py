"""Regression cases for the Megatron lab's robustness (audit rows 11, 12, 25).

Real processes, real Unix sockets, real fork and signals; the Lab cases use the CPU train step of
test_pieces (no Megatron, no GPU). What the GPU run adds (a real Qwen3-14B job through the zygote
and the lab, with the same faults injected) is in evidence/fix-megatron/.

Linux only (the zygote reads /proc and uses prctl).
"""
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fastrsi_megatron import client, gap, zygote  # noqa: E402

try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None

LINUX = sys.platform.startswith("linux")

# a rank that starts a child of its own (as Megatron's data loader and compile workers do), records
# both pids and sleeps
TRAIN = """import json, os, subprocess, sys, time
kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
json.dump({"pid": os.getpid(), "kid": kid.pid},
          open(os.path.join(os.environ["OUT"], "p%s.json" % os.environ["RANK"]), "w"))
time.sleep(float(os.environ.get("SLEEP", "300")))
"""


def _stat(pid):
    """(state, ppid) of pid, or None when it is gone (reaped)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(")", 1)[1].split()
        return rest[0], int(rest[1])
    except (OSError, IndexError, ValueError):
        return None


def _running(pid):
    s = _stat(pid)
    return s is not None and s[0] != "Z"


def _wait(cond, timeout=15.0):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _raw(sock, data, timeout=10):
    """Send raw bytes as one request; return the first reply line (parsed) or None."""
    c = socket.socket(socket.AF_UNIX)
    c.settimeout(timeout)
    c.connect(sock)
    c.sendall(data)
    line = c.makefile("rb").readline()
    c.close()
    return json.loads(line) if line else None


@unittest.skipUnless(LINUX, "zygote needs Linux")
class ZygoteSupervision(unittest.TestCase):
    """Row 11: a bad request or a gone client must not kill the zygote, and no rank or child of a
    rank may outlive its job."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(os.path.join(self.d, "heavy.py"), "w") as f:
            f.write("X = 1\n")
        with open(os.path.join(self.d, "train.py"), "w") as f:
            f.write(TRAIN)
        self.sock = os.path.join(self.d, "z.sock")
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + self.d,
                   FASTRSI_ZYGOTE_FORKSAFE="0")
        self.z = subprocess.Popen([sys.executable, "-m", "fastrsi_megatron.zygote", "serve",
                                   "--sock", self.sock, "--preload", "heavy"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  env=env)
        self.ready = json.loads(self.z.stdout.readline())
        self.pids = []

    def tearDown(self):
        for p in self.pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.z.poll() is None:
            try:
                list(zygote.request(self.sock, {"op": "stop"}))
            except OSError:
                self.z.kill()
        self.z.wait(10)

    def req(self, nproc=2, **env):
        return {"argv": [os.path.join(self.d, "train.py")], "nproc": nproc, "port": 29998,
                "entry": "script", "log_dir": self.d, "env": dict({"OUT": self.d}, **env),
                "t0": time.time()}

    def launch(self, req):
        """Send a start request, read `forked`, wait for every rank's pid file; return the open
        client socket and the pids (ranks and their children)."""
        c = socket.socket(socket.AF_UNIX)
        c.connect(self.sock)
        c.sendall((json.dumps(req) + "\n").encode())
        f = c.makefile("rb")
        ev = json.loads(f.readline())
        self.assertEqual(ev["event"], "forked", ev)
        files = [os.path.join(self.d, f"p{r}.json") for r in range(req["nproc"])]
        self.assertTrue(_wait(lambda: all(os.path.exists(p) and os.path.getsize(p) for p in files)))
        pids = []
        for p in files:
            with open(p) as fh:
                o = json.load(fh)
            pids += [o["pid"], o["kid"]]
        self.pids += pids
        return c, f, pids

    def pong(self):
        return [e["event"] for e in zygote.request(self.sock, {"op": "ping"})] == ["pong"]

    def test_malformed_request_lines_get_an_error_and_the_zygote_keeps_serving(self):
        """Input: a line that is not JSON, a JSON value that is not an object, and an object with
        no `nproc`. Expect: each answered with an `error` event, then ping answers. Before the
        fix, json.loads outside the try killed the zygote on the first line."""
        for line in (b"not json\n", b"[1, 2]\n", b'{"argv": ["x.py"]}\n'):
            ev = _raw(self.sock, line)
            self.assertEqual(ev and ev["event"], "error", (line, ev))
            self.assertIsNone(self.z.poll(), line)
        self.assertTrue(self.pong())

    def test_client_gone_mid_run_stops_and_reaps_the_job(self):
        """Input: 2 ranks, each with a child, sleeping 300 s; the client closes its socket after
        `forked`. Expect (default policy, as torchrun when its launcher dies): every rank and child
        gone and reaped (no zombie left under the zygote) within seconds, and the zygote answers
        ping. Before the fix the ranks ran on and the zygote died sending `exited` to the closed
        socket."""
        c, f, pids = self.launch(self.req())
        f.close()                               # the socket closes with its last reference
        c.close()
        self.assertTrue(_wait(lambda: not any(_running(p) for p in pids)), [_stat(p) for p in pids])
        self.assertTrue(_wait(lambda: not any(_stat(p) and _stat(p)[1] == self.z.pid for p in pids)))
        self.assertTrue(self.pong())

    def test_client_gone_with_keep_policy_lets_the_job_finish_and_survives(self):
        """Input: on_client_gone=keep, ranks sleep 1 s, the client closes after `forked`.
        Expect: the ranks finish on their own, are reaped, and the zygote then answers ping
        (sending `exited` to the closed socket is not fatal)."""
        r = self.req(SLEEP="1")
        r["on_client_gone"] = "keep"
        c, f, pids = self.launch(r)
        f.close()
        c.close()
        ranks = pids[0::2]
        self.assertTrue(_wait(lambda: not any(_stat(p) for p in ranks)))
        self.assertTrue(_wait(self.pong))
        self.assertIsNone(self.z.poll())

    def test_zygote_death_kills_every_rank_and_its_children(self):
        """Input: a running job, then SIGKILL to the zygote (no handler can run). Expect: both
        ranks and both of their children stop within seconds (PR_SET_PDEATHSIG on each rank, and
        the rank's guard kills the rank's process group). Before the fix all four ran on."""
        c, f, pids = self.launch(self.req())
        self.z.kill()
        self.z.wait(10)
        self.assertTrue(_wait(lambda: not any(_running(p) for p in pids)), [_stat(p) for p in pids])
        c.close()

    def test_failing_rank_still_stops_the_others_with_sigterm(self):
        """The existing policy stays: rank 1 exits 7, rank 0 is stopped with SIGTERM (-15)."""
        script = os.path.join(self.d, "fail.py")
        with open(script, "w") as f:
            f.write("import os, sys, time\nif os.environ['RANK'] == '1':\n    sys.exit(7)\n"
                    "time.sleep(60)\n")
        r = self.req()
        r["argv"] = [script]
        evs = list(zygote.request(self.sock, r))
        self.assertEqual(evs[-1]["rc"], [-15, 7])


class LabFixture:
    """A Lab on a CPU model with a one-rank gloo stand-in; check steps sleep `slow` seconds."""

    def __init__(self, tc, slow=0.0):
        from fastrsi_megatron import hook
        from fastrsi_megatron.test_pieces import _MegatronLikeOptimizer, _Sched
        self.hook = hook
        torch.manual_seed(0)
        self.batches = [torch.randn(4, 8) for _ in range(12)]
        self.model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8),
                                         torch.nn.Linear(8, 4))
        self.opt, self.sched, self.slow = _MegatronLikeOptimizer(self.model), _Sched(), slow
        fx = self

        def train_step(fwd, it, models, opt, sched, *rest, **kw):
            if fx.slow:
                time.sleep(fx.slow)
            b = next(it)
            models[0].zero_grad()
            loss = models[0](b).square().mean()
            loss.backward()
            opt.step()
            sched.step(1)
            random.random()                     # Python and numpy RNG move in a step too
            if np is not None:
                np.random.rand()
            return ({"lm loss": loss.detach()}, False, False, False, 0, 1.0)

        class Ranks:
            rank, world = 0, 1
            group = None

            def round(self, msg, handle):
                return [handle(msg)]

            def bcast(self, msg=None):
                return msg

            def gather(self, r):
                return [r]
        self.train_step = train_step
        env = {"FASTRSI_WARMUP": "2", "FASTRSI_CHECK_STEPS_MAX": "4", "FASTRSI_CHECK_STEPS": "2"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(hook.G, "Ranks", return_value=Ranks()), \
                mock.patch.object(hook.Lab, "_mb", return_value=1):
            self.it = iter(self.batches)
            self.lab = hook.Lab(train_step, [None, self.it, [self.model], self.opt, self.sched],
                                {"iteration": 0})


@unittest.skipUnless(torch is not None and LINUX, "needs torch on Linux")
class LabServing(unittest.TestCase):
    """Row 12: rank 0's serve loop survives a bad request, a client that times out before the
    reply and a client that disconnects; the other ranks never see an exception from it."""

    def test_server_reply_to_a_gone_client_returns_false(self):
        """gap.Server: the client sends a request and closes before the reply. Expect: reply()
        returns False and raises nothing. Before the fix: BrokenPipeError out of reply()."""
        with tempfile.TemporaryDirectory() as d:
            s = gap.Server(os.path.join(d, "s.sock"))
            c = socket.socket(socket.AF_UNIX)
            c.connect(s.path)
            c.sendall(b'{"verb": "status"}\n')
            c.close()
            req, reply = s.next(timeout=5)
            time.sleep(0.1)
            self.assertIs(reply({"ok": True, "pad": "x" * 100000}), False)
            s.close()

    def test_lab_keeps_serving_after_bad_requests_a_timeout_and_a_disconnect(self):
        """Input, in order: `not json`, `[]`, `{"verb": 5}`, a check whose client times out after
        0.2 s (the check takes 1 s), a status whose client closes at once. Expect: errors for the
        bad ones, then status answers, a check gives the same loss as before, and exit returns
        serve(). Before the fix serve() raised BrokenPipeError on the timed-out client's reply and
        the next request got no answer."""
        fx = LabFixture(self)
        with tempfile.TemporaryDirectory() as d:
            sock = os.path.join(d, "lab.sock")
            out = {}
            with mock.patch.dict(os.environ, {"FASTRSI_SOCK": sock}):
                t = threading.Thread(target=lambda: out.setdefault("ret", fx.lab.serve()))
                t.start()
                self.assertTrue(_wait(lambda: os.path.exists(sock)))
                base = client.call(sock, {"verb": "check", "steps": 2}, 30)
                self.assertTrue(base["ok"], base)
                for line in (b"not json\n", b"[]\n", b'{"verb": 5}\n'):
                    r = _raw(sock, line)
                    self.assertFalse(r["ok"], line)
                fx.slow = 0.5
                with self.assertRaises(TimeoutError):
                    client.call(sock, {"verb": "check", "steps": 2}, timeout=0.2)
                c = socket.socket(socket.AF_UNIX)
                c.connect(sock)
                c.sendall(b'{"verb": "status"}\n')
                c.close()
                fx.slow = 0.0
                st = client.call(sock, {"verb": "status"}, 30)
                self.assertEqual(st.get("state"), "base", st)
                again = client.call(sock, {"verb": "check", "steps": 2}, 30)
                self.assertEqual(again["output"], base["output"])
                self.assertTrue(client.call(sock, {"verb": "exit"}, 30)["ok"])
                t.join(10)
            self.assertFalse(t.is_alive())
            self.assertEqual(out.get("ret"), "exit")


@unittest.skipUnless(torch is not None, "needs torch")
class ResetFailsClosed(unittest.TestCase):
    """Row 25: reset restores every piece of state a step changes, proves it bit for bit, and
    fails (writing nothing) when the live state no longer matches the snapshot."""

    def test_snapshot_covers_buffers_python_numpy_rng_and_grad_scaler(self):
        """Input: S taken; two steps move weights, Adam state, BatchNorm running stats (buffers),
        Python and numpy RNG and a grad scaler's scale. Expect: digest differs after the steps and
        equals S's after restore; the next Python/numpy draws equal those drawn right after S."""
        from fastrsi_megatron import state as ST
        fx = LabFixture(self)

        class Scaler:                           # Megatron's DynamicGradScaler state_dict shape
            def __init__(self):
                self._scale, self.growth = torch.tensor([65536.0]), 0

            def state_dict(self):
                return {"scale": self._scale, "growth_tracker": self.growth}

            def load_state_dict(self, sd):
                self._scale, self.growth = sd["scale"].clone(), sd["growth_tracker"]
        fx.opt.grad_scaler = Scaler()
        s = ST.Snapshot([fx.model], fx.opt, fx.sched)
        d0 = ST.digest([fx.model], fx.opt, fx.sched)
        self.assertIn("buffers", s.restored)
        self.assertIn("grad_scaler", s.restored)
        state = (random.getstate(), np.random.get_state() if np is not None else None)
        py0 = random.random()
        np0 = np.random.rand() if np is not None else None
        random.setstate(state[0])
        if np is not None:
            np.random.set_state(state[1])
        fx.lab._steps(2)
        fx.opt.grad_scaler._scale.mul_(0.5)
        fx.opt.grad_scaler.growth = 3
        self.assertNotEqual(ST.digest([fx.model], fx.opt, fx.sched)["all"], d0["all"])
        r = s.restore()
        self.assertTrue(r["ok"], r)
        d1 = ST.digest([fx.model], fx.opt, fx.sched)
        self.assertEqual(d1, d0)
        self.assertEqual(random.random(), py0)
        if np is not None:
            self.assertEqual(np.random.rand(), np0)
        self.assertEqual(fx.opt.grad_scaler._scale.item(), 65536.0)

    def test_restore_with_extra_or_missing_state_fails_and_writes_nothing(self):
        """Input: S, a step, then an optimizer state tensor S does not have (extra); then one S has
        that the live optimizer lost (missing). Expect: ok False with the names, and the live
        weights untouched (fail closed, not a partial restore). Before the fix restore wrote the
        tensors it knew and reset answered ok True."""
        from fastrsi_megatron import state as ST
        fx = LabFixture(self)
        s = ST.Snapshot([fx.model], fx.opt, fx.sched)
        fx.lab._steps(1)
        after = ST.digest([fx.model], fx.opt, fx.sched)
        p = fx.opt.shard_fp32_from_float16_groups[0][0]
        fx.opt.optimizer.state[p]["master_param"] = torch.zeros(2)
        r = s.restore()
        self.assertFalse(r["ok"])
        self.assertEqual(r["n_extra"], 1)
        self.assertTrue(r["extra"][0].endswith("master_param"), r)
        del fx.opt.optimizer.state[p]["master_param"]
        self.assertEqual(ST.digest([fx.model], fx.opt, fx.sched), after)
        del fx.opt.optimizer.state[p]["exp_avg"]
        r = s.restore()
        self.assertFalse(r["ok"])
        self.assertEqual(r["n_missing"], 1)

    def test_restore_with_a_changed_shape_fails(self):
        from fastrsi_megatron import state as ST
        fx = LabFixture(self)
        s = ST.Snapshot([fx.model], fx.opt, fx.sched)
        p = fx.opt.shard_fp32_from_float16_groups[0][0]
        fx.opt.optimizer.state[p]["exp_avg"] = torch.zeros(3)
        r = s.restore()
        self.assertFalse(r["ok"])
        self.assertTrue(r["mismatch"], r)

    def test_reset_and_check_verbs_fail_when_restore_fails(self):
        """Input: a lab, then an optimizer state key S does not have. Expect: reset ok False with
        an empty `reset` list and the rank's error; check ok False without running a step. Before
        the fix reset answered ok True with a fixed list."""
        fx = LabFixture(self)
        self.assertTrue(fx.lab.verb("reset", {"verb": "reset"})["ok"])
        p = fx.opt.shard_fp32_from_float16_groups[0][0]
        fx.opt.optimizer.state[p]["master_param"] = torch.zeros(2)
        r = fx.lab.verb("reset", {"verb": "reset"})
        self.assertFalse(r["ok"], r)
        self.assertEqual(r["reset"], [])
        n = fx.sched.num_steps
        c = fx.lab.verb("check", {"verb": "check", "steps": 2})
        self.assertFalse(c["ok"], c)
        self.assertEqual(fx.sched.num_steps, n)

    def test_reset_round_trip_is_bitwise_and_resume_starts_from_s_with_the_frozen_batches(self):
        """Input: digest at S, a check, digest, reset, digest; then resume. Expect: digest after
        the check differs, after reset equals S's on every piece; resume restores S and the job's
        next train steps are fed the frozen batches first, then the real iterator: their losses
        equal the check's (the job continues as if the lab had not run)."""
        fx = LabFixture(self)
        d0 = fx.lab.verb("digest", {"verb": "digest"})
        c = fx.lab.verb("check", {"verb": "check", "steps": 2})
        d1 = fx.lab.verb("digest", {"verb": "digest"})
        self.assertNotEqual(d1["digest"], d0["digest"])
        self.assertTrue(fx.lab.verb("reset", {"verb": "reset"})["ok"])
        d2 = fx.lab.verb("digest", {"verb": "digest"})
        self.assertEqual(d2["ranks"], d0["ranks"])
        fx.lab.verb("check", {"verb": "check", "steps": 2})
        r = fx.lab.verb("resume", {"verb": "resume"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(fx.lab.verb("digest", {"verb": "digest"})["digest"], d0["digest"])
        it = fx.lab.replay(fx.it)
        losses = [fx.train_step(None, it, [fx.model], fx.opt, fx.sched)[0]["lm loss"].item()
                  for _ in range(2)]
        self.assertEqual(losses, c["output"]["loss"])


if __name__ == "__main__":
    unittest.main()
