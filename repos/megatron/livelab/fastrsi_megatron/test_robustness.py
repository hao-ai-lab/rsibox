"""Regression cases for the Megatron lab's robustness (audit rows 11, 12, 25).

Real processes, real Unix sockets, real fork and signals; the Lab cases use the CPU train step of
test_pieces (no Megatron, no GPU). What the GPU run adds (a real Qwen3-14B job through the zygote
and the lab, with the same faults injected) is summarized in the PR description.

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
LIVELAB_DIR = os.path.dirname(HERE)
sys.path.insert(0, LIVELAB_DIR)
from fastrsi_megatron import LIVELAB, client, gap, zygote  # noqa: E402

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
with open(os.path.join(os.environ["OUT"], "p%s.json" % os.environ["RANK"]), "w") as f:
    json.dump({"pid": os.getpid(), "kid": kid.pid}, f)
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


def _sock_ready(path):
    if not os.path.exists(path):
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(path)
            return True
        except OSError:
            return False


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
        self.addCleanup(self.z.stdout.close)
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
                t = threading.Thread(target=lambda: out.setdefault("ret", fx.lab.serve()), daemon=True)
                t.start()
                try:
                    self.assertTrue(_wait(lambda: _sock_ready(sock)))
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
                finally:
                    if t.is_alive():
                        try:
                            client.call(sock, {"verb": "exit"}, 5)
                        except OSError:
                            pass
                        t.join(5)
            self.assertFalse(t.is_alive())
            self.assertEqual(out.get("ret"), "exit")


def _gloo_rank(rank, port, sock, outq):
    """One rank of a real 2-rank gloo job serving the lab on a CPU model (spawned)."""
    from fastrsi_megatron import hook
    from fastrsi_megatron.test_pieces import _MegatronLikeOptimizer, _Sched
    os.environ.update(GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE="2", FASTRSI_SOCK=sock, FASTRSI_WARMUP="1",
                      FASTRSI_CHECK_STEPS_MAX="2", FASTRSI_CHECK_STEPS="1",
                      FASTRSI_ROUND_TIMEOUT_S="3", FASTRSI_HANG_S="5")
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))

    def train_step(fwd, it, models, opt, sched, *rest, **kw):
        models[0].zero_grad()
        loss = models[0](next(it)).square().mean()
        loss.backward()
        opt.step()
        sched.step(1)
        return ({"lm loss": loss.detach()}, False, False, False, 0, 1.0)
    try:
        with mock.patch.object(hook.Lab, "_mb", return_value=1):
            lab = hook.Lab(train_step, [None, iter([torch.randn(4, 8) for _ in range(8)]),
                                        [model], _MegatronLikeOptimizer(model), _Sched()],
                           {"iteration": 0})
            outq.put((rank, f"served {lab.serve()}"))
    except BaseException as error:
        outq.put((rank, f"{type(error).__name__}: {error}"))


@unittest.skipUnless(torch is not None and torch.distributed.is_available(), "needs torch")
class RanksRound(unittest.TestCase):
    """Real 2-rank gloo job, real lab, real socket (no numpy needed): a rank fault must reach the
    right rank; a rank that dies in a round must fail it on rank 0 within the round timeout (3 s
    here, FASTRSI_ROUND_TIMEOUT_S) after rank 0's own part, not block it for the group's 7-day
    timeout; and a healthy rank that answers long before rank 0 must not fail the round."""

    def setUp(self):
        import multiprocessing
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.folder = tempfile.TemporaryDirectory()
        self.sock = os.path.join(self.folder.name, "lab.sock")
        context = multiprocessing.get_context("spawn")
        self.outq = context.Queue()
        self.procs = [context.Process(target=_gloo_rank, args=(rank, port, self.sock, self.outq),
                                      daemon=True) for rank in range(2)]
        for proc in self.procs:
            proc.start()
        self.assertTrue(_wait(lambda: _sock_ready(self.sock), 120), "lab never ready")
        first = client.call(self.sock, {"verb": "status"}, 30)
        self.assertEqual(first["state"], "base", first)

    def tearDown(self):
        for proc in self.procs:
            if proc.is_alive():
                proc.kill()
            proc.join(10)
        self.folder.cleanup()

    def call(self, request, timeout=30):
        return client.call(self.sock, request, timeout)

    def assert_poisoned_by_a_failed_round(self):
        status = self.call({"verb": "status"})
        self.assertEqual(status["state"], "poisoned", status)
        # first the failed round; the kernel's Session then adds `tampered` for the fingerprint
        # that round could not give
        self.assertEqual(status["poison"][0]["check"], "participant_failed", status["poison"])
        refused = self.call({"verb": "install", "patch": {"files": {}}})
        self.assertEqual(refused["codes"], ["session_poisoned"], refused)

    def test_inject_targets_a_named_rank_and_a_lost_rank_fails_the_round(self):
        """Input: inject fail_commit with target "rank1", then "rank9"; then inject lost on rank 1
        and a status (a fingerprint round, in which rank 1 exits). Expect: "rank1" armed on rank 1
        only; "rank9" refused; the status answers within 15 s, poisoned by participant_failed.
        Before the fix both targets were skipped on every rank with ok True, and the status
        never came (rank 0 hung in the gather)."""
        named = self.call({"verb": "inject", "fault": "fail_commit", "target": "rank1"})
        self.assertTrue(named["ok"], named)
        self.assertEqual([bool(reply.get("skipped")) for reply in named["ranks"]], [True, False])
        unknown = self.call({"verb": "inject", "fault": "fail_commit", "target": "rank9"})
        self.assertFalse(unknown["ok"], unknown)
        self.assertTrue(self.call({"verb": "inject", "fault": "lost", "target": 1})["ok"])
        started = time.time()
        status = self.call({"verb": "status"}, 15)
        self.assertLess(time.time() - started, 15)
        self.assertEqual(status["state"], "poisoned", status)
        self.assert_poisoned_by_a_failed_round()
        self.procs[1].join(10)
        self.assertEqual(self.procs[1].exitcode, 18)

    def test_rank_answering_long_before_rank_0_does_not_fail_the_round(self):
        """Input: rank 0 sleeps 5 s in its check step (hang fault, FASTRSI_HANG_S=5), rank 1
        answers at once; round timeout 3 s. Expect: the check is ok with the loss of both
        steps, rank 1 still alive, status base. With replies on a 3 s gloo collective, rank 1
        raised "Timed out waiting 3000ms" out of serve() and rank 0 then hung."""
        self.assertTrue(self.call({"verb": "inject", "fault": "hang", "target": 0})["ok"])
        started = time.time()
        check = self.call({"verb": "check", "steps": 1}, 60)
        self.assertGreaterEqual(time.time() - started, 5)
        self.assertTrue(check["ok"], check)
        self.assertEqual(len(check["output"]["loss"]), 1)
        self.assertTrue(self.procs[1].is_alive())
        self.assertEqual(self.call({"verb": "status"})["state"], "base")

    def test_reply_after_a_failed_round_does_not_kill_rank_0(self):
        """Input: rank 1 sleeps 5 s in its check step (hang fault), round timeout 3 s, then its
        reply arrives about 2 s after rank 0 gave up. Expect: the check answers ok False within
        5 s, the lab is poisoned, and rank 0 survives the late reply: alive 4 s later and still
        answering status (poisoned); rank 1 is alive or exited with LOST_ROUND_EXIT, never 0.
        Before the fix rank 0 died with SIGABRT (gloo: "Cannot lock pointer to unbound buffer")
        when the reply landed in a receive buffer it had dropped."""
        self.assertTrue(self.call({"verb": "inject", "fault": "hang", "target": 1})["ok"])
        started = time.time()
        check = self.call({"verb": "check", "steps": 1}, 30)
        self.assertLess(time.time() - started, 5)
        self.assertFalse(check["ok"], check)
        time.sleep(4)                           # rank 1's reply lands meanwhile
        self.assertTrue(self.procs[0].is_alive(), f"rank 0 exit code {self.procs[0].exitcode}")
        self.assert_poisoned_by_a_failed_round()
        # rank 1 either still blocks in its send or receive (gloo's macOS transport) or its
        # send or receive failed and it left with LOST_ROUND_EXIT (Linux): never a clean 0
        from fastrsi_megatron import hook
        exit_code = self.procs[1].exitcode
        self.assertIn(exit_code, (None, hook.LOST_ROUND_EXIT), f"rank 1 exit code {exit_code}")

    def test_rank_dying_while_rank_0_is_busy_fails_the_round_within_the_bound(self):
        """Input: rank 1 exits at its check step (crash fault) while rank 0 sleeps 5 s in its own
        step; round timeout 3 s. Expect: the check answers ok False within 5 + 3 s and some
        slack (15 s), then the lab is poisoned by participant_failed. On base (replies gathered
        with gather_object on the 7-day group) the check got no answer in 30 s."""
        self.assertTrue(self.call({"verb": "inject", "fault": "hang", "target": 0})["ok"])
        self.assertTrue(self.call({"verb": "inject", "fault": "crash", "target": 1})["ok"])
        started = time.time()
        check = self.call({"verb": "check", "steps": 1}, 30)
        self.assertLess(time.time() - started, 15)
        self.assertFalse(check["ok"], check)
        self.assertIn("reply of rank 1", check["error"])
        self.assert_poisoned_by_a_failed_round()
        self.procs[1].join(10)
        self.assertEqual(self.procs[1].exitcode, 17)

    def test_exit_after_failed_round_replies_not_ok_and_exits_without_barrier_hang(self):
        """Real 2-rank gloo job running through install_hook: after rank 1 is lost and a round
        fails, sending `exit` replies ok False and rank 0 exits 0 within seconds instead of
        blocking on dist.barrier(group=lab.ranks.group) for the 7-day timeout."""
        code = ("import os, socket, torch\n"
                "from unittest import mock\n"
                "from fastrsi_megatron import hook\n"
                "from fastrsi_megatron.test_pieces import _MegatronLikeOptimizer, _Sched\n"
                "rank = int(os.environ['RANK'])\n"
                "os.environ.update(GLOO_SOCKET_IFNAME='lo0' if os.uname().sysname == 'Darwin' else 'lo',\n"
                "                  FASTRSI_WARMUP='1', FASTRSI_CHECK_STEPS_MAX='2',\n"
                "                  FASTRSI_CHECK_STEPS='1', FASTRSI_ROUND_TIMEOUT_S='2')\n"
                "torch.distributed.init_process_group('gloo', rank=rank, world_size=2)\n"
                "torch.manual_seed(0)\n"
                "model = torch.nn.Sequential(torch.nn.Linear(8, 4))\n"
                "class Tr:\n"
                "    @staticmethod\n"
                "    def train_step(fwd, it, models, opt, sched, *rest, **kw):\n"
                "        models[0].zero_grad()\n"
                "        loss = models[0](next(it)).square().mean()\n"
                "        loss.backward(); opt.step(); sched.step(1)\n"
                "        return ({'lm loss': loss.detach()}, False, False, False, 0, 1.0)\n"
                "with mock.patch.object(hook.Lab, '_mb', return_value=1):\n"
                "    hook.install_hook(Tr)\n"
                "    Tr.train_step(None, iter([torch.randn(4, 8) for _ in range(8)]),\n"
                "                  [model], _MegatronLikeOptimizer(model), _Sched(), iteration=0)\n")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            sock = os.path.join(folder, "hook-exit.sock")
            env_base = {**os.environ, "PYTHONPATH": os.path.dirname(HERE) + os.pathsep + LIVELAB_DIR,
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "WORLD_SIZE": "2",
                        "FASTRSI_SOCK": sock}
            procs = [subprocess.Popen([sys.executable, "-c", code],
                                      env={**env_base, "RANK": str(r)},
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                     for r in range(2)]
            try:
                self.assertTrue(_wait(lambda: _sock_ready(sock), 60), "hook lab never ready")
                self.assertEqual(client.call(sock, {"verb": "status"}, 20)["state"], "base")
                self.assertTrue(client.call(sock, {"verb": "inject", "fault": "lost", "target": 1}, 20)["ok"])
                st = client.call(sock, {"verb": "status"}, 20)
                self.assertEqual(st["state"], "poisoned", st)
                started = time.time()
                ex = client.call(sock, {"verb": "exit"}, 10)
                self.assertLess(time.time() - started, 5)
                self.assertFalse(ex["ok"], ex)
                self.assertEqual(procs[0].wait(10), 0)
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                    p.wait(10)


@unittest.skipUnless(torch is not None, "needs torch")
class LostRound(unittest.TestCase):
    def test_rank_that_loses_a_round_exits_non_zero_with_a_reason(self):
        """A rank other than 0 whose receive or send of a round fails (rank 0 gave up, or is gone)
        must not leave like a clean finish. Input (in a fresh interpreter): the CPU lab as rank 1
        of 2, whose next receive raises the error gloo gives for a closed peer. Expect: exit code
        hook.LOST_ROUND_EXIT and one stderr line naming the error. Before: the exception left
        serve(), and the test worker on Linux exited 0."""
        from fastrsi_megatron import hook
        code = ("import sys\n"
                "from fastrsi_megatron.test_robustness import LabFixture\n"
                "class Gone:\n"
                "    rank, world, group = 1, 2, None\n"
                "    def bcast(self, msg=None):\n"
                "        raise RuntimeError('Connection closed by peer')\n"
                "lab = LabFixture(None).lab\n"
                "lab.rank, lab.ranks = 1, Gone()\n"
                "lab.serve()\n"
                "sys.exit(0)\n")
        env = {**os.environ, "PYTHONPATH": os.path.dirname(HERE)}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, timeout=120)
        self.assertEqual(out.returncode, hook.LOST_ROUND_EXIT, out.stderr[-2000:])
        self.assertIn("rank 1: lost the lab's round with rank 0 (RuntimeError: Connection closed",
                      out.stderr)


@unittest.skipUnless(torch is not None, "needs torch")
class InstallPatchModules(unittest.TestCase):
    """install on modules outside hook.MODULES: a module the job never imported, or one without a
    source file, is a refusal in the interface's shape, and an installed module is covered by
    status and tamper checks."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        sys.path.insert(0, self.folder)
        for name, factor in (("mg_target_a", 2), ("mg_target_b", 3)):
            with open(os.path.join(self.folder, name + ".py"), "w") as source:
                source.write(f"def scale(x):\n    return x * {factor}\n")
        import mg_target_a
        import mg_target_b
        self.first, self.second = mg_target_a, mg_target_b

    def tearDown(self):
        sys.path.remove(self.folder)
        for name in ("mg_target_a", "mg_target_b"):
            sys.modules.pop(name, None)

    def test_unloaded_or_sourceless_module_is_refused_as_patch_invalid(self):
        """Input: a patch for `no.such.module`, then one for `sys` (loaded, no __file__).
        Expect: each ok False, outcome refused, codes [patch_invalid], every key the interface
        names, the module in the finding. Before the fix: KeyError, then AttributeError, out of
        verb(), and a reply without outcome, codes, id, mechanisms or findings."""
        lab = LabFixture(self).lab
        for name in ("no.such.module", "sys"):
            reply = lab.verb("install", {"verb": "install", "patch": {"files": {name: "x = 1\n"}}})
            self.assertEqual((reply["ok"], reply["outcome"], reply["codes"], reply["id"],
                              reply["mechanisms"]), (False, "refused", ["patch_invalid"], None, []))
            self.assertIn(repr(name), reply["findings"][0]["observed"])

    def test_mixed_patch_names_every_unloaded_module_and_writes_nothing(self):
        fx = LabFixture(self)
        files = {"mg_target_a": "def scale(x): return 99", "no.second": "", "no.first": ""}
        reply = fx.lab.verb("install", {"patch": {"files": files}})
        self.assertEqual(reply["codes"], ["patch_invalid"])
        self.assertEqual(reply["not_loaded"], ["no.first", "no.second"])
        self.assertEqual(self.first.scale(2), 4)

    def test_journaled_module_attribute_install_and_undo(self):
        fx = LabFixture(self)
        cell = "journal.set_attr(model[0][0], 'training', False)"
        patch = {"files": {}, "repairs": [{"cell": cell, "declares": []}]}
        installed = fx.lab.verb("install", {"patch": patch})
        self.assertEqual(installed["mechanisms"], ["R10 repair"], installed)
        self.assertFalse(fx.model[0].training)
        self.assertTrue(fx.lab.verb("uninstall", {})["ok"])
        self.assertTrue(fx.model[0].training)

    def test_invalid_files_type_is_a_structured_refusal(self):
        lab = LabFixture(self).lab
        reply = lab.verb("install", {"patch": {"files": ["not a mapping"]}})
        self.assertEqual(reply["codes"], ["patch_invalid"])

    def test_inject_target_names_one_rank_or_is_refused(self):
        """Input: targets None, 1, "1", "rank1", then "rank01", "rank2", "x", True, -1 on two
        ranks. Expect: None, 1, 1, 1, then a plain ValueError naming the valid targets."""
        from fastrsi_megatron import hook
        for target, rank in ((None, None), (1, 1), ("1", 1), ("rank1", 1)):
            self.assertEqual(hook.target_rank(target, 2), rank)
        for target in ("rank01", "rank2", "x", True, -1):
            with self.assertRaisesRegex(ValueError,
                                        r"names no rank: give 0\.\.1 or rank0\.\.rank1"):
                hook.target_rank(target, 2)

    def test_installed_module_is_covered_by_tamper_checks_at_every_depth(self):
        """Input: install an edit of mg_target_a, then one of mg_target_b (tracked one level
        up the stack), uninstall both, then change mg_target_a.scale's code outside the journal.
        Expect: status edited, then base with no poison after the uninstalls (no false tamper at
        depth 0), then poisoned by tampered. Before the fix the tamper went unseen (status
        stayed base): only hook.MODULES were fingerprinted."""
        lab = LabFixture(self).lab
        for name, factor in (("mg_target_a", 10), ("mg_target_b", 30)):
            reply = lab.verb("install", {"verb": "install", "patch": {"files": {
                name: f"def scale(x):\n    return x * {factor}\n"}}})
            self.assertTrue(reply["ok"], reply)
        self.assertEqual((self.first.scale(1), self.second.scale(1)), (10, 30))
        self.assertEqual(lab.verb("status", {"verb": "status"})["state"], "edited")
        for _ in range(2):
            self.assertTrue(lab.verb("uninstall", {"verb": "uninstall"})["ok"])
        status = lab.verb("status", {"verb": "status"})
        self.assertEqual((status["state"], status["poison"]), ("base", []))
        self.first.scale.__code__ = (lambda x: x * 999).__code__
        status = lab.verb("status", {"verb": "status"})
        self.assertEqual((status["state"], [found["check"] for found in status["poison"]]),
                         ("poisoned", ["tampered"]))


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
        scaler = fx.opt.grad_scaler
        fx.opt.grad_scaler = None
        refused = s.restore()
        self.assertFalse(refused["ok"])
        self.assertIn("optimizer 0: grad scaler presence changed", refused["groups"])
        fx.opt.grad_scaler = scaler
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

    def test_tracker_states_survive_a_step_after_restore(self):
        """Megatron's CudaRNGStatesTracker: get_states() returns its states, set_states() keeps the
        dict it is given and fork() writes new states into it. Input: S, restore, a step that
        forks the tracker, restore again. Expect: the tracker state equals S's both times. Before
        the fix the second restore brought back the step's state (the snapshot's own dict had been
        written by fork); the GPU run's digest showed it as an `rng` difference."""
        from fastrsi_megatron import state as ST

        class Tracker:                          # the semantics of megatron.core.tensor_parallel.random
            def __init__(self):
                self.states_ = {"model-parallel-rng": torch.ByteTensor([1, 2, 3])}

            def get_states(self):
                return {k: v for k, v in self.states_.items()}

            def set_states(self, states):
                self.states_ = states

            def step(self):
                self.states_["model-parallel-rng"] = self.states_["model-parallel-rng"] + 1
        t = Tracker()
        fx = LabFixture(self)
        with mock.patch.object(ST, "_tracker", return_value=t):
            s = ST.Snapshot([fx.model], fx.opt, fx.sched)
            for _ in range(2):
                self.assertTrue(s.restore()["ok"])
                self.assertEqual(t.states_["model-parallel-rng"].tolist(), [1, 2, 3])
                t.step()

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

    def test_digest_hashes_the_same_bytes_without_numpy(self):
        """numpy is optional for the lab, but tensor bytes were read through Tensor.numpy():
        without numpy every digest failed. Input: a lab's state, digested with numpy and with
        state.np set to None (the no-numpy path). Expect: both succeed and the tensor pieces are
        equal (rng differs: without numpy there is no numpy RNG state). Fails without the fix only
        where numpy is missing (torch-only), where the first digest is already the no-numpy one."""
        from fastrsi_megatron import state as ST
        fixture = LabFixture(self)
        with_numpy = ST.digest([fixture.model], fixture.opt, fixture.sched)
        with mock.patch.object(ST, "np", None):
            without = ST.digest([fixture.model], fixture.opt, fixture.sched)
        pieces = ("weights", "master_weights", "buffers", "optimizer", "scheduler")
        self.assertEqual({piece: without[piece] for piece in pieces},
                         {piece: with_numpy[piece] for piece in pieces})

    def test_no_numpy_bytes_are_read_in_bounded_pieces(self):
        """Without numpy the bytes are read with ctypes.string_at, whose size is a C int: one
        read of 2**32 + 5 bytes would hash 5 bytes. Input: 10 bytes read 3 at a time (the read
        size is a parameter, so no 4 GiB tensor is needed), and in one read. Expect: both equal
        sha256 of the same 10 bytes."""
        import hashlib
        from fastrsi_megatron import state as ST
        data = torch.arange(10, dtype=torch.uint8)
        want = hashlib.sha256(bytes(data.tolist())).hexdigest()
        with mock.patch.object(ST, "np", None):
            for bytes_per_read in (3, ST.BYTES_PER_READ):
                digest_state = hashlib.sha256()
                ST._feed(digest_state, data, bytes_per_read)
                self.assertEqual(digest_state.hexdigest(), want, bytes_per_read)

    def test_failed_digest_gives_no_hash(self):
        """A digest that fails on a rank must not return a hash: before the fix the verb hashed
        the ranks' [None] into a valid-looking sha256, so two failed digests (before and after a
        step) compared equal, the same as a bitwise round trip. Input: an optimizer state tensor
        on the meta device (no data, its digest raises). Expect: ok False, digest None, the
        rank's error."""
        fixture = LabFixture(self)
        master = fixture.opt.shard_fp32_from_float16_groups[0][0]
        fixture.opt.optimizer.state[master]["exp_avg"] = torch.zeros(3, device="meta")
        reply = fixture.lab.verb("digest", {"verb": "digest"})
        self.assertFalse(reply["ok"], reply)
        self.assertIsNone(reply["digest"])
        self.assertIn("meta", reply["error"])


class KeptBuffersLifetime(unittest.TestCase):
    """Kept buffers from a failed round live at module level (`gap._KEPT_BUFFERS`) for the
    process lifetime rather than disappearing if the `Ranks` instance is deleted."""

    def test_failed_round_buffers_persist_in_module_list_across_ranks_deletion(self):
        import gc
        import types
        import weakref

        class FakeTensor:
            pass

        class FakeWork:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail

            def wait(self, timeout=None):
                if self.should_fail:
                    raise RuntimeError("peer timed out")

        created_tensors = []
        created_works = []

        class FakeTorch:
            int64 = "int64"
            uint8 = "uint8"

            @staticmethod
            def tensor(data, dtype=None):
                t = FakeTensor()
                created_tensors.append(t)
                return t

            @staticmethod
            def frombuffer(buf, dtype=None):
                t = FakeTensor()
                created_tensors.append(t)
                return t

            @staticmethod
            def zeros(n, dtype=None):
                t = FakeTensor()
                created_tensors.append(t)
                return t

            @staticmethod
            def empty(n, dtype=None):
                t = FakeTensor()
                created_tensors.append(t)
                return t

        class FakeDist:
            @staticmethod
            def get_rank():
                return 0

            @staticmethod
            def get_world_size():
                return 2

            @staticmethod
            def new_group(backend=None, timeout=None):
                return object()

            @staticmethod
            def isend(tensor, dst=None, group=None, tag=None):
                w = FakeWork(should_fail=False)
                created_works.append(w)
                return w

            @staticmethod
            def irecv(tensor, src=None, group=None, tag=None):
                w = FakeWork(should_fail=True)
                created_works.append(w)
                return w

        fake_torch_pkg = types.ModuleType("torch")
        fake_torch_pkg.int64 = FakeTorch.int64
        fake_torch_pkg.uint8 = FakeTorch.uint8
        fake_torch_pkg.tensor = FakeTorch.tensor
        fake_torch_pkg.frombuffer = FakeTorch.frombuffer
        fake_torch_pkg.zeros = FakeTorch.zeros
        fake_torch_pkg.empty = FakeTorch.empty
        fake_torch_pkg.distributed = FakeDist

        before = len(gap._KEPT_BUFFERS)
        try:
            with mock.patch.dict(sys.modules, {"torch": fake_torch_pkg,
                                               "torch.distributed": FakeDist}):
                ranks = gap.Ranks(round_timeout_s=1.0)
                ranks_ref = weakref.ref(ranks)
                with self.assertRaises(gap.RoundFailed):
                    ranks.round({"op": "status"}, lambda msg: {"ok": True})
                self.assertFalse(hasattr(ranks, "abandoned"))
            work_refs = [weakref.ref(w) for w in created_works]
            tensor_refs = [weakref.ref(t) for t in created_tensors]
            self.assertGreater(len(work_refs), 0)
            self.assertGreater(len(tensor_refs), 0)
            created_works.clear()
            created_tensors.clear()
            del ranks
            gc.collect()
            self.assertIsNone(ranks_ref(), "Ranks instance should be collected after del")
            kept_slice = gap._KEPT_BUFFERS[before:]
            self.assertEqual(len(kept_slice), len(work_refs))
            for wr in work_refs:
                self.assertIsNotNone(wr(), "posted work must remain alive in gap._KEPT_BUFFERS")
            for tr in tensor_refs:
                self.assertIsNotNone(tr(), "posted tensor must remain alive in gap._KEPT_BUFFERS")
        finally:
            del gap._KEPT_BUFFERS[before:]
            gc.collect()


class HookExitAfterFailedRound(unittest.TestCase):
    """Pure unit tests for `hook.Lab.serve()` and `hook.install_hook()` on `exit`:
    after a failed round (or when `exit` itself fails), `exit` replies ok False, `serve()`
    returns `"exit_failed"`, and `install_hook()` skips `dist.barrier` and calls `os._exit(0)`."""

    _SCRIPT = """import json, os, sys, tempfile, threading, types
from unittest import mock
if {use_stub}:
    t = types.ModuleType('torch')
    t.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, synchronize=lambda: None)
    t.distributed = types.ModuleType('torch.distributed')
    t.distributed.barrier = lambda group=None: None
    t.is_tensor = lambda x: False
    sys.modules['torch'] = t
    sys.modules['torch.distributed'] = t.distributed
from fastrsi_megatron import client, gap, hook

def run_case(mode):
    barrier_calls = []
    exit_calls = []
    raw_calls = []
    serve_rets = []
    class FakeRanks:
        rank, world = 0, 2
        group = object()
        def __init__(self):
            self.failed = 'prior failure' if mode == 'earlier_failed' else None
        def round(self, msg, handle):
            if mode == 'earlier_failed':
                raise gap.RoundFailed('an earlier round failed: ' + self.failed)
            if mode == 'exit_fails' and msg.get('op') == 'exit':
                self.failed = 'exit: rank 1 lost'
                raise gap.RoundFailed(self.failed)
            return [handle(msg), {{'ok': True, 'rank': 1}}]
    fake_ranks = FakeRanks()
    class Tr:
        @staticmethod
        def train_step(*a, **k):
            raw_calls.append(a)
            return ({{'lm loss': 1.0}}, False, False, False, 0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        sock = os.path.join(d, 'lab.sock')
        ready = os.path.join(d, 'ready.json')
        fake_lab = hook.Lab.__new__(hook.Lab)
        fake_lab.ts, fake_lab.a, fake_lab.k = Tr.train_step, [], {{}}
        fake_lab.model, fake_lab.optimizer, fake_lab.scheduler = None, None, None
        fake_lab.ranks, fake_lab.rank = fake_ranks, 0
        fake_lab.steps_max, fake_lab.default_steps, fake_lab.warmup = 2, 1, 1
        fake_lab.faults, fake_lab.timings, fake_lab.setup_findings = {{}}, {{}}, []
        fake_lab.local = hook.C.LocalParticipant('rank0', scope={{}})
        fake_lab.host, fake_lab.replayed, fake_lab.batches = None, None, mock.Mock(items=None)
        if mode == 'non_round_poison':
            fake_lab._host().fault('tampered', 'hand edit', 'base')
        orig_serve = fake_lab.serve
        def traced_serve():
            ret = orig_serve()
            serve_rets.append(ret)
            return ret
        fake_lab.serve = traced_serve
        with mock.patch.dict(os.environ, {{'FASTRSI_SOCK': sock, 'FASTRSI_READY': ready}}), \\
             mock.patch.object(hook, 'Lab', return_value=fake_lab), \\
             mock.patch.object(hook.dist, 'barrier', side_effect=lambda group=None: barrier_calls.append(group)), \\
             mock.patch.object(hook.os, '_exit', side_effect=lambda code: (exit_calls.append(code), (_ for _ in ()).throw(SystemExit(code)))):
            hook.install_hook(Tr)
            def hit_exit():
                import time
                deadline = time.monotonic() + 10.0
                while True:
                    try:
                        return client.call(sock, {{'verb': 'exit'}}, 10)
                    except (FileNotFoundError, ConnectionRefusedError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
            res_box = []
            th = threading.Thread(target=lambda: res_box.append(hit_exit()), daemon=True)
            th.start()
            try:
                Tr.train_step()
            except SystemExit:
                pass
            th.join(10)
            return {{'reply_ok': res_box[0]['ok'], 'serve_ret': serve_rets[0], 'barriers': len(barrier_calls), 'exits': exit_calls, 'raw_calls': len(raw_calls)}}

print(json.dumps({{m: run_case(m) for m in ('earlier_failed', 'exit_fails', 'healthy', 'non_round_poison')}}))
"""

    def test_hook_exit_skips_barrier_only_when_round_failed(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([LIVELAB_DIR, LIVELAB])}
        script = self._SCRIPT.format(use_stub="True" if torch is None else "False")
        out = subprocess.run([sys.executable, "-W", "error::ResourceWarning", "-c", script],
                             capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(res["earlier_failed"], {"reply_ok": False, "serve_ret": "exit_failed", "barriers": 0, "exits": [0], "raw_calls": 0})
        self.assertEqual(res["exit_fails"], {"reply_ok": False, "serve_ret": "exit_failed", "barriers": 0, "exits": [0], "raw_calls": 0})
        self.assertEqual(res["healthy"], {"reply_ok": True, "serve_ret": "exit", "barriers": 1, "exits": [0], "raw_calls": 0})
        self.assertEqual(res["non_round_poison"], {"reply_ok": True, "serve_ret": "exit", "barriers": 1, "exits": [0], "raw_calls": 0})


if __name__ == "__main__":
    unittest.main()
