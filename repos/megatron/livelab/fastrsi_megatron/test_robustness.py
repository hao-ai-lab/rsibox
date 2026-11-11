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
        self.assertTrue(_wait(lambda: os.path.exists(self.sock), 120), "lab never ready")
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

    def test_rank_dying_while_rank_0_is_busy_fails_the_round_within_the_bound(self):
        """Input: rank 1 exits at its check step (crash fault) while rank 0 sleeps 5 s in its own
        step; round timeout 3 s. Expect: the check answers ok False within 5 + 3 s and some
        slack (15 s), then the lab is poisoned by participant_failed. With replies on a gloo
        collective, rank 0 gave no answer in 45 s."""
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
        fx = LabFixture(self)
        with_numpy = ST.digest([fx.model], fx.opt, fx.sched)
        with mock.patch.object(ST, "np", None):
            without = ST.digest([fx.model], fx.opt, fx.sched)
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
        fx = LabFixture(self)
        master = fx.opt.shard_fp32_from_float16_groups[0][0]
        fx.opt.optimizer.state[master]["exp_avg"] = torch.zeros(3, device="meta")
        reply = fx.lab.verb("digest", {"verb": "digest"})
        self.assertFalse(reply["ok"], reply)
        self.assertIsNone(reply["digest"])
        self.assertIn("meta", reply["error"])


if __name__ == "__main__":
    unittest.main()
