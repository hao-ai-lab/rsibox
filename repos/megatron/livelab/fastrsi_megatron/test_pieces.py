"""Golden CPU cases for the Megatron lab's pieces that need no GPU and no Megatron.

The adapter as a whole (hook.py on a real Megatron job) is proven by the conformance run on
GPU (receipt in ../README.md); these cases pin the three pieces a reader cannot see in it.
"""
import difflib
import json
import os
from pathlib import Path
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
# hook.py hosts the live-lab kernel from the framework's core/livelab, which importing the
# package puts on sys.path (FASTRSI_LIVELAB overrides it)
from fastrsi_megatron import LIVELAB, client, gap  # noqa: E402

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

    def test_source_lookups_see_the_overlay_text_not_the_disk_file(self):
        """inspect.getsource (and so @triton.jit, torch.compile's Dynamo and tracebacks) reads
        linecache, which reads the file on disk when it exists: an overlaid module would show the
        unedited source while it runs the edit. Input: pkg/mod.py `f() -> 1 # DISK`, its lines
        already in linecache (a zygote that imported it before the fork), an overlay with
        `f() -> 2 # OVERLAY`, and overlay.install() called a second time (zygote --refresh).
        Expect: f() == 2, getsource and linecache show the overlay line, and one finder."""
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "src", "pkg"))
            open(os.path.join(folder, "src", "pkg", "__init__.py"), "w").close()
            with open(os.path.join(folder, "src", "pkg", "mod.py"), "w") as source:
                source.write("def f():\n    return 1  # DISK\n")
            os.makedirs(os.path.join(folder, "ov"))
            with open(os.path.join(folder, "ov", "pkg.mod.py"), "w") as source:
                source.write("def f():\n    return 2  # OVERLAY\n")
            code = ("import inspect, json, linecache, sys, importlib.util\n"
                    "path = importlib.util.find_spec('pkg.mod').origin\n"
                    "linecache.getlines(path)\n"
                    "from fastrsi_megatron import overlay\n"
                    "overlay.install()\n"
                    "import pkg.mod as module\n"
                    "finders = [finder for finder in sys.meta_path\n"
                    "           if type(finder).__name__ == 'OverlayFinder']\n"
                    "print(json.dumps([module.f(), inspect.getsource(module.f),\n"
                    "                  linecache.getline(path, 2), len(finders)]))\n")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.path.dirname(HERE),
                                                                os.path.join(folder, "src")]),
                   "FASTRSI_OVERLAY": os.path.join(folder, "ov")}
            env.pop("FASTRSI_KERNEL", None)
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env=env)
            self.assertEqual(json.loads(run.stdout or "null"),
                             [2, "def f():\n    return 2  # OVERLAY\n",
                              "    return 2  # OVERLAY\n", 1], run.stderr)


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


class ConformanceJob(unittest.TestCase):
    def test_killed_job_leaves_no_folder_or_socket_and_ports_are_fresh(self):
        """The conformance lab starts a job per spawn, respawn and fresh check (c8 alone respawns
        four times): each left a fastrsi-mg-* folder and its socket behind, and every lab of one
        process reused one fixed port. Input: a job `echo started; exec sleep 30` and a file at its
        socket path, then kill(). Expect: the process is gone, its folder and socket removed, the
        log tail read before; two _free_port() calls give bindable ports."""
        from fastrsi_megatron import conformance_lab
        job = conformance_lab._Job("echo started; exec sleep 30", {}, "test")
        open(job.sock, "w").close()
        deadline = time.time() + 10
        while not job.tail() and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(job.tail(), ["started"])
        job.kill()
        self.assertIsNotNone(job.p.poll())
        self.assertFalse(os.path.exists(job.dir), job.dir)
        self.assertFalse(os.path.exists(job.sock), job.sock)
        for port in (conformance_lab._free_port(), conformance_lab._free_port()):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))


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


def _quantized_class():
    """A small stand-in for Transformer Engine's FP8/MXFP8 parameter: its bytes live in plain
    tensors (`_data` int8 codes, `_scale` one fp32 scale); reading it dequantizes, and copy_ into
    it quantizes again with a scale taken from the values, as TE does. Defined lazily (needs torch)."""

    class Quantized(torch.Tensor):
        @staticmethod
        def __new__(cls, data, scale):
            t = torch.Tensor._make_subclass(cls, torch.empty(data.shape, dtype=torch.float32))
            t._data, t._scale = data, scale
            return t

        def dequant(self):
            return self._data.float() * self._scale

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.copy_ and isinstance(args[0], cls):
                dst, src = args[0], args[1]
                src = src.dequant() if isinstance(src, cls) else src
                with torch.no_grad():
                    dst._scale.copy_(src.abs().max() / 127)
                    dst._data.copy_(torch.round(src / dst._scale).to(torch.int8))
                return dst
            if func is torch.Tensor.copy_:
                src = args[1].dequant() if isinstance(args[1], cls) else args[1]
                return torch.Tensor.copy_(args[0], src, *args[2:], **kwargs)
            unwrap = lambda x: x.dequant() if isinstance(x, cls) else x  # noqa: E731
            with torch._C.DisableTorchFunctionSubclass():
                return func(*[unwrap(a) for a in args], **{k: unwrap(v) for k, v in kwargs.items()})

    return Quantized


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
        self.assertFalse(s0.restore()["ok"])          # Adam's new state: refused, nothing written
        self.assertTrue(s0.restore(allow_extra=True)["ok"])   # the lab's start: kept, zeroed next
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

    def test_quantized_weights_come_back_byte_for_byte(self):
        """fp8_param runs keep each weight as a quantized tensor (TE's MXFP8Tensor) whose bytes
        sit in plain data and scale tensors. Input: a quantized weight with codes in [-50, 50]
        and scale 0.01, a snapshot, then a step that changes its codes and scale; restore.
        Expect: the codes and the scale are the snapshot's, byte for byte. Copying through the
        quantized tensor (what the snapshot did before) brings back the same values with a new
        scale (max / 127) and other codes, so a check would not start from the built weights.
        What it does not prove: that every quantized type keeps all its bytes in attributes."""
        from fastrsi_megatron import state as ST
        Q = _quantized_class()
        torch.manual_seed(0)
        data = torch.randint(-50, 51, (4, 8), dtype=torch.int8)
        scale = torch.tensor(0.01)
        q = Q(data.clone(), scale.clone())
        self.assertEqual([n for n, _ in ST.raw_parts(q)], ["._data", "._scale"])
        model = torch.nn.Module()
        model.register_buffer("w", q)
        opt = _MegatronLikeOptimizer(torch.nn.Linear(2, 2))
        snap = ST.Snapshot([model], opt, _Sched())
        with torch.no_grad():                       # what an optimizer step does to its bytes
            q._data.add_(3)
            q._scale.mul_(2)
        self.assertTrue(snap.restore()["ok"])
        self.assertTrue(torch.equal(q._data, data))
        self.assertTrue(torch.equal(q._scale, scale))

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
        self.assertIn(LIVELAB, sys.path)
        from fastrsi_megatron import hook
        f = hook.unknown_state_finding(["opt0.state.3.master_param", "opt0.state.4.master_param"])
        self.assertEqual(f["check"], "optimizer_state_unknown")
        self.assertEqual(f["observed"],
                         "state keys not reset: ['opt0.state.3.master_param', 'opt0.state.4.master_param']")
        self.assertEqual(f["n_unknown"], 2)
        json.dumps(f)   # findings travel in replies and receipts

    def test_lab_setup_freezes_batches_restores_start_state_and_records_unknown_state(self):
        """Lab._setup freezes the batches, snapshots S0, runs warm-up steps, restores S0, zeroes
        Adam moments, snapshots S, and records an optimizer_state_unknown finding when the first
        optimizer step creates a non-Adam state key. Input: a toy model, a Megatron-like optimizer
        whose step adds `master_param` to state, and a fake train step on CPU. Expect: the lab
        starts, restores the initial weights, zeroes the 6 Adam tensors, keeps `master_param` untouched,
        records `optimizer_state_unknown` in `setup_findings`, and repeats the same warm-up losses on
        `_restore() + _steps()`."""
        from unittest import mock
        from fastrsi_megatron import hook
        torch.manual_seed(0)
        batches = [torch.randn(4, 8) for _ in range(6)]
        model = torch.nn.Linear(8, 4)
        initial_weight = model.weight.detach().clone()
        optimizer = _MegatronLikeOptimizer(model)
        scheduler = _Sched()
        original_step = optimizer.step

        def step_with_master():
            original_step()
            first_param = optimizer.shard_fp32_from_float16_groups[0][0]
            optimizer.optimizer.state[first_param].setdefault("master_param", torch.full((2,), 7.0))
        optimizer.step = step_with_master

        def fake_train_step(forward_func, data_iterator, models, step_optimizer, step_scheduler, *rest, **kwargs):
            batch = next(data_iterator)
            models[0].zero_grad()
            loss = models[0](batch).square().mean()
            loss.backward()
            step_optimizer.step()
            step_scheduler.step(1)
            return ({"lm loss": loss.detach()}, False, False, False, 0, 1.25)

        fake_ranks = type("_FakeRanks", (), {"rank": 0, "world": 1})
        env = {"FASTRSI_WARMUP": "2", "FASTRSI_CHECK_STEPS_MAX": "4", "FASTRSI_CHECK_STEPS": "2"}
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(hook.G, "Ranks", return_value=fake_ranks()), \
             mock.patch.object(hook.Lab, "_mb", return_value=1):
            lab = hook.Lab(fake_train_step, [None, iter(batches), [model], optimizer, scheduler], {"iteration": 0})
            self.assertTrue(torch.equal(model.weight.data, initial_weight))
            self.assertEqual(scheduler.num_steps, 0)
            self.assertEqual(lab.timings["zeroed"], 3 * 2)  # weight + bias x (exp_avg, exp_avg_sq, step)
            self.assertEqual(len(lab.setup_findings), 1)
            finding = lab.setup_findings[0]
            self.assertEqual(finding["check"], "optimizer_state_unknown")
            self.assertEqual(finding["n_unknown"], 1)
            self.assertIn("opt0.state.0.master_param", finding["observed"])
            first_param = optimizer.shard_fp32_from_float16_groups[0][0]
            self.assertEqual(optimizer.optimizer.state[first_param]["master_param"].tolist(), [7.0, 7.0])
            first_check = lab._steps(2)
            lab._restore()
            second_check = lab._steps(2)
            self.assertEqual(first_check["loss"], second_check["loss"])
            status = lab.verb("status", {"verb": "status"})
            self.assertEqual(status["setup_findings"], [finding])
            self.assertEqual(status["timings"]["zeroed"], 6)


class ConformanceLabDiffsTest(unittest.TestCase):
    """Pure-CPU tests for unified `patch['diff']` support in `MegatronLab`."""

    @staticmethod
    def _make_diff(rel_path, old_src, new_src):
        return "".join(difflib.unified_diff(
            old_src.splitlines(True), new_src.splitlines(True),
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
        ))

    def test_install_uninstall_and_fresh_check_resolve_stacked_unified_diffs(self):
        from fastrsi_megatron import conformance_lab as CL

        with tempfile.TemporaryDirectory() as root:
            mlp_rel = CL.MLP.replace(".", "/") + ".py"
            att_rel = CL.ATT.replace(".", "/") + ".py"
            mlp_base = "def forward(x):\n    return x * 1\n"
            att_base = "def attend(x):\n    return x\n"
            for rel, txt in ((mlp_rel, mlp_base), (att_rel, att_base)):
                p = Path(root, rel)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(txt, encoding="utf-8")

            lab = CL.MegatronLab.__new__(CL.MegatronLab)
            lab.cmd = "true"
            lab.root = root
            lab.steps = 3
            lab.check_timeout = 30.0
            lab.receipts = None
            lab.poison = []
            lab._live_stack = [{}]
            lab.base = {CL.MLP: mlp_base, CL.ATT: att_base}

            sent = []

            def fake_call(req, timeout=None):
                sent.append(req)
                if req["verb"] == "install":
                    return {"ok": True, "outcome": "committed", "id": f"txn-{len(sent)}",
                            "codes": [], "mechanisms": ["R0 swap"], "findings": []}
                if req["verb"] == "uninstall":
                    return {"ok": True, "undone": ["txn-top"], "state": "edited", "findings": []}
                return {"ok": True}

            lab._call = fake_call

            mlp_v1 = mlp_base.replace("x * 1", "x * 2")
            mlp_v2 = mlp_v1.replace("x * 2", "x * 4")
            diff1 = self._make_diff(mlp_rel, mlp_base, mlp_v1)
            diff2 = self._make_diff(mlp_rel, mlp_v1, mlp_v2)
            combined = self._make_diff(mlp_rel, mlp_base, mlp_v2)

            # First diff installs cleanly and sends resolved `files` over the wire.
            r1 = lab.install({"diff": diff1})
            self.assertTrue(r1["ok"], r1)
            self.assertEqual(sent[-1]["patch"]["files"], {CL.MLP: mlp_v1})
            self.assertNotIn("diff", sent[-1]["patch"])

            # Stale diff (re-applying diff1 on top of v1) refuses locally with patch_invalid.
            calls_before = len(sent)
            stale = lab.install({"diff": diff1})
            self.assertEqual((stale["ok"], stale["outcome"], stale["codes"]), (False, "refused", ["patch_invalid"]))
            self.assertEqual(len(sent), calls_before)

            # Stacked diff2 applies cleanly against live v1 text.
            r2 = lab.install({"diff": diff2})
            self.assertTrue(r2["ok"], r2)
            self.assertEqual(sent[-1]["patch"]["files"], {CL.MLP: mlp_v2})

            # Uninstall pops one live-stack frame, so diff2 re-applies cleanly against v1.
            self.assertTrue(lab.uninstall()["ok"])
            r3 = lab.install({"diff": diff2})
            self.assertTrue(r3["ok"], r3)
            self.assertEqual(sent[-1]["patch"]["files"], {CL.MLP: mlp_v2})

            # fresh_check resolves unified diff against disk (base) into FASTRSI_OVERLAY.
            seen_overlay = {}

            class _FakeFreshJob:
                def __init__(self, _cmd, env, _tag):
                    ov_file = Path(env["FASTRSI_OVERLAY"], CL.MLP + ".py")
                    seen_overlay["mlp"] = ov_file.read_text(encoding="utf-8")
                    Path(env["FASTRSI_FRESH"]).write_text(
                        json.dumps({"ok": True, "output": {"loss": [1.0]}}), encoding="utf-8"
                    )

                def wait_exit(self):
                    return 0

                def tail(self, _n=8):
                    return []

                def kill(self):
                    return None

            with mock.patch.object(CL, "_Job", _FakeFreshJob):
                fresh_ok = lab.fresh_check({"diff": combined})
                self.assertTrue(fresh_ok["ok"], fresh_ok)
                self.assertEqual(seen_overlay["mlp"], mlp_v2)
                fresh_bad = lab.fresh_check({"diff": diff2})
                self.assertFalse(fresh_bad["ok"])

    def test_fresh_check_retries_once_on_eaddrinuse_only(self):
        from fastrsi_megatron import conformance_lab as CL

        lab = CL.MegatronLab.__new__(CL.MegatronLab)
        lab.cmd = "true"
        lab.root = "/unused"
        lab.steps = 3
        lab.check_timeout = 30.0
        lab.receipts = None
        lab.poison = []
        lab._live_stack = [{}]
        lab.base = {}

        def run_scenario(outcomes):
            events = []
            ports = iter([29501, 29502, 29503])

            class _SeqJob:
                def __init__(self, _cmd, env, _tag):
                    idx = len(events)
                    rc, tail, write_out = outcomes[idx]
                    self.rc, self._tail = rc, tail
                    events.append({"port": env["PORT"], "killed": False})
                    self._idx = idx
                    if write_out:
                        Path(env["FASTRSI_FRESH"]).write_text(
                            json.dumps({"ok": True, "output": {"loss": [1.5]}}), encoding="utf-8"
                        )

                def wait_exit(self):
                    return self.rc

                def tail(self, n=30):
                    return list(self._tail[-n:])

                def kill(self):
                    events[self._idx]["killed"] = True

            with mock.patch.object(CL, "_Job", _SeqJob), \
                 mock.patch.object(CL, "_free_port", side_effect=lambda: next(ports)):
                res = lab.fresh_check({"files": {}})
            return res, events

        # 1. Transient "Address already in use" kills attempt 1 and succeeds on attempt 2 with a new port.
        res_ok, ev_ok = run_scenario([
            (1, ["OSError: [Errno 98] Address already in use"], False),
            (0, [], True),
        ])
        self.assertTrue(res_ok["ok"], res_ok)
        self.assertEqual(res_ok["output"], {"loss": [1.5]})
        self.assertEqual(ev_ok, [{"port": "29501", "killed": True}, {"port": "29502", "killed": True}])

        # 2. Persistent EADDRINUSE stops after exactly 1 retry (2 total attempts, both killed).
        res_twice, ev_twice = run_scenario([
            (1, ["socket error: EADDRINUSE"], False),
            (1, ["socket error: EADDRINUSE"], False),
        ])
        self.assertFalse(res_twice["ok"])
        self.assertIn("EADDRINUSE", res_twice["error"])
        self.assertEqual(len(ev_twice), 2)
        self.assertTrue(all(e["killed"] for e in ev_twice))

        # 3. Real training/code error never retries.
        res_err, ev_err = run_scenario([
            (1, ["RuntimeError: model exploded"], False),
        ])
        self.assertFalse(res_err["ok"])
        self.assertIn("model exploded", res_err["error"])
        self.assertEqual(ev_err, [{"port": "29501", "killed": True}])

        # 4. Worker-side EADDRINUSE sitting above a 25-line torch.distributed.elastic
        # ChildFailedError footer still triggers the single retry (tail(80) sees past the box).
        elastic_footer_25 = [
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
            "============================================================",
            *[f"elastic_detail_line_{i}" for i in range(22)],
            "============================================================",
        ]
        self.assertEqual(len(elastic_footer_25), 25)
        eaddr_with_box = (
            [f"startup_line_{i}" for i in range(10)]
            + ["[rank0]: OSError: [Errno 98] Address already in use"]
            + elastic_footer_25
        )
        res_box_retry, ev_box_retry = run_scenario([
            (1, eaddr_with_box, False),
            (0, [], True),
        ])
        self.assertTrue(res_box_retry["ok"], res_box_retry)
        self.assertEqual(res_box_retry["output"], {"loss": [1.5]})
        self.assertEqual(
            ev_box_retry,
            [{"port": "29501", "killed": True}, {"port": "29502", "killed": True}],
        )

        # 5. Worker-side RuntimeError above torch.distributed.elastic's ~20-line ChildFailedError
        # box does not retry and is surfaced in the returned error via tail[-25:], while earlier
        # preamble lines beyond the last 25 are omitted.
        elastic_footer_20 = [
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
            "============================================================",
            *[f"elastic_detail_line_{i}" for i in range(17)],
            "============================================================",
        ]
        self.assertEqual(len(elastic_footer_20), 20)
        runtime_with_box = (
            [f"startup_line_{i}" for i in range(15)]
            + ["[rank1]: RuntimeError: worker rank1 exploded"]
            + elastic_footer_20
        )
        res_box_err, ev_box_err = run_scenario([
            (1, runtime_with_box, False),
        ])
        self.assertFalse(res_box_err["ok"])
        self.assertEqual(ev_box_err, [{"port": "29501", "killed": True}])
        self.assertIn("worker rank1 exploded", res_box_err["error"])
        self.assertIn("ChildFailedError:", res_box_err["error"])
        self.assertNotIn("startup_line_0", res_box_err["error"])


@unittest.skipUnless(torch is not None, "needs torch")
class PipelineAndVppTest(unittest.TestCase):
    def test_merge_check_output_prefers_non_none_stage_and_frozen_batches_vpp(self):
        """PP > 1 takes loss/grad_norm from the stage that computed them; VPP lists replay cleanly."""
        from fastrsi_megatron import hook, state

        merged = hook._merge_check_output([
            {"ok": True, "rank": 0, "loss": [None, None], "grad_norm": [1.5, 1.4]},
            {"ok": True, "rank": 1, "loss": [None, None], "grad_norm": [None, None]},
            {"ok": True, "rank": 2, "loss": [9.1, 9.0], "grad_norm": [1.5, 1.4]},
        ])
        self.assertEqual(merged, {"loss": [9.1, 9.0], "grad_norm": [1.5, 1.4]})

        fallback = hook._merge_check_output([
            {"ok": True, "rank": 0, "loss": [None], "grad_norm": [None]},
        ])
        self.assertEqual(fallback, {"loss": [None], "grad_norm": [None]})

        self.assertIsNone(state.FrozenBatches(None, 2).iterator())
        fb_single = state.FrozenBatches(iter([10, 20, 30]), 2)
        self.assertEqual(list(fb_single.iterator()), [10, 20])

        fb_vpp = state.FrozenBatches([iter([1, 2, 3]), None, iter(["a", "b", "c"])], 2)
        it_vpp = fb_vpp.iterator()
        self.assertEqual(len(it_vpp), 3)
        self.assertEqual(list(it_vpp[0]), [1, 2])
        self.assertIsNone(it_vpp[1])
        self.assertEqual(list(it_vpp[2]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
