"""Golden CPU cases for the zygote: real fork, real Unix socket, real child processes; the
preloaded "heavy" module is a temp module that records the pid that imported it. No torch, no
GPU: what the GPU run adds (CUDA usable after the fork, Megatron start) is in the receipt
(../README.md)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fastrsi_megatron import zygote  # noqa: E402

HEAVY = "import os, time\nPID = os.getpid()\nT = time.time()\n"
SCRIPT = """import heavy, json, os, sys
out = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MY_FLAG",
                                      "OMP_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")}
out.update(heavy_pid=heavy.PID, pid=os.getpid(), sid=os.getsid(0), argv=sys.argv[1:],
           cpus=sorted(os.sched_getaffinity(0)))
with open(os.path.join(os.environ["OUT"], "r%s.json" % os.environ["RANK"]), "w") as f:
    json.dump(out, f)
if os.environ.get("IGNORE_TERM"):
    import signal
    if os.environ["RANK"] == "0":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open(os.path.join(os.environ["OUT"], "ignoring"), "w").close()
    else:
        while not os.path.exists(os.path.join(os.environ["OUT"], "ignoring")):
            import time; time.sleep(0.01)
if os.environ.get("FAIL_RANK") == os.environ["RANK"]:
    sys.exit(7)
if os.environ.get("HANG"):
    import time; time.sleep(60)
if os.environ.get("ORPHAN"):    # a child in a session of its own, left behind when the rank exits
    import subprocess
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                           start_new_session=True)
    with open(os.path.join(os.environ["OUT"], "orphan%s" % os.environ["RANK"]), "w") as f:
        f.write(str(kid.pid))
"""


class Zygote(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(os.path.join(self.d, "heavy.py"), "w") as f:
            f.write(HEAVY)
        with open(os.path.join(self.d, "train.py"), "w") as f:
            f.write(SCRIPT)
        self.sock = os.path.join(self.d, "z.sock")
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + self.d,
                   FASTRSI_ZYGOTE_FORKSAFE="0")
        self.z = subprocess.Popen([sys.executable, "-m", "fastrsi_megatron.zygote", "serve",
                                   "--sock", self.sock, "--preload", "heavy"],
                                  stdout=subprocess.PIPE, text=True, env=env)
        self.addCleanup(self.z.stdout.close)
        self.ready = json.loads(self.z.stdout.readline())

    def tearDown(self):
        if self.z.poll() is None:
            list(zygote.request(self.sock, {"op": "stop"}))
        self.z.wait(10)

    def start(self, nproc=2, env=None, **kw):
        req = {"argv": [os.path.join(self.d, "train.py"), "--flag", "x"], "nproc": nproc,
               "port": 29999, "entry": "script", "log_dir": self.d,
               "env": dict({"OUT": self.d}, **(env or {})), "t0": time.time(), **kw}
        return list(zygote.request(self.sock, req))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_ranks_get_torchrun_env_and_reuse_the_zygote_imports(self):
        """Input: zygote preloaded `heavy` (records the importing pid); start 2 ranks with
        MY_FLAG=on. Expect: ranks 0 and 1, WORLD_SIZE 2, MASTER_PORT 29999, MY_FLAG on, the
        script's argv, each rank its own session, and heavy.PID == the zygote's pid (the rank
        did not import it again). A torchrun + exec start would show heavy.PID == rank pid."""
        evs = self.start(env={"MY_FLAG": "on"})
        self.assertEqual([e["event"] for e in evs], ["forked", "exited"])
        self.assertEqual(evs[1]["rc"], [0, 0])
        for r in (0, 1):
            o = json.loads(Path(self.d, f"r{r}.json").read_text())
            self.assertEqual((o["RANK"], o["LOCAL_RANK"], o["WORLD_SIZE"], o["MASTER_PORT"],
                              o["MY_FLAG"]), (str(r), str(r), "2", "29999", "on"))
            self.assertEqual(o["argv"], ["--flag", "x"])
            self.assertEqual(o["heavy_pid"], self.ready["pid"])
            self.assertEqual(o["sid"], o["pid"])
            self.assertEqual(o["OMP_NUM_THREADS"], "1")     # torchrun's default for 2 ranks
            self.assertEqual(
                o["TORCHINDUCTOR_COMPILE_THREADS"],
                str(zygote.default_compile_threads(2)),
            )
        self.assertGreater(self.ready["gc_frozen"], 0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_one_failing_rank_stops_the_others_and_the_zygote_serves_again(self):
        """Input: rank 1 exits 7 while rank 0 sleeps 60 s. Expect: rank 0 terminated (rc -15)
        within seconds, rc [-15, 7]; then a second start on the same zygote succeeds.
        Without the stop policy the request would wait the full 60 s."""
        t = time.time()
        evs = self.start(env={"FAIL_RANK": "1", "HANG": "1"})
        self.assertEqual(evs[-1]["rc"], [-15, 7])
        self.assertLess(time.time() - t, 20)
        self.assertEqual(self.start()[-1]["rc"], [0, 0])

    @unittest.skipUnless(sys.platform.startswith("linux"), "rank supervision requires Linux")
    def test_failed_rank_escalates_when_peer_ignores_term(self):
        events = self.start(env={"FAIL_RANK": "1", "HANG": "1", "IGNORE_TERM": "1"})
        self.assertEqual(events[-1]["rc"], [-9, 7])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: /proc, child subreaper")
    def test_an_orphan_that_left_the_rank_group_is_reaped_after_the_job(self):
        """A rank starts a child in a session of its own (out of the rank's process group, so the
        guard's group kill misses it) and exits 0; the child is adopted by the zygote (child
        subreaper). Expect: after `exited`, the child is gone (reap_all finds it by parent pid)
        and the zygote serves again. Before the fix reap_all matched the process group and left
        it running."""
        events = self.start(nproc=1, env={"ORPHAN": "1"})
        self.assertEqual(events[-1]["rc"], [0])
        with open(os.path.join(self.d, "orphan0")) as pid_file:
            orphan = int(pid_file.read())

        def gone():
            try:
                with open(f"/proc/{orphan}/stat") as stat_file:
                    return stat_file.read().rsplit(")", 1)[1].split()[0] == "Z"
            except OSError:
                return True
        deadline = time.time() + 15
        while not gone() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(gone(), f"orphan {orphan} still running")
        self.assertEqual(self.start(nproc=1)[-1]["rc"], [0])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_edited_source_is_a_finding_unless_accepted(self):
        """Input: heavy.py edited after the zygote imported it. Expect: the request is refused
        with a "stale" finding naming heavy.py (the ranks would run the old code); with
        accept=["stale"] it forks and the rank still sees the zygote's import."""
        time.sleep(0.01)
        with open(os.path.join(self.d, "heavy.py"), "w") as f:
            f.write(HEAVY + "X = 2\n")
        evs = self.start()
        self.assertEqual(evs[0]["event"], "refused")
        self.assertEqual(evs[0]["findings"][0]["kind"], "stale")
        self.assertTrue(evs[0]["findings"][0]["files"][0].endswith("heavy.py"))
        self.assertEqual(self.start(accept=["stale"])[-1]["rc"], [0, 0])

    def test_import_time_env_change_is_a_finding_nccl_is_not(self):
        """Input: a request with NCCL_ALGO (read at communicator init, after the fork) and
        TORCH_NCCL_ASYNC_ERROR_HANDLING (read at process-group creation) and NVTE_FUSED_ATTN
        (Transformer Engine may read it at import). Expect: refused, finding "env" lists only
        NVTE_FUSED_ATTN."""
        evs = self.start(env={"NCCL_ALGO": "Ring", "NVTE_FUSED_ATTN": "0",
                              "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3"})
        self.assertEqual(evs[0]["event"], "refused")
        self.assertEqual(evs[0]["findings"][0]["keys"], ["NVTE_FUSED_ATTN"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_overlay_of_a_preloaded_module_is_refused(self):
        """A fresh check of a code edit sets FASTRSI_OVERLAY; overlay.py can only replace a
        module before its first import, and the zygote imported `heavy` already. Input: an
        overlay dir with heavy.py. Expect: refused with an "overlay" finding naming heavy (a
        silent run of the unedited code is the failure this prevents); an overlay of a module
        the zygote did not import forks normally."""
        ov = os.path.join(self.d, "ov")
        os.makedirs(ov)
        with open(os.path.join(ov, "heavy.py"), "w") as f:
            f.write(HEAVY)
        evs = self.start(env={"FASTRSI_OVERLAY": ov})
        self.assertEqual(evs[0]["event"], "refused")
        self.assertEqual(evs[0]["findings"][0]["kind"], "overlay")
        self.assertEqual(evs[0]["findings"][0]["modules"], ["heavy"])
        os.rename(os.path.join(ov, "heavy.py"), os.path.join(ov, "light.py"))
        self.assertEqual(self.start(env={"FASTRSI_OVERLAY": ov})[-1]["rc"], [0, 0])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_code_restart_reimports_the_refreshed_package_from_the_overlay(self):
        """Input: the zygote preloaded `heavy` (PID = the zygote's pid); an overlay replaces `heavy`
        with PID = -1. Without refresh the request is refused ("overlay"). With refresh=["heavy"],
        each rank drops `heavy` and imports the overlay text: heavy_pid is -1 in both ranks.
        Proves a code edit to a preloaded package runs in a zygote rank; not that Megatron
        re-imports cleanly (that is the GPU receipt)."""
        ov = os.path.join(self.d, "ov")
        os.makedirs(ov)
        with open(os.path.join(ov, "heavy.py"), "w") as f:
            f.write("PID = -1\n")
        ev = self.start(env={"FASTRSI_OVERLAY": ov})
        self.assertEqual(ev[0]["event"], "refused")
        self.assertEqual(ev[0]["findings"][0]["kind"], "overlay")
        ev = self.start(env={"FASTRSI_OVERLAY": ov}, refresh=["heavy"])
        self.assertEqual([e["event"] for e in ev], ["forked", "exited"], ev)
        self.assertEqual(ev[-1]["rc"], [0, 0])
        for r in (0, 1):
            out = json.loads(Path(self.d, f"r{r}.json").read_text())
            self.assertEqual(out["heavy_pid"], -1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: needs os.sched_getaffinity and /proc (rank pinning, process tree)")
    def test_each_rank_is_pinned_to_its_cpu_list(self):
        """A fork keeps the zygote's CPU affinity; ranks on GPUs of different sockets need
        their own. Input: cpus ["0", "0"] for 2 ranks (CPU 0 always exists). Expect: each
        rank's sched_getaffinity == [0]; without `cpus`, the zygote's full set."""
        self.start(cpus=["0", "0"])
        for r in (0, 1):
            self.assertEqual(json.loads(Path(self.d, f"r{r}.json").read_text())["cpus"], [0])
        self.start()
        full = sorted(os.sched_getaffinity(0))
        self.assertEqual(json.loads(Path(self.d, "r1.json").read_text())["cpus"], full)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: /proc and child subreaper")
    def test_zygote_leading_own_process_group_reaps_job_without_touching_pgrp_sibling(self):
        """When the zygote leads its own process group (`os.getpgrp() == pid`), a failing job
        and its `setsid()` orphans are reaped without signaling the zygote itself or an
        unrelated sibling sharing the zygote's process group."""
        sock2 = os.path.join(self.d, "z-pgrp.sock")
        out2 = os.path.join(self.d, "pgrp-out")
        os.makedirs(out2)
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + self.d,
                   FASTRSI_ZYGOTE_FORKSAFE="0")
        z_as_leader = (
            "import json, os, subprocess, sys\n"
            "os.setsid()\n"
            "sib = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setpgid(0, os.getppid())\n"
            "    os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "from fastrsi_megatron import zygote\n"
            "print(json.dumps({'z_pid': os.getpid(), 'z_pgrp': os.getpgrp(),\n"
            "                  'pre_sib': sib.pid, 'pgrp_other': pid}), flush=True)\n"
            "try:\n"
            f"    zygote.serve({sock2!r}, ['heavy'], forksafe=False)\n"
            "finally:\n"
            "    sib_alive = sib.poll() is None\n"
            "    other_alive = True\n"
            "    try:\n"
            "        os.kill(pid, 0)\n"
            "    except ProcessLookupError:\n"
            "        other_alive = False\n"
            "    sib.kill(); sib.wait()\n"
            "    try:\n"
            "        os.kill(pid, 9); os.waitpid(pid, 0)\n"
            "    except OSError:\n"
            "        pass\n"
            "    print(json.dumps({'sib_alive': sib_alive, 'other_alive': other_alive}), flush=True)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", z_as_leader],
                                stdout=subprocess.PIPE, text=True, env=env)
        try:
            meta = json.loads(proc.stdout.readline())
            self.assertEqual(meta["z_pid"], meta["z_pgrp"])
            ready = json.loads(proc.stdout.readline())
            self.assertEqual(ready["event"], "ready")
            req_orphan = {"argv": [os.path.join(self.d, "train.py")], "nproc": 1, "port": 29997,
                          "entry": "script", "log_dir": out2,
                          "env": {"OUT": out2, "ORPHAN": "1"}, "t0": time.time()}
            evs_orphan = list(zygote.request(sock2, req_orphan))
            self.assertEqual(evs_orphan[-1]["rc"], [0])
            orphan = int(Path(out2, "orphan0").read_text())
            deadline = time.time() + 10
            while os.path.exists(f"/proc/{orphan}/stat") and time.time() < deadline:
                time.sleep(0.05)
            self.assertFalse(os.path.exists(f"/proc/{orphan}/stat"))
            req_fail = {"argv": [os.path.join(self.d, "train.py")], "nproc": 2, "port": 29997,
                        "entry": "script", "log_dir": out2,
                        "env": {"OUT": out2, "FAIL_RANK": "1", "HANG": "1"}, "t0": time.time()}
            evs_fail = list(zygote.request(sock2, req_fail))
            self.assertEqual(evs_fail[-1]["rc"], [-15, 7])
            self.assertEqual([e["event"] for e in zygote.request(sock2, {"op": "ping"})], ["pong"])
            self.assertEqual([e["event"] for e in zygote.request(sock2, {"op": "stop"})], ["stopped"])
            summary = json.loads(proc.stdout.readline())
            self.assertTrue(summary["sib_alive"])
            self.assertTrue(summary["other_alive"])
            self.assertEqual(proc.wait(10), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(10)
            proc.stdout.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: /proc and child subreaper")
    def test_job_started_helpers_are_reaped_while_prejob_helper_and_child_survive(self):
        """A pre-job helper started by the zygote (and a child that helper spawns during the job)
        survives `handle()`'s `reap_all(keep=before)`, while helpers/grandchildren started by a
        rank during the job (in-group and `SIGTERM`-ignoring `setsid()`) are killed and reaped."""
        sock3 = os.path.join(self.d, "z-helpers.sock")
        out3 = os.path.join(self.d, "helpers-out")
        os.makedirs(out3)
        rank_script = os.path.join(self.d, "spawn_helpers.py")
        Path(rank_script).write_text(
            "import json, os, signal, subprocess, sys, time\n"
            "in_grp = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "out_code = (\n"
            "    'import os, signal, subprocess, sys, time\\n'\n"
            "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
            "    'gc = subprocess.Popen([sys.executable, \"-c\", \"import time; time.sleep(60)\"], start_new_session=True)\\n'\n"
            "    'with open(os.environ[\"GC_FILE\"], \"w\") as fh:\\n'\n"
            "    '    fh.write(str(gc.pid))\\n'\n"
            "    'time.sleep(60)\\n'\n"
            ")\n"
            "gc_file = os.path.join(os.environ['OUT'], 'grandchild.pid')\n"
            "out_env = {**os.environ, 'GC_FILE': gc_file}\n"
            "out_grp = subprocess.Popen([sys.executable, '-c', out_code], start_new_session=True, env=out_env)\n"
            "while not os.path.exists(gc_file) or not os.path.getsize(gc_file):\n"
            "    time.sleep(0.01)\n"
            "with open(gc_file) as gfh:\n"
            "    gc_pid = int(gfh.read())\n"
            "with open(os.path.join(os.environ['OUT'], 'job_helpers.json'), 'w') as fh:\n"
            "    json.dump({'in_grp': in_grp.pid, 'out_grp': out_grp.pid, 'grandchild': gc_pid}, fh)\n",
            encoding="utf-8",
        )
        helper_code = (
            "import os, subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "with open(sys.argv[1], 'w') as fh:\n"
            "    fh.write(f'{os.getpid()},{kid.pid}')\n"
            "try:\n"
            "    time.sleep(60)\n"
            "finally:\n"
            "    kid.kill(); kid.wait()\n"
        )
        z_runner = (
            "import json, os, subprocess, sys, time\n"
            "from fastrsi_megatron import zygote\n"
            f"helper_pid_file = {os.path.join(out3, 'pre_helper.txt')!r}\n"
            f"helper = subprocess.Popen([sys.executable, '-c', {helper_code!r}, helper_pid_file])\n"
            "while not os.path.exists(helper_pid_file) or not os.path.getsize(helper_pid_file):\n"
            "    time.sleep(0.01)\n"
            "try:\n"
            f"    zygote.serve({sock3!r}, ['heavy'], forksafe=False)\n"
            "finally:\n"
            "    with open(helper_pid_file) as hfh:\n"
            "        h_pid, hk_pid = [int(x) for x in hfh.read().split(',')]\n"
            "    h_alive = helper.poll() is None\n"
            "    hk_alive = os.path.exists(f'/proc/{hk_pid}/stat')\n"
            "    helper.kill(); helper.wait()\n"
            "    try:\n"
            "        os.kill(hk_pid, 9)\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "    print(json.dumps({'h_alive': h_alive, 'hk_alive': hk_alive}), flush=True)\n"
        )
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + self.d,
                   FASTRSI_ZYGOTE_FORKSAFE="0")
        proc = subprocess.Popen([sys.executable, "-c", z_runner],
                                stdout=subprocess.PIPE, text=True, env=env)
        try:
            ready = json.loads(proc.stdout.readline())
            self.assertEqual(ready["event"], "ready")
            req = {"argv": [rank_script], "nproc": 1, "port": 29996,
                   "entry": "script", "log_dir": out3, "env": {"OUT": out3}, "t0": time.time()}
            evs = list(zygote.request(sock3, req))
            self.assertEqual(evs[-1]["rc"], [0])
            pids = json.loads(Path(out3, "job_helpers.json").read_text())
            for name, pid in pids.items():
                deadline = time.time() + 10
                while os.path.exists(f"/proc/{pid}/stat") and time.time() < deadline:
                    time.sleep(0.05)
                self.assertFalse(os.path.exists(f"/proc/{pid}/stat"), f"{name} ({pid}) survived")
            self.assertEqual([e["event"] for e in zygote.request(sock3, {"op": "stop"})], ["stopped"])
            res = json.loads(proc.stdout.readline())
            self.assertTrue(res["h_alive"])
            self.assertTrue(res["hk_alive"])
            self.assertEqual(proc.wait(10), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(10)
            proc.stdout.close()


class Pieces(unittest.TestCase):
    def test_unpatch_puts_back_the_patched_cuda_functions(self):
        """The patches keep CUDA out of the zygote's imports; a rank initializes CUDA and should
        call torch's own functions. Input: a fake cuda with the device-properties stand-in
        installed. Expect: unpatch() names it and get_device_properties is the original again;
        a function nobody patched is left alone."""
        class Cuda:
            def is_initialized(self):
                return False
            def get_device_properties(self, d=None):
                return "orig"
            def current_device(self):
                return 7
        c = Cuda()
        orig = c.get_device_properties
        zygote.install_properties(c, probe=lambda: [{"multi_processor_count": 1}])
        self.assertIsNot(c.get_device_properties, orig)
        self.assertEqual(zygote.unpatch(c), ["get_device_properties"])
        self.assertEqual(c.get_device_properties(0), "orig")
        self.assertEqual(c.current_device(), 7)

    def test_children_are_found_by_parent_pid_not_process_group(self):
        """reap_all() kills what _children() returns. Each rank calls setsid(), so its process
        group is its own pid, never the zygote's: matching on the group finds no rank and no
        adopted orphan, and they outlive the job. Input: a /proc-format directory with the
        zygote (1000), a rank (2001, its own session), an orphan the subreaper took in (3005,
        comm `a b) (c` with spaces and parentheses) and an unrelated process (4000, group
        1000). Expect: children [2001, 3005]; the tree of 1000 holds all three. Proves the
        parse of real stat lines; not the Linux run (ZygoteSupervision cases)."""
        lines = {1000: "1000 (python3) S 1 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100",
                 2001: "2001 (python3) S 1000 2001 2001 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 200",
                 3005: "3005 (a b) (c) R 1000 3005 3005 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 3",
                 4000: "4000 (other) S 1 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 400"}
        with tempfile.TemporaryDirectory() as proc:
            for pid, line in lines.items():
                os.makedirs(os.path.join(proc, str(pid)))
                with open(os.path.join(proc, str(pid), "stat"), "w") as stat_file:
                    stat_file.write(line + "\n")
            os.makedirs(os.path.join(proc, "self"))
            self.assertEqual(zygote.stat_ppid(lines[3005]), 1000)
            self.assertEqual(sorted(zygote._children(1000, proc)), [2001, 3005])
            self.assertEqual(sorted(zygote._tree(1000, proc)), [1000, 2001, 3005])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only: /proc")
    def test_children_the_zygote_had_before_the_job_survive_the_reap(self):
        """reap_all kills every child it finds by parent pid. A process the zygote itself started
        before the job (a helper) must survive it. Input (in a fresh interpreter, so the reap
        touches only its own children): a helper child, then a job child, then
        reap_all(keep={helper}). Expect: the job child reaped, the helper still running."""
        code = ("import json, os, signal, subprocess, sys, time\n"
                "from fastrsi_megatron import zygote\n"
                "sleep = [sys.executable, '-c', 'import time; time.sleep(60)']\n"
                "helper = subprocess.Popen(sleep)\n"
                "finished = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(17)'])\n"
                "os.waitid(os.P_PID, finished.pid, os.WEXITED | os.WNOWAIT)\n"
                "job = subprocess.Popen(sleep)\n"
                "reaped = zygote.reap_all(timeout=10, keep={helper.pid, finished.pid})\n"
                "alive = helper.poll() is None\n"
                "helper.kill(); helper.wait()\n"
                "print(json.dumps([job.pid in reaped, alive, finished.wait()]))\n")
        env = {**os.environ, "PYTHONPATH": os.path.dirname(HERE), "FASTRSI_ZYGOTE_FORKSAFE": "0"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                             timeout=60)
        self.assertEqual(json.loads(out.stdout or "null"), [True, True, 17], out.stderr)

    def test_auto_numa_bind_returns_the_cpus_forksafe_bound(self):
        """forksafe.bind_rank_numa_and_cpus answers {"bound_cpus", "numa_node", "source"}; bind()
        read "bound" and "cpus", so an automatic NUMA pin always reported None. Input:
        FASTRSI_ZYGOTE_AUTO_NUMA=1 and forksafe's answer for a GPU local to CPUs 0-3 (the sysfs
        read itself is forksafe's, tested in core/livelab). Expect: [0, 1, 2, 3]; with no CPUs
        found, None."""
        from unittest import mock
        answer = {"bound_cpus": [0, 1, 2, 3], "numa_node": 0, "source": "sysfs"}
        with mock.patch.dict(os.environ, {"FASTRSI_ZYGOTE_AUTO_NUMA": "1"}), \
                mock.patch.object(zygote.forksafe, "bind_rank_numa_and_cpus", return_value=answer):
            self.assertEqual(zygote.bind({}, 0), [0, 1, 2, 3])
            answer.update(bound_cpus=[], source="none")
            self.assertIsNone(zygote.bind({}, 0))

    def test_code_restart_unbinds_a_refreshed_submodule_from_its_parent(self):
        """`--refresh pkg.sub` with `pkg` kept loaded: `from pkg import sub` returns the parent's
        attribute when it exists, so without an unbind it keeps the zygote's old module and the
        overlay never runs. Input (in a fresh interpreter, as a forked rank): pkg/sub.py `VAL = 1`
        imported, an overlay `pkg.sub.py` with `VAL = 2`, drop_modules for refresh ["pkg.sub"],
        then the overlay finder. Expect: pkg.sub dropped, `from pkg import sub` gives VAL 2, and
        pkg stays the same module object."""
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "src", "pkg"))
            open(os.path.join(folder, "src", "pkg", "__init__.py"), "w").close()
            with open(os.path.join(folder, "src", "pkg", "sub.py"), "w") as source:
                source.write("VAL = 1\n")
            os.makedirs(os.path.join(folder, "ov"))
            with open(os.path.join(folder, "ov", "pkg.sub.py"), "w") as source:
                source.write("VAL = 2\n")
            code = ("import json, sys, pkg, pkg.sub\n"
                    "from fastrsi_megatron import overlay, zygote\n"
                    "old = pkg\n"
                    "dropped = zygote.drop_modules({'refresh': ['pkg.sub']})\n"
                    "overlay.install()\n"
                    "from pkg import sub\n"
                    "print(json.dumps(['pkg.sub' in dropped, sub.VAL,\n"
                    "                  sys.modules['pkg'] is old]))\n")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.path.dirname(HERE),
                                                                os.path.join(folder, "src")]),
                   "FASTRSI_OVERLAY": os.path.join(folder, "ov"), "FASTRSI_ZYGOTE_FORKSAFE": "0"}
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env=env)
            self.assertEqual(json.loads(out.stdout or "null"), [True, 2, True], out.stderr)

    def test_cpu_list_parses_ranges(self):
        self.assertEqual(zygote.cpu_list("0-3,8,10-11"), {0, 1, 2, 3, 8, 10, 11})

    def test_set_threads_applies_omp_num_threads_to_torch_in_the_rank(self):
        """torch reads OMP_NUM_THREADS and torch._inductor.config reads
        TORCHINDUCTOR_COMPILE_THREADS when they load in the zygote, so a forked rank would
        keep an unclamped pool (every core or 32 Inductor workers) without explicit clamping.
        Input: fake torch and torch._inductor.config modules with OMP_NUM_THREADS=1 and
        TORCHINDUCTOR_COMPILE_THREADS=1. Expect set_num_threads(1) and compile_threads == 1;
        with no valid value, nothing is called."""
        recorded_calls = []

        class FakeTorch:
            def set_num_threads(self, num_threads):
                recorded_calls.append(num_threads)

        class FakeInductorConfig:
            compile_threads = 32

        fake_inductor_config = FakeInductorConfig()
        old_torch = sys.modules.get("torch")
        old_inductor = sys.modules.get("torch._inductor.config")
        sys.modules["torch"] = FakeTorch()
        sys.modules["torch._inductor.config"] = fake_inductor_config
        try:
            self.assertEqual(
                zygote.set_threads(
                    {"OMP_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"}
                ),
                1,
            )
            self.assertEqual(fake_inductor_config.compile_threads, 1)
            self.assertIsNone(
                zygote.set_threads(
                    {"OMP_NUM_THREADS": "invalid", "TORCHINDUCTOR_COMPILE_THREADS": "invalid"}
                )
            )
        finally:
            if old_torch is not None:
                sys.modules["torch"] = old_torch
            else:
                del sys.modules["torch"]
            if old_inductor is not None:
                sys.modules["torch._inductor.config"] = old_inductor
            else:
                del sys.modules["torch._inductor.config"]
        self.assertEqual(recorded_calls, [1])

    def test_clamp_pre_fork_threads_sets_default_single_thread_env_vars(self):
        """Pre-fork thread clamping must default OMP, OpenBLAS, and MKL thread variables to 1
        while allowing multi-threaded TorchInductor compile (PRELOAD_COMPILE_THREADS = 8)
        during preload/warmup without overwriting caller-provided values."""
        target_env = {"OMP_NUM_THREADS": "4"}
        zygote.clamp_pre_fork_threads(target_env)
        self.assertEqual(target_env["OMP_NUM_THREADS"], "4")
        self.assertEqual(target_env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(target_env["MKL_NUM_THREADS"], "1")
        self.assertEqual(
            target_env["TORCHINDUCTOR_COMPILE_THREADS"],
            str(zygote.PRELOAD_COMPILE_THREADS),
        )
        self.assertEqual(zygote.default_compile_threads(2, cpu_count=64), 8)
        self.assertEqual(zygote.default_compile_threads(8, cpu_count=64), 4)
        self.assertEqual(zygote.default_compile_threads(16, cpu_count=16), 2)

    def test_quiesce_compile_pools_after_warmup_resets_threads_and_shuts_down_pools(self):
        """After preload or warmup compilation completes, quiesce_compile_pools_after_warmup()
        must reset torch's intra-op thread pool to 1, set inductor compile_threads to 1, and
        shut down any active AsyncCompile thread/process pools so steady-state training steps
        run with zero background compile threads."""
        recorded_torch_threads = []
        shutdown_calls = []

        class FakeTorch:
            def set_num_threads(self, num_threads):
                recorded_torch_threads.append(num_threads)

        class FakeInductorConfig:
            compile_threads = 8

        class FakePool:
            def __init__(self, pool_name):
                self.pool_name = pool_name

            def shutdown(self, wait=False):
                shutdown_calls.append((self.pool_name, wait))

        class FakeAsyncCompile:
            _pool = FakePool("class_thread_pool")
            _process_pool = FakePool("class_process_pool")

        class FakeAsyncCompileModule:
            __name__ = "torch._inductor.async_compile"
            AsyncCompile = FakeAsyncCompile
            _pool = FakePool("module_thread_pool")

        fake_async_module = FakeAsyncCompileModule()
        old_torch = sys.modules.get("torch")
        old_inductor = sys.modules.get("torch._inductor.config")
        old_async = sys.modules.get("torch._inductor.async_compile")
        sys.modules["torch"] = FakeTorch()
        sys.modules["torch._inductor.config"] = FakeInductorConfig()
        sys.modules["torch._inductor.async_compile"] = fake_async_module
        try:
            summary = zygote.quiesce_compile_pools_after_warmup()
            self.assertEqual(summary["torch_threads"], 1)
            self.assertEqual(summary["compile_threads"], 1)
            self.assertEqual(len(summary["pools_shutdown"]), 3)
            self.assertIsNone(FakeAsyncCompile._pool)
            self.assertIsNone(FakeAsyncCompile._process_pool)
            self.assertIsNone(fake_async_module._pool)
        finally:
            if old_torch is not None:
                sys.modules["torch"] = old_torch
            else:
                del sys.modules["torch"]
            if old_inductor is not None:
                sys.modules["torch._inductor.config"] = old_inductor
            else:
                del sys.modules["torch._inductor.config"]
            if old_async is not None:
                sys.modules["torch._inductor.async_compile"] = old_async
            else:
                del sys.modules["torch._inductor.async_compile"]
        self.assertEqual(recorded_torch_threads, [1])
        self.assertEqual(
            shutdown_calls,
            [
                ("module_thread_pool", False),
                ("class_thread_pool", False),
                ("class_process_pool", False),
            ],
        )

    def test_script_imports_reads_top_level_and_guarded_imports(self):
        """pretrain_gpt.py style: plain imports, from-imports, imports under `if`/`try`; a
        relative import is skipped. Expect them in source order, no duplicates."""
        src = ("import os\nfrom megatron.training import pretrain\nif True:\n    import a.b\n"
               "try:\n    import c\nexcept ImportError:\n    import d\nfrom . import e\nimport os\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        self.assertEqual(zygote.script_imports(f.name),
                         ["os", "megatron.training", "a.b", "c", "d"])

    def test_preload_stops_when_an_import_initializes_cuda(self):
        """A module whose import initializes CUDA must stop the zygote with its name, since no
        forked rank could use the GPU. The CUDA check is injected (no GPU in CI)."""
        d = tempfile.mkdtemp()
        for m in ("zy_ok", "zy_cuda"):
            open(os.path.join(d, m + ".py"), "w").close()
        sys.path.insert(0, d)
        with self.assertRaises(SystemExit) as e:
            zygote.preload(["zy_ok", "zy_cuda"], check=lambda: "zy_cuda" in sys.modules)
        self.assertIn("importing zy_cuda initialized CUDA", str(e.exception))


    def test_device_properties_answer_from_the_probe_until_cuda_is_initialized(self):
        """flashinfer reads get_device_properties(0).multi_processor_count at import, which
        initializes CUDA in the zygote. Input: a fake torch.cuda (not initialized, original
        raises) and a probe reporting 2 devices with 152 and 148 SMs. Expect: device None, 0,
        "cuda:1" answer 152, 152, 148 without the original; index 2 and any call after CUDA
        is initialized go to the original. Without install the first call raises."""
        class Cuda:
            init = False
            def is_initialized(self):
                return self.init
            def get_device_properties(self, d=None):
                raise RuntimeError("original: would initialize CUDA")
        c = Cuda()
        with self.assertRaises(RuntimeError):
            c.get_device_properties(0)
        r = zygote.install_properties(c, probe=lambda: [
            {"multi_processor_count": 152, "major": 10}, {"multi_processor_count": 148, "major": 10}])
        self.assertEqual(r["devices"], 2)
        self.assertEqual([c.get_device_properties(d).multi_processor_count
                          for d in (None, 0, "cuda:1")], [152, 152, 148])
        with self.assertRaises(RuntimeError):
            c.get_device_properties(2)
        c.init = True
        with self.assertRaises(RuntimeError):
            c.get_device_properties(0)

    def test_zygote_leading_own_process_group_excludes_self_and_pgrp_peers(self):
        """When the zygote leads its own process group (`pgrp == pid`), `_children`, `_tree`,
        and `kill_tree` match on parent PID, never matching the zygote itself or an unrelated
        peer in the zygote's process group."""
        from unittest import mock
        lines = {1000: "1000 (python3) S 1 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100",
                 1500: "1500 (helper) S 1000 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 150",
                 2001: "2001 (rank0) S 1000 2001 2001 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 200",
                 4000: "4000 (sibling) S 1 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 400"}
        with tempfile.TemporaryDirectory() as proc_dir:
            for pid, line in lines.items():
                os.makedirs(os.path.join(proc_dir, str(pid)))
                Path(proc_dir, str(pid), "stat").write_text(line + "\n", encoding="utf-8")
            self.assertEqual(zygote._children(1000, proc_dir), [1500, 2001])
            self.assertEqual(zygote._tree(2001, proc_dir), [2001])
            killed_pids, killed_pgs = [], []
            with mock.patch.object(zygote.os, "kill", side_effect=lambda p, s: killed_pids.append((p, s))), \
                 mock.patch.object(zygote.os, "killpg", side_effect=lambda g, s: killed_pgs.append((g, s))):
                zygote.kill_tree(2001, proc_root=proc_dir)
            self.assertEqual(killed_pids, [(2001, zygote.signal.SIGTERM)])
            self.assertEqual(killed_pgs, [(2001, zygote.signal.SIGTERM)])

    def test_zygote_as_pid_1_children_reap_and_sigterm_handling(self):
        """As PID 1 (`/proc/1/stat` has `ppid == 0`), `_children(1)` excludes PID 1 itself,
        `reap_all` kills and reaps job ranks/orphans without signaling PID 1, and `serve()`
        installs (and restores) a `SIGTERM` handler when running as PID 1."""
        import signal
        from unittest import mock
        lines = {1: "1 (zygote) S 0 1 1 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 1",
                 2: "2 (rank0) S 1 2 2 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 20",
                 3: "3 (guard) S 2 2 2 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 30",
                 4: "4 (orphan) S 1 4 4 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 40"}
        with tempfile.TemporaryDirectory() as proc_dir:
            for pid, line in lines.items():
                os.makedirs(os.path.join(proc_dir, str(pid)))
                Path(proc_dir, str(pid), "stat").write_text(line + "\n", encoding="utf-8")
            self.assertEqual(zygote._children(1, proc_dir), [2, 4])
            self.assertEqual(zygote._tree(1, proc_dir), [1, 2, 4, 3])
            killed, reaped_state = [], set()

            def fake_kill(pid, sig):
                killed.append(pid)

            def fake_waitpid(pid, flags):
                if pid in (2, 4) and pid not in reaped_state:
                    reaped_state.add(pid)
                    import shutil
                    shutil.rmtree(os.path.join(proc_dir, str(pid)), ignore_errors=True)
                    return pid, 0
                raise ChildProcessError()

            with mock.patch.object(zygote.os, "getpid", return_value=1), \
                 mock.patch.object(zygote.os, "kill", side_effect=fake_kill), \
                 mock.patch.object(zygote.os, "killpg"), \
                 mock.patch.object(zygote.os, "waitpid", side_effect=fake_waitpid):
                reaped = zygote.reap_all(timeout=1.0, keep=(), proc_root=proc_dir)
            self.assertEqual(sorted(reaped), [2, 4])
            self.assertNotIn(1, killed)
            self.assertIn(3, killed)

        # Verify serve() installs a SIGTERM exit handler when pid == 1 and restores it on stop.
        with tempfile.TemporaryDirectory() as d:
            sock = os.path.join(d, "pid1.sock")
            installed = []
            orig_signal = signal.signal

            def spy_signal(sig, handler):
                installed.append((sig, handler))
                return orig_signal(sig, handler)

            class FakeConn:
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False
                def settimeout(self, t):
                    pass
                def makefile(self):
                    import io
                    return io.StringIO('{"op": "stop"}\n')
                def sendall(self, data):
                    pass

            class FakeSock:
                def bind(self, s):
                    pass
                def listen(self, n):
                    pass
                def accept(self):
                    return FakeConn(), None
                def close(self):
                    pass

            with mock.patch.object(zygote.os, "getpid", return_value=1), \
                 mock.patch.object(zygote.os, "unlink"), \
                 mock.patch.object(zygote, "preload", return_value={}), \
                 mock.patch.object(zygote, "sources", return_value={}), \
                 mock.patch.object(zygote.socket, "socket", return_value=FakeSock()), \
                 mock.patch.object(zygote.signal, "getsignal", return_value=signal.SIG_DFL), \
                 mock.patch.object(zygote.signal, "signal", side_effect=spy_signal), \
                 mock.patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()):
                zygote.serve(sock, [], forksafe=False)
            self.assertEqual(len(installed), 2)
            self.assertEqual(installed[0][0], signal.SIGTERM)
            self.assertTrue(callable(installed[0][1]))
            self.assertEqual(installed[1], (signal.SIGTERM, signal.SIG_DFL))
            with self.assertRaises(SystemExit) as se:
                installed[0][1](signal.SIGTERM, None)
            self.assertEqual(se.exception.code, 128 + signal.SIGTERM)

    def test_proc_from_another_pid_namespace_translates_nspid_and_ignores_foreign_proc(self):
        """When `/proc` belongs to an ancestor PID namespace (`NSpid` present), `_ppids`,
        `_children`, and `_tree` translate only processes in our PID namespace (plus unshared
        descendants) to local PIDs and exclude host/sibling processes. When `/proc` is from an
        untranslatable foreign namespace, foreign PIDs are never returned or signaled."""
        from unittest import mock
        my_pid = os.getpid()

        # 1. Ancestor-namespace /proc with NSpid + ns/pid:
        with tempfile.TemporaryDirectory() as anc:
            os.symlink("50000", os.path.join(anc, "self"))
            entries = {
                1: ("1", "0", "pid:[111]"),                               # host init
                50000: (f"50000\t{my_pid}", "1", "pid:[999]"),             # zygote
                50001: ("50001\t2001", "50000", "pid:[999]"),             # rank 0
                50002: ("50002\t3005", "50001", "pid:[999]"),             # rank 0 child
                50003: ("50003\t2001", "49999", "pid:[777]"),             # sibling ns collision
                50004: ("50004\t3006\t1", "50001", "pid:[888]"),          # deeper unshared child
                50005: ("50005\t4000", "1", "pid:[111]"),                 # unrelated host proc
            }
            for opid, (nspid, oppid, ns) in entries.items():
                pdir = os.path.join(anc, str(opid))
                os.makedirs(os.path.join(pdir, "ns"))
                Path(pdir, "status").write_text(f"Pid:\t{opid}\nPPid:\t{oppid}\nNSpid:\t{nspid}\n",
                                                encoding="utf-8")
                os.symlink(ns, os.path.join(pdir, "ns", "pid"))
            self.assertEqual(zygote._ppids(anc), {my_pid: 0, 2001: my_pid, 3005: 2001, 3006: 2001})
            self.assertEqual(zygote._children(proc_root=anc), [2001])
            self.assertEqual(zygote._tree(my_pid, proc_root=anc), [my_pid, 2001, 3005, 3006])

        # 2. Untranslatable foreign /proc (colliding PIDs, no matching NSpid):
        with tempfile.TemporaryDirectory() as foreign:
            os.symlink("99999", os.path.join(foreign, "self"))
            os.makedirs(os.path.join(foreign, "77777"))
            Path(foreign, "77777", "stat").write_text(
                f"77777 (host) S {my_pid} 77777 77777 0 -1\n", encoding="utf-8"
            )
            self.assertEqual(zygote._ppids(foreign), {})
            self.assertEqual(zygote._children(proc_root=foreign), [])
            self.assertEqual(zygote._tree(2001, proc_root=foreign), [2001])
            killed, waited = [], []
            with mock.patch.object(zygote.os, "kill", side_effect=lambda p, s: killed.append(p)), \
                 mock.patch.object(zygote.os, "killpg"), \
                 mock.patch.object(zygote.os, "waitpid",
                                   side_effect=lambda p, f: (waited.append(p), (p, 0))[1] if p == 2001 else (_ for _ in ()).throw(ChildProcessError())):
                reaped = zygote.reap_all(timeout=0.5, keep=(), proc_root=foreign, known=[2001])
            self.assertNotIn(77777, killed)
            self.assertNotIn(77777, waited)
            self.assertEqual(killed, [2001])
            self.assertEqual(reaped, [2001])

        # 3. Real-procfs markers (`stat` file present) where `self` is absent (child-ns /proc):
        with tempfile.TemporaryDirectory() as child_ns:
            Path(child_ns, "stat").write_text("cpu 0 0 0 0\n", encoding="utf-8")
            os.makedirs(os.path.join(child_ns, "1"))
            Path(child_ns, "1", "stat").write_text(f"1 (other) S {my_pid} 1 1 0 -1\n", encoding="utf-8")
            self.assertEqual(zygote._ppids(child_ns), {})
            self.assertEqual(zygote._children(proc_root=child_ns), [])

    def test_translated_ppids_without_ns_pid_isolates_sibling_namespace_collisions(self):
        """When `/proc/self/ns/pid` is unreadable (`self_ns is None`), `_translated_ppids()`
        seeds `outer_to_local` solely with `self` and translates only proven `ppid` descendants,
        never admitting sibling-namespace processes whose local PID collides with `my_pid` or a rank."""
        my_pid = os.getpid()
        with tempfile.TemporaryDirectory() as anc:
            os.symlink("50000", os.path.join(anc, "self"))
            entries = {
                1: ("1", "0"),                               # host init
                50000: (f"50000\t{my_pid}", "1"),            # zygote (self)
                50001: ("50001\t2001", "50000"),             # our rank 0
                50002: ("50002\t3005", "50001"),             # our rank 0 child
                50004: ("50004\t3006\t1", "50001"),          # our deeper unshared child
                60000: (f"60000\t{my_pid}", "49999"),        # sibling ns colliding with my_pid
                60001: ("60001\t20", "60000"),               # sibling ns child (must not appear as child of my_pid)
                60002: ("60002\t2001", "49999"),             # sibling ns colliding with rank 0 local pid
            }
            for opid, (nspid, oppid) in entries.items():
                pdir = os.path.join(anc, str(opid))
                os.makedirs(pdir)
                Path(pdir, "status").write_text(f"Pid:\t{opid}\nPPid:\t{oppid}\nNSpid:\t{nspid}\n",
                                                encoding="utf-8")
            self.assertEqual(zygote._ppids(anc), {my_pid: 0, 2001: my_pid, 3005: 2001, 3006: 2001})
            self.assertEqual(zygote._children(proc_root=anc), [2001])
            self.assertEqual(zygote._tree(my_pid, proc_root=anc), [my_pid, 2001, 3005, 3006])

    def test_reap_unusable_proc_signals_all_targets_up_front_under_one_shared_deadline(self):
        """`_reap_unusable_proc()` must `SIGKILL` all non-kept known ranks before waiting, and
        bound waiting for stuck ranks to a single shared `timeout` rather than `len(known) * timeout`."""
        from unittest import mock
        events = []
        now = [0.0]

        def fake_kill(pid, sig):
            events.append(("kill", pid, sig))

        def fake_waitpid(pid, flags):
            events.append(("wait", pid))
            if pid == 2003:
                return 2003, 0
            if pid in (2001, 2004):
                return 0, 0
            raise ChildProcessError()

        def fake_sleep(dt):
            now[0] += dt

        with tempfile.TemporaryDirectory() as foreign:
            os.symlink("99999", os.path.join(foreign, "self"))
            with mock.patch.object(zygote.os, "kill", side_effect=fake_kill), \
                 mock.patch.object(zygote.os, "killpg"), \
                 mock.patch.object(zygote.os, "waitpid", side_effect=fake_waitpid), \
                 mock.patch.object(zygote.time, "monotonic", side_effect=lambda: now[0]), \
                 mock.patch.object(zygote.time, "sleep", side_effect=fake_sleep):
                reaped = zygote.reap_all(
                    timeout=0.06, keep=(2002,), proc_root=foreign, known=[2001, 2002, 2003, 2004]
                )
        self.assertEqual(reaped, [2003])
        kill_pids = [e[1] for e in events if e[0] == "kill"]
        self.assertEqual(kill_pids, [2001, 2003, 2004])
        first_wait = next(i for i, e in enumerate(events) if e[0] == "wait")
        self.assertEqual([e[1] for e in events[:first_wait]], [2001, 2003, 2004])
        self.assertLessEqual(now[0], 0.08)

    def test_reap_all_refreshes_left_before_logging_reap_timeout(self):
        """When the final child is reaped right as the deadline expires, `reap_all()` must
        refresh `left` before logging `reap_timeout`, while still logging for truly stuck children."""
        import shutil
        from unittest import mock
        lines = {1000: "1000 (zygote) S 1 1000 1000 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 1",
                 2001: "2001 (rank0) S 1000 2001 2001 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 20"}
        with tempfile.TemporaryDirectory() as proc_dir:
            for pid, line in lines.items():
                os.makedirs(os.path.join(proc_dir, str(pid)))
                Path(proc_dir, str(pid), "stat").write_text(line + "\n", encoding="utf-8")

            def reap_and_remove(pid, flags):
                shutil.rmtree(os.path.join(proc_dir, str(pid)), ignore_errors=True)
                return pid, 0

            with mock.patch.object(zygote.os, "getpid", return_value=1000), \
                 mock.patch.object(zygote.os, "kill"), \
                 mock.patch.object(zygote.os, "killpg"), \
                 mock.patch.object(zygote.os, "waitpid", side_effect=reap_and_remove), \
                 mock.patch.object(zygote, "log") as log_mock:
                reaped = zygote.reap_all(timeout=0.0, keep=(), proc_root=proc_dir)
            self.assertEqual(reaped, [2001])
            log_mock.assert_not_called()

            # A child that does NOT exit by deadline still emits `reap_timeout`.
            os.makedirs(os.path.join(proc_dir, "2002"))
            Path(proc_dir, "2002", "stat").write_text(
                "2002 (stuck) S 1000 2002 2002 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 21\n",
                encoding="utf-8",
            )
            with mock.patch.object(zygote.os, "getpid", return_value=1000), \
                 mock.patch.object(zygote.os, "kill"), \
                 mock.patch.object(zygote.os, "killpg"), \
                 mock.patch.object(zygote.os, "waitpid", return_value=(0, 0)), \
                 mock.patch.object(zygote, "log") as log_mock:
                reaped = zygote.reap_all(timeout=0.0, keep=(), proc_root=proc_dir)
            self.assertEqual(reaped, [])
            log_mock.assert_called_once_with("reap_timeout", left=[2002])


if __name__ == "__main__":
    unittest.main()
