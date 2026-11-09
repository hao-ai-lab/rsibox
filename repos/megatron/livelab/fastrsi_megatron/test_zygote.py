"""Golden CPU cases for the zygote: real fork, real Unix socket, real child processes; the
preloaded "heavy" module is a temp module that records the pid that imported it. No torch, no
GPU: what the GPU run adds (CUDA usable after the fork, Megatron start) is in the receipt
(../README.md)."""
import json
import os
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
json.dump(out, open(os.path.join(os.environ["OUT"], "r%s.json" % os.environ["RANK"]), "w"))
if os.environ.get("FAIL_RANK") == os.environ["RANK"]:
    sys.exit(7)
if os.environ.get("HANG"):
    import time; time.sleep(60)
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

    def test_ranks_get_torchrun_env_and_reuse_the_zygote_imports(self):
        """Input: zygote preloaded `heavy` (records the importing pid); start 2 ranks with
        MY_FLAG=on. Expect: ranks 0 and 1, WORLD_SIZE 2, MASTER_PORT 29999, MY_FLAG on, the
        script's argv, each rank its own session, and heavy.PID == the zygote's pid (the rank
        did not import it again). A torchrun + exec start would show heavy.PID == rank pid."""
        evs = self.start(env={"MY_FLAG": "on"})
        self.assertEqual([e["event"] for e in evs], ["forked", "exited"])
        self.assertEqual(evs[1]["rc"], [0, 0])
        for r in (0, 1):
            o = json.load(open(os.path.join(self.d, f"r{r}.json")))
            self.assertEqual((o["RANK"], o["LOCAL_RANK"], o["WORLD_SIZE"], o["MASTER_PORT"],
                              o["MY_FLAG"]), (str(r), str(r), "2", "29999", "on"))
            self.assertEqual(o["argv"], ["--flag", "x"])
            self.assertEqual(o["heavy_pid"], self.ready["pid"])
            self.assertEqual(o["sid"], o["pid"])
            self.assertEqual(o["OMP_NUM_THREADS"], "1")     # torchrun's default for 2 ranks
            self.assertEqual(o["TORCHINDUCTOR_COMPILE_THREADS"], "1")
        self.assertGreater(self.ready["gc_frozen"], 0)

    def test_one_failing_rank_stops_the_others_and_the_zygote_serves_again(self):
        """Input: rank 1 exits 7 while rank 0 sleeps 60 s. Expect: rank 0 terminated (rc -15)
        within seconds, rc [-15, 7]; then a second start on the same zygote succeeds.
        Without the stop policy the request would wait the full 60 s."""
        t = time.time()
        evs = self.start(env={"FAIL_RANK": "1", "HANG": "1"})
        self.assertEqual(evs[-1]["rc"], [-15, 7])
        self.assertLess(time.time() - t, 20)
        self.assertEqual(self.start()[-1]["rc"], [0, 0])

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
            out = json.load(open(os.path.join(self.d, "r%d.json" % r)))
            self.assertEqual(out["heavy_pid"], -1)

    def test_each_rank_is_pinned_to_its_cpu_list(self):
        """A fork keeps the zygote's CPU affinity; ranks on GPUs of different sockets need
        their own. Input: cpus ["0", "0"] for 2 ranks (CPU 0 always exists). Expect: each
        rank's sched_getaffinity == [0]; without `cpus`, the zygote's full set."""
        self.start(cpus=["0", "0"])
        for r in (0, 1):
            self.assertEqual(json.load(open(os.path.join(self.d, f"r{r}.json")))["cpus"], [0])
        self.start()
        full = sorted(os.sched_getaffinity(0))
        self.assertEqual(json.load(open(os.path.join(self.d, "r1.json")))["cpus"], full)


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
        """Pre-fork thread clamping must default OMP, OpenBLAS, MKL, and TorchInductor thread
        variables to 1 when unset without overwriting caller-provided values."""
        target_env = {"OMP_NUM_THREADS": "4"}
        zygote.clamp_pre_fork_threads(target_env)
        self.assertEqual(target_env["OMP_NUM_THREADS"], "4")
        self.assertEqual(target_env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(target_env["MKL_NUM_THREADS"], "1")
        self.assertEqual(target_env["TORCHINDUCTOR_COMPILE_THREADS"], "1")

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


if __name__ == "__main__":
    unittest.main()
