"""Golden cases for the rsibox launchers and CLI (CPU).

The local launcher runs for real: a daemon in its own process, driven by the CLI only. Docker,
gVisor and Slurm are checked as the commands and specs they generate. What these do not prove:
that an engine boots in a container, under gVisor or under Slurm; those are GPU receipts.

    python3 -m unittest discover -s rsibox -t . -v      # from core/livelab
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from rsibox import client as C
from rsibox import gvisor as G
from rsibox import launch as L
from rsibox.test_rsibox import LIVELAB, edited_ops


class LocalLauncher(unittest.TestCase):
    def test_up_then_the_cli_then_down_in_real_processes(self):
        """The README quick start: a daemon in its own process, driven by the CLI only."""
        state = tempfile.mkdtemp(prefix="rsibox-up-")
        self.addCleanup(shutil.rmtree, state, True)
        record = L.up({"launcher": "local", "lab": "toy", "state": state})
        self.addCleanup(L.down, state)
        self.assertTrue(record["ok"], record)
        self.assertEqual(os.stat(record["box_file"]).st_mode & 0o777, 0o600)     # the token is private

        env = {**os.environ, "RSIBOX": record["box_file"], "PYTHONPATH": LIVELAB}

        def cli(*args):
            out = subprocess.run([sys.executable, "-m", "rsibox", *args], env=env, capture_output=True,
                                 text=True, cwd=LIVELAB, timeout=60)
            return out.returncode, out.stdout

        self.assertEqual(cli("wait", "--timeout", "30")[0], 0)
        code, text = cli("source", "toy.ops")
        self.assertEqual(code, 0)
        path = os.path.join(state, "ops.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(edited_ops(text))
        code, out = cli("install", "--file", f"toy.ops={path}")
        self.assertEqual((code, json.loads(out)["outcome"]), (0, "committed"))
        self.assertTrue(json.loads(cli("check")[1])["proven"])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("def broken(:\n")
        code, out = cli("install", "--file", f"toy.ops={path}")
        self.assertEqual(code, 2, out)                             # a refused install exits 2
        down = L.down(state)
        self.assertTrue(down["close"]["ok"], down)
        self.assertEqual(C.Client.from_box_file(record["box_file"], timeout=2).info()["http"], 0)


class Plans(unittest.TestCase):
    SPEC = {"lab": "vllm", "real": True, "state": "/shared/box1", "repo": "/src/rsibox",
            "env": {"RSIBOX_VLLM_MODEL": "/models/m"}}

    def test_docker_overrides_the_image_entrypoint_and_mounts_repo_and_state(self):
        """Found on a real engine: serving images set their own entrypoint, which swallowed ours."""
        argv = L.plan({**self.SPEC, "launcher": "docker", "image": "img", "gpus": "device=0",
                       "runtime": "runsc"})["argv"]
        self.assertEqual(argv[argv.index("--entrypoint") + 1:][:5], ["python3", "img", "-u", "-m", "rsibox"])
        self.assertIn("/src/rsibox:/rsibox:ro", argv)
        self.assertIn("/shared/box1:/rsibox-state", argv)
        self.assertEqual(argv[argv.index("--runtime") + 1], "runsc")
        self.assertIn("PYTHONPATH=/rsibox/core/livelab", argv)
        self.assertNotIn("--privileged", argv)

    def test_nested_gvisor_wraps_the_daemon_and_binds_the_state_folder(self):
        argv = L.plan({**self.SPEC, "launcher": "docker", "image": "img", "nested_runsc": "/gv/runsc",
                       "allow_unsupported_driver": True})["argv"]
        tail = argv[argv.index("img") + 1:]
        self.assertEqual(tail[:6], ["-m", "rsibox.gvisor", "--runsc", "/gv/runsc", "--bind", "/rsibox-state"])
        self.assertIn("--allow-unsupported-driver", tail)
        self.assertIn("--privileged", argv)

    def test_slurm_script_binds_the_node_and_carries_setup_env_and_sbatch_args(self):
        what = L.plan({**self.SPEC, "launcher": "slurm", "python": "python3", "setup": "source env.sh",
                       "sbatch_args": ["--gres=gpu:1", "--time=04:00:00"]})
        lines = what["script"].splitlines()
        self.assertEqual(lines[0], "#!/bin/bash")
        self.assertIn("#SBATCH --gres=gpu:1", lines)
        self.assertIn("#SBATCH --output=/shared/box1/slurm-%j.log", lines)
        self.assertLess(lines.index("source env.sh"), lines.index("export RSIBOX_VLLM_MODEL=/models/m"))
        self.assertTrue(lines[-1].startswith("exec python3 -u -m rsibox serve --lab vllm --state /shared/box1 "
                                             "--bind 0.0.0.0"), lines[-1])
        self.assertEqual(what["argv"], ["sbatch", "--parsable", "/shared/box1/rsibox.sbatch"])


class GvisorSpec(unittest.TestCase):
    PRESENT = ["null", "nvidia0", "nvidia1", "nvidia5", "nvidia-uvm", "nvidia-uvm-tools", "nvidiactl",
               "nvidia-caps", "nvidia-modeset"]

    def test_one_visible_gpu_is_passed_in_and_renumbered_to_zero(self):
        """Found on a real engine: with only card 5 in the sandbox the driver lists one GPU,
        index 0, and vLLM's lookup of index 5 failed with NVML `Invalid Argument`."""
        names = G.gpu_device_names("5", self.PRESENT)
        self.assertEqual(names, ["nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia5"])
        env = G.sandbox_env({"CUDA_VISIBLE_DEVICES": "5", "HOME": "/tmp"}, names, self.PRESENT)
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "0", "HOME": "/tmp"})

    def test_all_gpus_keep_their_numbers_and_no_gpu_means_no_proxy(self):
        names = G.gpu_device_names(None, self.PRESENT)
        self.assertEqual(names[3:], ["nvidia0", "nvidia1", "nvidia5"])
        self.assertEqual(G.sandbox_env({"CUDA_VISIBLE_DEVICES": "1"}, names, self.PRESENT),
                         {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(G.gpu_device_names("0", ["null", "kfd"]), [])
        self.assertNotIn("--nvproxy", G.runsc_argv("runsc", "/b", "n", gpu=False))

    def test_the_spec_is_read_only_root_with_write_through_binds(self):
        """Pain point: a box file written inside the sandbox that never reaches the host."""
        spec = G.build_spec(["python3", "-m", "rsibox", "serve"], {"A": "1"}, "/w", binds=["/state"])
        self.assertEqual(spec["root"], {"path": "/", "readonly": True})
        self.assertIn({"destination": "/state", "type": "bind", "source": "/state", "options": ["rbind", "rw"]},
                      spec["mounts"])
        argv = G.runsc_argv("/gv/runsc", "/b", "n", allow_unsupported_driver=True)
        self.assertIn("--overlay2=root:memory", argv)              # not all:memory, which would hide binds
        self.assertIn("--nvproxy-allow-unsupported-driver", argv)
        self.assertNotIn("net", [n["type"] for n in spec["linux"]["namespaces"]])   # host network


if __name__ == "__main__":
    unittest.main()
