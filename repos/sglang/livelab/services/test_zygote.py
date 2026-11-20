"""Golden CPU tests for the SGLang zygote (stdlib unittest; no real torch or sglang needed).

Every case starts a real `zygote.py serve` process over an AF_UNIX socket and exercises the
real `os.fork()` child path against toy/stand-in modules on PYTHONPATH:
1. Request env + unset applied in the forked child; callable and module entries both run;
   sibling forks do not inherit each other's env mutations.
2. `forksafe` guards (`install`, `install_properties`, `pin_triton_target`) keep CUDA
   uninitialized across preload; the forked child runs `forksafe.unpatch` and honors the
   requested `CUDA_VISIBLE_DEVICES` (vLLM zygote lesson).
3. `CUTE_DSL_ARCH` is populated and `cuda.bindings.driver.cuInit` is stubbed before preload,
   then restored in the child (FlashInfer/CuTe DSL pre-fork cuInit lesson).
4. Edited package modules and stale `linecache` entries (including `mtime=None` pinned lines)
   are refreshed in the child so `@triton.jit` / `inspect.getsource` never see pre-edit text
   (Megatron overlay lesson); edits outside `--refresh` return a `stale` finding.
5. Forked children call `setsid` (leaving the zygote's process group) and are still tracked,
   killed, and reaped by `ppid` rather than `pgrp` (Megatron M-H17).
6. Overlong AF_UNIX socket paths (`> MAX_SOCK_PATH`, `< 107` bytes) fail immediately before
   any preload import runs.

Run from the repo root:
    python3 -m unittest discover -s repos/sglang/livelab/services -t repos/sglang/livelab -v
"""
import json, os, shutil, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zygote  # noqa: E402

ZYGOTE_PY = HERE / "zygote.py"


def wait_for_file(path, timeout=3.0, log_path=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                return json.loads(text)
        time.sleep(0.02)
    extra = Path(log_path).read_text(encoding="utf-8") if log_path and os.path.isfile(log_path) else ""
    raise AssertionError(f"timed out waiting for JSON file {path}: {extra}")


class SglangZygoteTests(unittest.TestCase):
    def setUp(self):
        # Short /tmp directory so AF_UNIX socket paths stay ~26 bytes on macOS and Linux.
        self.work_dir = Path(tempfile.mkdtemp(dir="/tmp", prefix="sglz-"))
        self.pkg_dir = self.work_dir / "pkgs"
        self.pkg_dir.mkdir()
        self.sock_path = str(self.work_dir / "z.sock")
        self.addCleanup(shutil.rmtree, self.work_dir, True)

    def write_pkg(self, rel_path, content):
        full_path = self.pkg_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return str(full_path)

    def start_zygote(self, preload_list, refresh="toypkg", extra_env=None):
        env = dict(os.environ)
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.pkg_dir}{os.pathsep}{existing_py}" if existing_py else str(self.pkg_dir)
        env.pop("CUTE_DSL_ARCH", None)
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("FASTRSI_ZYGOTE_CC", None)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            [sys.executable, str(ZYGOTE_PY), "serve", "--sock", self.sock_path,
             "--preload", ",".join(preload_list), "--refresh", refresh],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            start_new_session=True,
        )

        def cleanup():
            if proc.poll() is None:
                try:
                    zygote.call(self.sock_path, {"cmd": "stop"}, timeout=2)
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        self.addCleanup(cleanup)
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr_text = proc.stderr.read() if proc.stderr else ""
                raise AssertionError(f"zygote exited early ({proc.returncode}): {stderr_text}")
            if os.path.exists(self.sock_path):
                ready_line = proc.stdout.readline()
                return proc, json.loads(ready_line)
            time.sleep(0.02)
        raise AssertionError("timed out waiting for zygote_ready")

    def test_child_applies_request_env_and_runs_callable_and_module_entries(self):
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/preload_probe.py",
            "import os, time\nIMPORT_PID = os.getpid()\nIMPORT_NS = time.monotonic_ns()\n",
        )
        self.write_pkg(
            "toypkg/runner.py",
            "import json, os, sys\n"
            "from toypkg import preload_probe\n"
            "def write_report(out_path, mode):\n"
            "    payload = {\n"
            "        'mode': mode,\n"
            "        'pid': os.getpid(),\n"
            "        'argv': list(sys.argv),\n"
            "        'token': os.environ.get('SGLANG_TEST_TOKEN'),\n"
            "        'shared': os.environ.get('SHARED_VAR'),\n"
            "        'has_drop': 'DROP_ME' in os.environ,\n"
            "        'preload_pid': preload_probe.IMPORT_PID,\n"
            "        'preload_ns': preload_probe.IMPORT_NS,\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n"
            "if __name__ == '__main__':\n"
            "    write_report(sys.argv[1], 'module')\n",
        )
        proc, ready = self.start_zygote(
            ["toypkg.preload_probe"],
            extra_env={"SHARED_VAR": "from_zygote", "DROP_ME": "1"},
        )
        out_one = str(self.work_dir / "child1.json")
        reply_one = zygote.request(
            self.sock_path,
            argv=["--model-path", "toy-model"],
            env={"SGLANG_TEST_TOKEN": "alpha", "SHARED_VAR": "from_child1"},
            unset=["DROP_ME"],
            entry={"callable": "toypkg.runner:write_report", "args": [out_one, "callable"]},
        )
        self.assertTrue(reply_one["ok"], reply_one)
        data_one = wait_for_file(out_one)
        self.assertEqual(data_one["mode"], "callable")
        self.assertEqual(data_one["token"], "alpha")
        self.assertEqual(data_one["shared"], "from_child1")
        self.assertFalse(data_one["has_drop"])
        self.assertEqual(data_one["argv"], ["toypkg.runner", "--model-path", "toy-model"])
        self.assertEqual(data_one["preload_pid"], ready["pid"])
        self.assertNotEqual(data_one["pid"], proc.pid)

        out_two = str(self.work_dir / "child2.json")
        reply_two = zygote.request(
            self.sock_path,
            argv=[out_two, "--tp-size", "2"],
            env={"SGLANG_TEST_TOKEN": "beta"},
            entry={"module": "toypkg.runner"},
        )
        self.assertTrue(reply_two["ok"], reply_two)
        data_two = wait_for_file(out_two)
        self.assertEqual(data_two["mode"], "module")
        self.assertEqual(data_two["token"], "beta")
        self.assertEqual(data_two["shared"], "from_zygote")
        self.assertTrue(data_two["has_drop"])
        self.assertEqual(data_two["preload_ns"], data_one["preload_ns"])

    def test_forksafe_guards_preload_unpatches_child_and_honors_cuda_visible_devices(self):
        self.write_pkg(
            "pynvml.py",
            "def nvmlInit(): pass\n"
            "def nvmlShutdown(): pass\n"
            "def nvmlDeviceGetCount(): return 4\n"
            "def nvmlDeviceGetHandleByIndex(idx): return idx\n"
            "def nvmlDeviceGetCudaComputeCapability(handle): return (10, 0)\n",
        )
        self.write_pkg(
            "torch/__init__.py",
            "import os, types\n"
            "class _DevProps:\n"
            "    multi_processor_count = 148\n"
            "    major = 10\n"
            "    minor = 0\n"
            "cuda = types.SimpleNamespace(initialized=False)\n"
            "def is_initialized(): return cuda.initialized\n"
            "def is_available():\n"
            "    if os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') != '1':\n"
            "        cuda.initialized = True\n"
            "    return True\n"
            "def device_count(): return 4\n"
            "def get_device_capability(device=None):\n"
            "    cuda.initialized = True\n"
            "    return (10, 0)\n"
            "def current_device():\n"
            "    cuda.initialized = True\n"
            "    return 0\n"
            "def get_device_properties(device=None):\n"
            "    cuda.initialized = True\n"
            "    return _DevProps()\n"
            "cuda.is_initialized = is_initialized\n"
            "cuda.is_available = is_available\n"
            "cuda.device_count = device_count\n"
            "cuda.get_device_capability = get_device_capability\n"
            "cuda.current_device = current_device\n"
            "cuda.get_device_properties = get_device_properties\n",
        )
        self.write_pkg("triton/__init__.py", "")
        self.write_pkg("triton/backends/__init__.py", "")
        self.write_pkg(
            "triton/backends/compiler.py",
            "class GPUTarget:\n"
            "    def __init__(self, backend, arch, warp_size):\n"
            "        self.backend, self.arch, self.warp_size = backend, arch, warp_size\n",
        )
        self.write_pkg("triton/backends/nvidia/__init__.py", "")
        self.write_pkg(
            "triton/backends/nvidia/driver.py",
            "import torch\n"
            "from triton.backends.compiler import GPUTarget\n"
            "class CudaDriver:\n"
            "    def get_current_target(self):\n"
            "        torch.cuda.initialized = True\n"
            "        return GPUTarget('cuda', 100, 32)\n",
        )
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/gpu_import.py",
            "import json, os, torch\n"
            "from triton.backends.nvidia.driver import CudaDriver\n"
            "PRELOAD_FACTS = {\n"
            "    'avail': torch.cuda.is_available(),\n"
            "    'cap': list(torch.cuda.get_device_capability(0)),\n"
            "    'cur': torch.cuda.current_device(),\n"
            "    'sms': torch.cuda.get_device_properties(0).multi_processor_count,\n"
            "    'triton_arch': CudaDriver().get_current_target().arch,\n"
            "    'cuda_init_during_preload': torch.cuda.is_initialized(),\n"
            "}\n"
            "def inspect_child(out_path):\n"
            "    payload = {\n"
            "        'preload_facts': PRELOAD_FACTS,\n"
            "        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
            "        'nvml_check_cleared': 'PYTORCH_NVML_BASED_CUDA_CHECK' not in os.environ,\n"
            "        'cap_wrapped': hasattr(torch.cuda.get_device_capability, '_fastrsi_forksafe'),\n"
            "        'cur_wrapped': hasattr(torch.cuda.current_device, '_fastrsi_forksafe'),\n"
            "        'props_wrapped': hasattr(torch.cuda.get_device_properties, '_fastrsi_props'),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        _, ready = self.start_zygote(
            ["torch", "toypkg.gpu_import"],
            extra_env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        )
        self.assertTrue(ready["guards"]["forksafe"]["installed"], ready)
        self.assertTrue(ready["guards"]["properties"]["installed"], ready)

        out_file = str(self.work_dir / "gpu_child.json")
        log_file = str(self.work_dir / "gpu_child.log")
        reply = zygote.request(
            self.sock_path,
            env={"CUDA_VISIBLE_DEVICES": "2,3"},
            log=log_file,
            entry={"callable": "toypkg.gpu_import:inspect_child", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        child_data = wait_for_file(out_file)
        self.assertEqual(
            child_data["preload_facts"],
            {
                "avail": True,
                "cap": [10, 0],
                "cur": 0,
                "sms": 148,
                "triton_arch": 100,
                "cuda_init_during_preload": False,
            },
        )
        self.assertEqual(child_data["cuda_visible_devices"], "2,3")
        self.assertTrue(child_data["nvml_check_cleared"])
        self.assertFalse(child_data["cap_wrapped"])
        self.assertFalse(child_data["cur_wrapped"])
        self.assertFalse(child_data["props_wrapped"])
        child_log = json.loads(Path(log_file).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(child_log["cuda_visible_devices"], "2,3")
        self.assertEqual(
            sorted(child_log["unpatched"]),
            ["current_device", "get_device_capability", "get_device_properties"],
        )

    def test_cute_dsl_arch_set_and_cuinit_stubbed_before_preload(self):
        self.write_pkg(
            "pynvml.py",
            "def nvmlInit(): pass\n"
            "def nvmlShutdown(): pass\n"
            "def nvmlDeviceGetCount(): return 1\n"
            "def nvmlDeviceGetHandleByIndex(idx): return 0\n"
            "def nvmlDeviceGetCudaComputeCapability(handle): return (10, 0)\n",
        )
        self.write_pkg("cuda/__init__.py", "")
        self.write_pkg("cuda/bindings/__init__.py", "")
        self.write_pkg(
            "cuda/bindings/driver.py",
            "class CUresult:\n"
            "    CUDA_SUCCESS = 0\n"
            "    CUDA_ERROR_NOT_INITIALIZED = 3\n"
            "DRIVER_POISONED = False\n"
            "def cuInit(flags=0):\n"
            "    global DRIVER_POISONED\n"
            "    DRIVER_POISONED = True\n"
            "    return (CUresult.CUDA_SUCCESS,)\n",
        )
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/cute_probe.py",
            "import json, os\n"
            "import cuda.bindings.driver as driver\n"
            "ARCH_AT_IMPORT = os.environ.get('CUTE_DSL_ARCH')\n"
            "if not ARCH_AT_IMPORT:\n"
            "    driver.cuInit(0)\n"
            "CUINIT_DURING_PRELOAD = list(driver.cuInit(0))\n"
            "POISONED_DURING_PRELOAD = driver.DRIVER_POISONED\n"
            "def report(out_path):\n"
            "    child_cuinit = list(driver.cuInit(0))\n"
            "    payload = {\n"
            "        'arch_at_import': ARCH_AT_IMPORT,\n"
            "        'cuinit_during_preload': CUINIT_DURING_PRELOAD,\n"
            "        'poisoned_during_preload': POISONED_DURING_PRELOAD,\n"
            "        'child_cuinit': child_cuinit,\n"
            "        'child_can_init_driver': driver.DRIVER_POISONED,\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        _, ready = self.start_zygote(["toypkg.cute_probe"])
        self.assertEqual(ready["guards"]["cute_dsl_arch"], "sm_100a")
        self.assertTrue(ready["guards"]["cuinit_stub"])

        out_file = str(self.work_dir / "cute.json")
        reply = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.cute_probe:report", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        result = wait_for_file(out_file)
        self.assertEqual(result["arch_at_import"], "sm_100a")
        self.assertEqual(result["cuinit_during_preload"], [3])
        self.assertFalse(result["poisoned_during_preload"])
        self.assertEqual(result["child_cuinit"], [0])
        self.assertTrue(result["child_can_init_driver"])

    def test_linecache_and_module_source_refreshed_after_file_edit(self):
        self.write_pkg("toypkg/__init__.py", "")
        kernel_path = self.write_pkg(
            "toypkg/kernels.py",
            "import inspect, json, linecache, os\n"
            "def jit_kernel():\n"
            "    return 'v1'\n"
            "def dump(out_path):\n"
            "    payload = {\n"
            "        'value': jit_kernel(),\n"
            "        'inspect_src': inspect.getsource(jit_kernel),\n"
            "        'module_lines': ''.join(linecache.getlines(__file__)),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.write_pkg("otherpkg/__init__.py", "")
        # Stays loaded across `--refresh toypkg`; pins kernel_path in linecache with mtime=None
        # (what live-swap/overlays write, which stdlib linecache.checkcache() skips).
        outside_path = self.write_pkg(
            "otherpkg/fixed.py",
            "import linecache\n"
            f"_TARGET = {kernel_path!r}\n"
            "_TEXT = open(_TARGET, encoding='utf-8').read()\n"
            "linecache.cache[_TARGET] = (len(_TEXT), None, _TEXT.splitlines(True), _TARGET)\n",
        )

        self.start_zygote(["toypkg.kernels", "otherpkg.fixed"], refresh="toypkg")

        # Same-length edit ('v1' -> 'v2') on disk after preload cached + pinned linecache.
        updated_kernel = Path(kernel_path).read_text(encoding="utf-8").replace("return 'v1'", "return 'v2'")
        Path(kernel_path).write_text(updated_kernel, encoding="utf-8")

        out_file = str(self.work_dir / "refreshed.json")
        reply = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.kernels:dump", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        self.assertTrue(
            any(os.path.samefile(path, kernel_path) for path in reply["refreshed"]),
            reply,
        )
        refreshed = wait_for_file(out_file)
        self.assertEqual(refreshed["value"], "v2")
        self.assertIn("return 'v2'", refreshed["inspect_src"])
        self.assertIn("return 'v2'", refreshed["module_lines"])

        # Editing a preloaded module outside --refresh refuses the fork unless accepted.
        Path(outside_path).write_text("FIXED = 99\n", encoding="utf-8")
        refused = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.kernels:dump", "args": [out_file]},
        )
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["findings"][0]["check"], "stale")

    def test_children_tracked_and_reaped_by_ppid_across_setsid(self):
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/daemon_sim.py",
            "import json, os, sys, time\n"
            "def hold(out_path):\n"
            "    payload = {'pid': os.getpid(), 'ppid': os.getppid(), 'pgid': os.getpgrp()}\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n"
            "    time.sleep(60)\n"
            "def exit_with(code):\n"
            "    sys.exit(code)\n",
        )
        proc, ready = self.start_zygote(["toypkg.daemon_sim"])
        hold_file = str(self.work_dir / "hold.json")
        sleeper = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.daemon_sim:hold", "args": [hold_file]},
        )
        if sleeper.get("pid"):
            def _reap_sleeper(target_pid=sleeper["pid"]):
                try:
                    os.kill(target_pid, signal.SIGKILL)
                except OSError:
                    pass
            self.addCleanup(_reap_sleeper)
        exiter = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.daemon_sim:exit_with", "args": [7]},
        )
        self.assertTrue(sleeper["ok"] and exiter["ok"])
        hold_info = wait_for_file(hold_file)
        self.assertEqual(hold_info["ppid"], ready["pid"])
        # setsid puts the forked child in its own process group, disjoint from the zygote's.
        self.assertEqual(hold_info["pgid"], hold_info["pid"])
        self.assertNotEqual(hold_info["pgid"], os.getpgid(proc.pid))

        deadline = time.time() + 5.0
        ping = {}
        while time.time() < deadline:
            ping = zygote.call(self.sock_path, {"cmd": "ping"})
            if str(exiter["pid"]) in ping["exits"]:
                break
            time.sleep(0.05)
        self.assertEqual(ping["exits"].get(str(exiter["pid"])), 7)
        self.assertEqual(ping["children"], [sleeper["pid"]])

        killed = zygote.call(self.sock_path, {"cmd": "kill"})
        self.assertIn(str(sleeper["pid"]), killed["killed"])
        self.assertIsNotNone(killed["killed"][str(sleeper["pid"])])
        with self.assertRaises(ProcessLookupError):
            os.kill(sleeper["pid"], 0)
        after = zygote.call(self.sock_path, {"cmd": "ping"})
        self.assertEqual(after["children"], [])

    def test_socket_path_cap_fails_fast_before_preload(self):
        self.assertLess(zygote.MAX_SOCK_PATH, 107)
        marker_path = self.work_dir / "preload_ran.txt"
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/side_effect.py",
            f"open({str(marker_path)!r}, 'w', encoding='utf-8').write('ran')\n",
        )
        long_sock = str(self.work_dir / ("s" * 115 + ".sock"))
        self.assertIsNotNone(zygote.sock_path_error(long_sock))
        self.assertIsNone(zygote.sock_path_error(self.sock_path))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.pkg_dir)
        completed = subprocess.run(
            [sys.executable, str(ZYGOTE_PY), "serve", "--sock", long_sock,
             "--preload", "toypkg.side_effect"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(marker_path.exists())
        self.assertIn("socket path too long for AF_UNIX", completed.stderr)


if __name__ == "__main__":
    unittest.main()
