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
import json, multiprocessing, os, shutil, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

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

    def start_zygote(self, preload_list, refresh="toypkg", extra_env=None, fork_subprocs=False):
        env = dict(os.environ)
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.pkg_dir}{os.pathsep}{existing_py}" if existing_py else str(self.pkg_dir)
        env.pop("CUTE_DSL_ARCH", None)
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("FASTRSI_ZYGOTE_CC", None)
        if extra_env:
            env.update(extra_env)
        cmd = [sys.executable, str(ZYGOTE_PY), "serve", "--sock", self.sock_path, "--refresh", refresh]
        if preload_list is not None:
            cmd.extend(["--preload", ",".join(preload_list)])
        if fork_subprocs:
            cmd.append("--fork-subprocs")
        proc = subprocess.Popen(
            cmd,
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

    def _write_fake_cuda_stack(self):
        self.write_pkg(
            "pynvml.py",
            "def nvmlInit(): pass\n"
            "def nvmlShutdown(): pass\n"
            "def nvmlDeviceGetCount(): return 2\n"
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
            "def device_count(): return 2\n"
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

    def test_fork_subprocs_default_keeps_spawn_invariance(self):
        self._write_fake_cuda_stack()
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/spawn_check.py",
            "import json, multiprocessing as mp, os, torch\n"
            "def run(out_path):\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    payload = {\n"
            "        'default_method': mp.get_start_method(),\n"
            "        'ctx_method': mp.get_context('spawn').get_start_method(),\n"
            "        'numa_bind_v2': os.environ.get('SGLANG_NUMA_BIND_V2'),\n"
            "        'cap_wrapped': hasattr(torch.cuda.get_device_capability, '_fastrsi_forksafe'),\n"
            "        'props_wrapped': hasattr(torch.cuda.get_device_properties, '_fastrsi_props'),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        _, ready = self.start_zygote(
            ["torch", "toypkg.spawn_check"],
            extra_env={"SGLANG_NUMA_BIND_V2": "1"},
        )
        self.assertFalse(ready["fork_subprocs"])
        out_file = str(self.work_dir / "spawn_default.json")
        log_file = str(self.work_dir / "spawn_default.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "toypkg.spawn_check:run", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        data = wait_for_file(out_file, log_path=log_file)
        self.assertEqual(data["default_method"], "spawn")
        self.assertEqual(data["ctx_method"], "spawn")
        self.assertEqual(data["numa_bind_v2"], "1")
        self.assertFalse(data["cap_wrapped"])
        self.assertFalse(data["props_wrapped"])
        child_event = json.loads(Path(log_file).read_text(encoding="utf-8").splitlines()[0])
        self.assertFalse(child_event["fork_subprocs"])

    def test_fork_subprocs_redirects_spawn_to_fork_and_unpatches_child(self):
        self._write_fake_cuda_stack()
        self.write_pkg("sglang/__init__.py", "")
        self.write_pkg("sglang/srt/__init__.py", "")
        self.write_pkg("sglang/srt/entrypoints/__init__.py", "")
        self.write_pkg("sglang/srt/managers/__init__.py", "")
        self.write_pkg("sglang/srt/server_args.py", "IMPORTED = True\n")
        self.write_pkg("sglang/srt/entrypoints/engine.py", "IMPORTED = True\n")
        self.write_pkg("sglang/srt/entrypoints/http_server.py", "IMPORTED = True\n")
        self.write_pkg("sglang/srt/managers/tokenizer_manager.py", "IMPORTED = True\n")
        self.write_pkg(
            "sglang/srt/managers/scheduler.py",
            "import os\nPRELOAD_PID = os.getpid()\n",
        )
        self.write_pkg(
            "sglang/srt/managers/detokenizer_manager.py",
            "import os\nPRELOAD_PID = os.getpid()\n",
        )
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/fork_server.py",
            "import json, multiprocessing as mp, os, torch\n"
            "import cuda.bindings.driver as driver\n"
            "from sglang.srt.managers import detokenizer_manager, scheduler\n"
            "def _scheduler_worker(grand_path):\n"
            "    pre_init = torch.cuda.is_initialized()\n"
            "    cap_wrapped = hasattr(torch.cuda.get_device_capability, '_fastrsi_forksafe')\n"
            "    cur_wrapped = hasattr(torch.cuda.current_device, '_fastrsi_forksafe')\n"
            "    props_wrapped = hasattr(torch.cuda.get_device_properties, '_fastrsi_props')\n"
            "    nvml_cleared = 'PYTORCH_NVML_BASED_CUDA_CHECK' not in os.environ\n"
            "    cuinit_res = list(driver.cuInit(0))\n"
            "    cap = list(torch.cuda.get_device_capability(0))\n"
            "    payload = {\n"
            "        'pid': os.getpid(),\n"
            "        'scheduler_preload_pid': scheduler.PRELOAD_PID,\n"
            "        'detok_preload_pid': detokenizer_manager.PRELOAD_PID,\n"
            "        'pre_init': pre_init,\n"
            "        'post_init': torch.cuda.is_initialized(),\n"
            "        'cap': cap,\n"
            "        'cap_wrapped': cap_wrapped,\n"
            "        'cur_wrapped': cur_wrapped,\n"
            "        'props_wrapped': props_wrapped,\n"
            "        'nvml_cleared': nvml_cleared,\n"
            "        'numa_bind_v2': os.environ.get('SGLANG_NUMA_BIND_V2'),\n"
            "        'cuinit_res': cuinit_res,\n"
            "        'driver_poisoned': driver.DRIVER_POISONED,\n"
            "        'nested_ctx': mp.get_context('spawn').get_start_method(),\n"
            "    }\n"
            "    tmp = grand_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, grand_path)\n"
            "def launch(out_path, grand_path):\n"
            "    # Parent ServerArgs checks must run under active forksafe guards (no CUDA init).\n"
            "    parent_cap_wrapped = hasattr(torch.cuda.get_device_capability, '_fastrsi_forksafe')\n"
            "    parent_props_wrapped = hasattr(torch.cuda.get_device_properties, '_fastrsi_props')\n"
            "    parent_cap = list(torch.cuda.get_device_capability(0))\n"
            "    parent_cur = torch.cuda.current_device()\n"
            "    parent_sms = torch.cuda.get_device_properties(0).multi_processor_count\n"
            "    parent_cuinit = list(driver.cuInit(0))\n"
            "    parent_cuda_before = torch.cuda.is_initialized()\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    default_method = mp.get_start_method()\n"
            "    ctx = mp.get_context('spawn')\n"
            "    ctx_method = ctx.get_start_method()\n"
            "    worker = ctx.Process(target=_scheduler_worker, args=(grand_path,))\n"
            "    worker.start()\n"
            "    worker.join(timeout=10)\n"
            "    payload = {\n"
            "        'parent_pid': os.getpid(),\n"
            "        'parent_cap_wrapped': parent_cap_wrapped,\n"
            "        'parent_props_wrapped': parent_props_wrapped,\n"
            "        'parent_cap': parent_cap,\n"
            "        'parent_cur': parent_cur,\n"
            "        'parent_sms': parent_sms,\n"
            "        'parent_cuinit': parent_cuinit,\n"
            "        'parent_cuda_before': parent_cuda_before,\n"
            "        'parent_cuda_after': torch.cuda.is_initialized(),\n"
            "        'numa_bind_v2': os.environ.get('SGLANG_NUMA_BIND_V2'),\n"
            "        'default_method': default_method,\n"
            "        'ctx_method': ctx_method,\n"
            "        'worker_exitcode': worker.exitcode,\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        # Start with `--fork-subprocs` and default preload list so scheduler + detokenizer preload.
        _, ready = self.start_zygote(
            None,
            refresh="toypkg",
            extra_env={"SGLANG_NUMA_BIND_V2": "1"},
            fork_subprocs=True,
        )
        self.assertTrue(ready["fork_subprocs"])
        self.assertIsInstance(ready["took"].get("sglang.srt.managers.scheduler"), (int, float), ready)
        self.assertIsInstance(ready["took"].get("sglang.srt.managers.detokenizer_manager"), (int, float), ready)

        out_file = str(self.work_dir / "fork_parent.json")
        grand_file = str(self.work_dir / "fork_grandchild.json")
        log_file = str(self.work_dir / "fork_parent.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "toypkg.fork_server:launch", "args": [out_file, grand_file]},
        )
        self.assertTrue(reply["ok"], reply)
        parent = wait_for_file(out_file, log_path=log_file)
        grand = wait_for_file(grand_file, log_path=log_file)

        self.assertTrue(parent["parent_cap_wrapped"])
        self.assertTrue(parent["parent_props_wrapped"])
        self.assertEqual(parent["parent_cap"], [10, 0])
        self.assertEqual(parent["parent_cur"], 0)
        self.assertEqual(parent["parent_sms"], 148)
        self.assertEqual(parent["parent_cuinit"], [3])
        self.assertFalse(parent["parent_cuda_before"])
        self.assertFalse(parent["parent_cuda_after"])
        self.assertEqual(parent["numa_bind_v2"], "0")
        self.assertEqual(parent["default_method"], "fork")
        self.assertEqual(parent["ctx_method"], "fork")
        self.assertEqual(parent["worker_exitcode"], 0)

        self.assertNotEqual(grand["pid"], parent["parent_pid"])
        self.assertEqual(grand["scheduler_preload_pid"], ready["pid"])
        self.assertEqual(grand["detok_preload_pid"], ready["pid"])
        self.assertFalse(grand["pre_init"])
        self.assertTrue(grand["post_init"])
        self.assertEqual(grand["cap"], [10, 0])
        self.assertFalse(grand["cap_wrapped"])
        self.assertFalse(grand["cur_wrapped"])
        self.assertFalse(grand["props_wrapped"])
        self.assertTrue(grand["nvml_cleared"])
        self.assertEqual(grand["numa_bind_v2"], "0")
        self.assertEqual(grand["cuinit_res"], [0])
        self.assertTrue(grand["driver_poisoned"])
        self.assertEqual(grand["nested_ctx"], "spawn")

    def test_fork_subprocs_falls_back_to_spawn_if_cuda_initializes_early(self):
        self._write_fake_cuda_stack()
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/early_cuda.py",
            "import json, multiprocessing as mp, os, torch\n"
            "def run(out_path):\n"
            "    torch.cuda.initialized = True\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    payload = {\n"
            "        'default_method': mp.get_start_method(),\n"
            "        'ctx_method': mp.get_context('spawn').get_start_method(),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.write_pkg(
            "toypkg/bad_scheduler.py",
            "import torch\ntorch.cuda.initialized = True\n",
        )
        # Preloading a scheduler module that initializes CUDA aborts serve cleanly.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.pkg_dir)
        bad_sock = str(self.work_dir / "bad.sock")
        aborted = subprocess.run(
            [sys.executable, str(ZYGOTE_PY), "serve", "--sock", bad_sock,
             "--preload", "torch,toypkg.bad_scheduler", "--fork-subprocs"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        self.assertNotEqual(aborted.returncode, 0)
        self.assertIn("initialized CUDA; fork cannot use GPU", aborted.stderr)

        # Per-request fork_subprocs=True falls back to 'spawn' with JSON events if CUDA inits early.
        self.start_zygote(["torch", "toypkg.early_cuda"])
        out_file = str(self.work_dir / "fallback.json")
        log_file = str(self.work_dir / "fallback.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            fork_subprocs=True,
            entry={"callable": "toypkg.early_cuda:run", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        data = wait_for_file(out_file, log_path=log_file)
        self.assertEqual(data["default_method"], "spawn")
        self.assertEqual(data["ctx_method"], "spawn")
        events = [json.loads(line) for line in Path(log_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        fallbacks = [ev for ev in events if ev.get("event") == "fork_subprocs_fallback"]
        self.assertEqual(
            [(ev["api"], ev["effective"], ev["reason"]) for ev in fallbacks],
            [("set_start_method", "spawn", "cuda_initialized"),
             ("get_context", "spawn", "cuda_initialized")],
            events,
        )

    def test_fork_subprocs_selective_refresh_preserves_unrelated_modules(self):
        self._write_fake_cuda_stack()
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/state_mod.py",
            "STATE_DICT = {'version': 1}\n"
            "STATE_ID = id(STATE_DICT)\n",
        )
        leaf_path = self.write_pkg(
            "toypkg/leaf_entry.py",
            "import json, multiprocessing as mp, os\n"
            "from toypkg import state_mod\n"
            "PRELOAD_STATE_ID = state_mod.STATE_ID\n"
            "def version(): return 'v1'\n"
            "def run(out_path):\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    payload = {\n"
            "        'version': version(),\n"
            "        'state_id': state_mod.STATE_ID,\n"
            "        'dict_id': id(state_mod.STATE_DICT),\n"
            "        'default_method': mp.get_start_method(),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.start_zygote(
            ["torch", "toypkg.state_mod", "toypkg.leaf_entry"],
            refresh="toypkg",
            fork_subprocs=True,
        )
        baseline_file = str(self.work_dir / "selective_base.json")
        r0 = zygote.request(
            self.sock_path,
            entry={"callable": "toypkg.leaf_entry:run", "args": [baseline_file]},
        )
        self.assertTrue(r0["ok"], r0)
        base_data = wait_for_file(baseline_file)
        self.assertEqual(base_data["version"], "v1")
        self.assertEqual(base_data["default_method"], "fork")

        updated = Path(leaf_path).read_text(encoding="utf-8").replace("return 'v1'", "return 'v2'")
        Path(leaf_path).write_text(updated, encoding="utf-8")

        out_file = str(self.work_dir / "selective_refreshed.json")
        log_file = str(self.work_dir / "selective_refreshed.log")
        r1 = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "toypkg.leaf_entry:run", "args": [out_file]},
        )
        self.assertTrue(r1["ok"], r1)
        data = wait_for_file(out_file, log_path=log_file)
        self.assertEqual(data["version"], "v2")
        self.assertEqual(data["state_id"], base_data["state_id"])
        self.assertEqual(data["dict_id"], base_data["dict_id"])
        self.assertEqual(data["default_method"], "fork")

        events = [json.loads(line) for line in Path(log_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual([ev for ev in events if ev.get("event") == "fork_subprocs_fallback"], [], events)
        child_ev = next(ev for ev in events if ev.get("event") == "zygote_child")
        self.assertTrue(child_ev["fork_subprocs"])
        self.assertEqual(child_ev["scope"], "dependents")
        self.assertEqual(child_ev["dropped_modules"], ["toypkg.leaf_entry"])
        self.assertEqual(child_ev["dropped"], 1)

    def test_fork_subprocs_falls_back_to_spawn_when_refresh_touches_torch_op_state(self):
        self._write_fake_cuda_stack()
        self.write_pkg("sglang/__init__.py", "")
        self.write_pkg("sglang/srt/__init__.py", "")
        self.write_pkg("sglang/srt/distributed/__init__.py", "")
        ps_path = self.write_pkg(
            "sglang/srt/distributed/parallel_state.py",
            "GROUP_TAG = 'base'\n",
        )
        self.write_pkg("sglang/srt/entrypoints/__init__.py", "")
        self.write_pkg(
            "sglang/srt/entrypoints/engine.py",
            "import json, multiprocessing as mp, os, torch\n"
            "from sglang.srt.distributed import parallel_state\n"
            "def run(out_path):\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    payload = {\n"
            "        'group_tag': parallel_state.GROUP_TAG,\n"
            "        'default_method': mp.get_start_method(),\n"
            "        'cap_wrapped': hasattr(torch.cuda.get_device_capability, '_fastrsi_forksafe'),\n"
            "        'numa_bind_v2': os.environ.get('SGLANG_NUMA_BIND_V2'),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.start_zygote(
            ["torch", "sglang.srt.distributed.parallel_state", "sglang.srt.entrypoints.engine"],
            refresh="sglang",
            extra_env={"SGLANG_NUMA_BIND_V2": "1"},
            fork_subprocs=True,
        )
        Path(ps_path).write_text("GROUP_TAG = 'edited'\n", encoding="utf-8")

        out_file = str(self.work_dir / "ps_fallback.json")
        log_file = str(self.work_dir / "ps_fallback.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "sglang.srt.entrypoints.engine:run", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        data = wait_for_file(out_file, log_path=log_file)
        self.assertEqual(data["group_tag"], "edited")
        self.assertEqual(data["default_method"], "spawn")
        self.assertFalse(data["cap_wrapped"])
        self.assertEqual(data["numa_bind_v2"], "1")

        events = [json.loads(line) for line in Path(log_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        fallbacks = [ev for ev in events if ev.get("event") == "fork_subprocs_fallback"]
        self.assertEqual(len(fallbacks), 1, events)
        self.assertEqual(fallbacks[0]["api"], "refresh_child")
        self.assertEqual(fallbacks[0]["effective"], "spawn")
        self.assertEqual(fallbacks[0]["reason"], "refresh_rebinds_torch_ops")
        child_ev = next(ev for ev in events if ev.get("event") == "zygote_child")
        self.assertFalse(child_ev["fork_subprocs"])
        self.assertGreaterEqual(child_ev["dropped"], 2)

    def test_fork_subprocs_mp_process_falls_back_to_spawn_after_early_cuda(self):
        self._write_fake_cuda_stack()
        self.write_pkg("toypkg/__init__.py", "")
        self.write_pkg(
            "toypkg/mp_proc_check.py",
            "import json, multiprocessing as mp, os, torch\n"
            "PARENT_ONLY = 'initial'\n"
            "def _worker(grand_path):\n"
            "    payload = {'parent_only': PARENT_ONLY, 'pid': os.getpid()}\n"
            "    tmp = grand_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, grand_path)\n"
            "def run(out_path, grand_path):\n"
            "    global PARENT_ONLY\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    method_before_cuda = mp.get_start_method()\n"
            "    PARENT_ONLY = 'mutated_in_parent'\n"
            "    torch.cuda.initialized = True\n"
            "    p = mp.Process(target=_worker, args=(grand_path,))\n"
            "    p.start()\n"
            "    p.join(timeout=10)\n"
            "    payload = {'method_before_cuda': method_before_cuda, 'exitcode': p.exitcode}\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.start_zygote(["torch", "toypkg.mp_proc_check"], refresh="toypkg", fork_subprocs=True)
        out_file = str(self.work_dir / "mp_parent.json")
        grand_file = str(self.work_dir / "mp_grand.json")
        log_file = str(self.work_dir / "mp_parent.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "toypkg.mp_proc_check:run", "args": [out_file, grand_file]},
        )
        self.assertTrue(reply["ok"], reply)
        parent = wait_for_file(out_file, log_path=log_file)
        grand = wait_for_file(grand_file, log_path=log_file)
        self.assertEqual(parent["method_before_cuda"], "fork")
        self.assertEqual(parent["exitcode"], 0)
        # Spawned child re-imports fresh instead of inheriting parent's post-import global mutation.
        self.assertEqual(grand["parent_only"], "initial")
        events = [json.loads(line) for line in Path(log_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        fallbacks = [ev for ev in events if ev.get("event") == "fork_subprocs_fallback"]
        self.assertTrue(
            any(ev["api"] == "get_context" and ev["requested"] is None and ev["effective"] == "spawn" for ev in fallbacks),
            events,
        )

        # In-process restore removes `_default_context` instance overrides cleanly.
        dctx = multiprocessing.context._default_context
        had_get = "get_context" in dctx.__dict__
        had_set = "set_start_method" in dctx.__dict__
        restore = zygote.enable_fork_subprocs()
        try:
            self.assertIn("get_context", dctx.__dict__)
            self.assertIn("set_start_method", dctx.__dict__)
        finally:
            restore()
        self.assertEqual("get_context" in dctx.__dict__, had_get)
        self.assertEqual("set_start_method" in dctx.__dict__, had_set)

    def test_fork_subprocs_custom_op_leaf_triggers_spawn_fallback_and_op_redefine_guard(self):
        self.write_pkg(
            "pynvml.py",
            "def nvmlInit(): pass\n"
            "def nvmlShutdown(): pass\n"
            "def nvmlDeviceGetCount(): return 1\n"
            "def nvmlDeviceGetHandleByIndex(idx): return 0\n"
            "def nvmlDeviceGetCudaComputeCapability(handle): return (10, 0)\n",
        )
        self.write_pkg(
            "torch/__init__.py",
            "import os, types\n"
            "_SCHEMAS = {}\n"
            "_IMPLS = {}\n"
            "class _C:\n"
            "    @staticmethod\n"
            "    def _get_schema(qual, overload=''):\n"
            "        key = (qual, overload)\n"
            "        if key not in _SCHEMAS:\n"
            "            raise RuntimeError(f'missing schema {qual}')\n"
            "        return _SCHEMAS[key]\n"
            "    @staticmethod\n"
            "    def parse_schema(schema):\n"
            "        return schema\n"
            "cuda = types.SimpleNamespace(initialized=False)\n"
            "cuda.is_initialized = lambda: cuda.initialized\n"
            "cuda.is_available = lambda: True\n"
            "cuda.device_count = lambda: 1\n"
            "cuda.get_device_capability = lambda device=None: (10, 0)\n"
            "cuda.current_device = lambda: 0\n"
            "cuda.get_device_properties = lambda device=None: types.SimpleNamespace(multi_processor_count=148, major=10, minor=0)\n",
        )
        self.write_pkg(
            "torch/library.py",
            "import torch\n"
            "class Library:\n"
            "    def __init__(self, ns, kind):\n"
            "        self.ns, self.kind = ns, kind\n"
            "    def define(self, schema):\n"
            "        name = schema.split('(', 1)[0].strip()\n"
            "        op, _, overload = name.partition('.')\n"
            "        key = (f'{self.ns}::{op}', overload)\n"
            "        if key in torch._SCHEMAS:\n"
            "            raise RuntimeError(f'duplicate op {key[0]}')\n"
            "        torch._SCHEMAS[key] = f'{self.ns}::{schema}'\n"
            "        return name\n"
            "    def impl(self, op_name, fn, dispatch_key='', *, allow_override=False):\n"
            "        key = (f'{self.ns}::{op_name}', dispatch_key)\n"
            "        if key in torch._IMPLS and not allow_override:\n"
            "            raise RuntimeError(f'duplicate impl {key[0]}')\n"
            "        torch._IMPLS[key] = fn\n",
        )
        self.write_pkg("sglang/__init__.py", "")
        self.write_pkg("sglang/srt/__init__.py", "")
        self.write_pkg("sglang/srt/layers/__init__.py", "")
        self.write_pkg("sglang/srt/entrypoints/__init__.py", "")
        leaf_path = self.write_pkg(
            "sglang/srt/layers/my_op_leaf.py",
            "import torch\n"
            "import torch.library\n"
            "_LIB = torch.library.Library('sglang', 'DEF')\n"
            "_LIB.define('demo(Tensor x) -> Tensor')\n"
            "_LIB.impl('demo', lambda x: 'v1', 'CPU')\n"
            "def call_demo(): return torch._IMPLS[('sglang::demo', 'CPU')](None)\n",
        )
        self.write_pkg(
            "sglang/srt/entrypoints/engine.py",
            "import json, multiprocessing as mp, os\n"
            "from sglang.srt.layers import my_op_leaf\n"
            "def run(out_path):\n"
            "    mp.set_start_method('spawn', force=True)\n"
            "    payload = {\n"
            "        'op_result': my_op_leaf.call_demo(),\n"
            "        'default_method': mp.get_start_method(),\n"
            "    }\n"
            "    tmp = out_path + '.tmp'\n"
            "    open(tmp, 'w', encoding='utf-8').write(json.dumps(payload))\n"
            "    os.replace(tmp, out_path)\n",
        )
        self.start_zygote(
            ["torch", "sglang.srt.layers.my_op_leaf", "sglang.srt.entrypoints.engine"],
            refresh="sglang",
            fork_subprocs=True,
        )
        Path(leaf_path).write_text(
            Path(leaf_path).read_text(encoding="utf-8").replace("'v1'", "'v2'"),
            encoding="utf-8",
        )
        out_file = str(self.work_dir / "custom_op.json")
        log_file = str(self.work_dir / "custom_op.log")
        reply = zygote.request(
            self.sock_path,
            log=log_file,
            entry={"callable": "sglang.srt.entrypoints.engine:run", "args": [out_file]},
        )
        self.assertTrue(reply["ok"], reply)
        data = wait_for_file(out_file, log_path=log_file)
        self.assertEqual(data["op_result"], "v2")
        self.assertEqual(data["default_method"], "spawn")
        events = [json.loads(line) for line in Path(log_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(
            any(ev.get("event") == "fork_subprocs_fallback" and ev.get("reason") == "refresh_rebinds_torch_ops" for ev in events),
            events,
        )
        self.assertTrue(any(ev.get("event") == "zygote_op_redefined" and ev.get("op") == "sglang::demo" for ev in events), events)
        self.assertTrue(any(ev.get("event") == "zygote_op_impl_replaced" and ev.get("op") == "sglang::demo" for ev in events), events)

        # Direct AST detection also catches `direct_register_custom_op` and `sglang.srt.utils.custom_op`.
        self.assertTrue(zygote._source_registers_torch_ops("from sglang.srt.utils.custom_op import direct_register_custom_op\n"))
        self.assertTrue(zygote._source_registers_torch_ops("direct_register_custom_op(op_name='foo', op_func=f)\n"))
        self.assertFalse(zygote._source_registers_torch_ops("from sglang.srt.utils.common import is_cuda\nx = 1\n"))

    def test_serve_programmatic_default_includes_subproc_preload_when_fork_subprocs(self):
        seen = []

        def fake_install(mods):
            seen.append(tuple(mods))
            return {"nvml_env_set": False}, lambda: None

        class _StopEarly(Exception):
            pass

        with (
            mock.patch.object(zygote, "install_guards", side_effect=fake_install),
            mock.patch.object(zygote, "preload", side_effect=_StopEarly),
        ):
            with self.assertRaises(_StopEarly):
                zygote.serve(self.sock_path, fork_subprocs=False)
            with self.assertRaises(_StopEarly):
                zygote.serve(self.sock_path, fork_subprocs=True)
        self.assertEqual(seen[0], zygote.DEFAULT_PRELOAD)
        self.assertEqual(seen[1], zygote.DEFAULT_PRELOAD + zygote.SUBPROC_PRELOAD)


if __name__ == "__main__":
    unittest.main()
