#!/usr/bin/env python3
"""Modal GPU CI app (`rsibox-gpu-ci`) with cached core and vLLM container images.

Runs all core + multi-engine suites in `run_core_gpu` (torch 2.9.1+cu128, Triton, Ray)
in parallel with `vllm-services` in `run_vllm_gpu` (real vLLM + pytest).
"""
from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import modal

DEFAULT_GPU = os.environ.get("RSIBOX_MODAL_GPU", "L4")
DEFAULT_REMOTE_TIMEOUT = int(os.environ.get("RSIBOX_MODAL_TIMEOUT", "3600"))

CORE_SUITE_ORDER = (
    "core-conformance-cases",
    "core-kernel",
    "core-forksafe",
    "core-rsibox",
    "core-survey",
    "core-doctor",
    "core-conformance-toylab",
    "template-adapter",
    "fastvideo-adapter",
    "fastvideo-services",
    "fastvideo-conformance",
    "vllm-adapter",
    "megatron-livelab",
    "megatron-tools",
    "verl-adapter",
    "verl-tools",
    "verl-conformance",
    "miles-adapter",
    "miles-conformance",
    "sglang-adapter",
    "sglang-services",
    "sglang-conformance",
    "review-checkers",
    "scripts-tests",
)
VLLM_SUITE_ORDER = ("vllm-services",)
ALL_SUITE_ORDER = (
    "core-conformance-cases",
    "core-kernel",
    "core-forksafe",
    "core-rsibox",
    "core-survey",
    "core-doctor",
    "core-conformance-toylab",
    "template-adapter",
    "fastvideo-adapter",
    "fastvideo-services",
    "fastvideo-conformance",
    "vllm-adapter",
    "vllm-services",
    "megatron-livelab",
    "megatron-tools",
    "verl-adapter",
    "verl-tools",
    "verl-conformance",
    "miles-adapter",
    "miles-conformance",
    "sglang-adapter",
    "sglang-services",
    "sglang-conformance",
    "review-checkers",
    "scripts-tests",
)
ALL_SUITES = frozenset(ALL_SUITE_ORDER)

app = modal.App("rsibox-gpu-ci")

core_gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libnuma1")
    .pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu128",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install("pytest", "numpy", "nvidia-ml-py", "setuptools", "wheel", "ray")
)

vllm_gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libnuma1")
    .pip_install("vllm", "pytest", "numpy", "nvidia-ml-py", "setuptools")
)


def select_suites(suites_csv: str) -> tuple[list[str] | None, bool]:
    """Parse optional comma-separated `--suites` filter into `(core_suites_or_None, run_vllm)`."""
    selected = [s.strip() for s in suites_csv.split(",") if s.strip()]
    if not selected:
        return None, True
    unknown = sorted(set(selected) - ALL_SUITES)
    if unknown:
        raise ValueError(f"unknown suite(s): {', '.join(unknown)}")
    selected_set = set(selected)
    core = [name for name in CORE_SUITE_ORDER if name in selected_set]
    return core, ("vllm-services" in selected_set)


def build_core_steps(src: Path, py: str, res_dir: Path, unittest_json_path: Path) -> list[dict]:
    """Return ordered suite specs for `run_core_gpu`; keeps engine PYTHONPATHs strictly isolated."""
    uj = str(unittest_json_path)

    def ut(name: str, start: str, top: str | None = None, pattern: str | None = None, cwd: str = ".") -> dict:
        argv = [py, uj, "--json", str(res_dir / f"{name}.json"), "-s", start]
        if top:
            argv += ["-t", top]
        if pattern:
            argv += ["-p", pattern]
        return {"name": name, "kind": "unittest", "group": "core", "argv": argv, "cwd": cwd, "env": {}}

    def cmd(name: str, argv: list[str], cwd: str = ".", env: dict | None = None) -> dict:
        return {"name": name, "kind": "cmd", "group": "core", "argv": argv, "cwd": cwd, "env": env or {}}

    steps = [
        ut("core-conformance-cases", "conformance", ".", cwd="core/livelab"),
        ut("core-kernel", "core/livelab/kernel", "core/livelab"),
        ut("core-forksafe", "core/livelab", pattern="test_forksafe.py"),
        ut("core-rsibox", "core/livelab/rsibox", "core/livelab"),
        ut("core-survey", "core/livelab/survey", "core/livelab"),
        cmd("core-doctor", [py, "-m", "rsibox", "doctor"], env={"PYTHONPATH": str(src / "core/livelab")}),
        cmd("core-conformance-toylab", [py, "-m", "conformance", "--lab", "kernel.toy_lab:ToyLab"],
            cwd="core/livelab"),
        ut("template-adapter", "template/livelab/adapter", "template/livelab"),
        ut("fastvideo-adapter", "repos/fastvideo/livelab/adapter", "repos/fastvideo/livelab"),
        ut("fastvideo-services", "repos/fastvideo/livelab/services", "repos/fastvideo/livelab"),
        cmd("fastvideo-conformance",
            [py, "-m", "conformance", "--lab", "adapter.toy_lab:ToyFastVideoLab"],
            cwd="core/livelab", env={"PYTHONPATH": str(src / "repos/fastvideo/livelab")}),
        ut("vllm-adapter", "repos/vllm/livelab/adapter", "repos/vllm/livelab"),
        ut("megatron-livelab", "repos/megatron/livelab/rsibox_megatron", "repos/megatron/livelab"),
    ]
    if (src / "repos/megatron/tools").is_dir():
        steps.append(ut("megatron-tools", "repos/megatron/tools"))
    steps += [
        ut("verl-adapter", "repos/verl/livelab/adapter", "repos/verl/livelab"),
        ut("verl-tools", "repos/verl/tools", "repos/verl"),
        cmd("verl-conformance", [py, "-m", "rsibox", "conformance", "verl"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("miles-adapter", "repos/miles/livelab/adapter", "repos/miles/livelab"),
        cmd("miles-conformance", [py, "-m", "rsibox", "conformance", "miles"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("sglang-adapter", "repos/sglang/livelab/adapter", "repos/sglang/livelab"),
        ut("sglang-services", "repos/sglang/livelab/services", "repos/sglang/livelab"),
        cmd("sglang-conformance", [py, "-m", "rsibox", "conformance", "sglang"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("review-checkers", ".github/review", ".github/review"),
        ut("scripts-tests", "scripts", pattern="test_*.py"),
    ]
    return steps


def build_vllm_step(py: str, res_dir: Path, vllm_model: str = "") -> dict:
    """Return the isolated `vllm-services` pytest step spec."""
    step_env = {"RSIBOX_VLLM_PYTHON": py}
    if vllm_model:
        step_env["RSIBOX_VLLM_MODEL"] = vllm_model
        step_env["RSIBOX_GPU_E2E"] = "1"
    return {
        "name": "vllm-services",
        "kind": "pytest",
        "group": "vllm",
        "argv": [
            py, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider",
            f"--junitxml={res_dir / 'vllm-services.xml'}",
            "repos/vllm/livelab/services",
        ],
        "cwd": ".",
        "env": step_env,
    }


def parse_results(step: dict, res_dir: Path) -> list[dict]:
    """Parse machine-readable test outcomes from unittest JSON or pytest JUnit XML."""
    if step["kind"] == "unittest":
        f = res_dir / f"{step['name']}.json"
        return json.loads(f.read_text())["tests"] if f.exists() else []
    if step["kind"] == "pytest":
        f = res_dir / f"{step['name']}.xml"
        if not f.exists():
            return []
        tests = []
        for tc in ET.parse(f).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            outcome, detail = "passed", ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    outcome = "failed" if child.tag == "failure" else "error"
                    detail = (child.get("message") or "")[:2000]
                elif child.tag == "skipped":
                    xf = (child.get("type") or "") == "pytest.xfail"
                    outcome = "xfailed" if xf else "skipped"
                    detail = child.get("message") or ""
            tests.append({"id": tid, "outcome": outcome, "seconds": float(tc.get("time") or 0),
                          "detail": detail})
        return tests
    return []


def _extract_tarball(tarball_bytes: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tf:
        try:
            tf.extractall(dest, filter="tar")
        except TypeError:
            tf.extractall(dest)


def _collect_gpu_info() -> dict:
    info: dict = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                rows.append({"index": parts[0], "name": parts[1], "driver_version": parts[2],
                             "memory_total": parts[3]})
        info["nvidia_smi"] = rows
        if rows:
            info["gpu_name"] = rows[0]["name"]
            info["driver_version"] = rows[0]["driver_version"]
            info["memory_total"] = rows[0]["memory_total"]
            info["gpu_count"] = len(rows)
    except Exception as exc:
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch  # noqa: PLC0415
        info["torch_version"] = str(torch.__version__)
        info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["torch_device_count"] = int(torch.cuda.device_count())
            info["torch_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import vllm  # noqa: PLC0415
        info["vllm_version"] = str(getattr(vllm, "__version__", "unknown"))
    except Exception:
        pass
    return info


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_step(step: dict, src: Path, res_dir: Path, timeout: int, label: str) -> dict:
    strip_exact = {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "ORT_INTER_OP_NUM_THREADS",
        "ORT_INTRA_OP_NUM_THREADS",
    }
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("RSIBOX_", "PYTHON", "VIRTUAL_ENV")) and k not in strip_exact}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RSIBOX_TEST_GPU"] = "1"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env.update(step.get("env", {}))

    header = f"$ (cd {step['cwd']} && {shlex.join(step['argv'])})\n"
    t0 = time.monotonic()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as log_tmp:
        log_tmp.write(header)
        log_tmp.flush()
        proc = subprocess.Popen(
            step["argv"],
            cwd=src / step["cwd"],
            env=env,
            stdout=log_tmp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            rc = "timeout"
        except BaseException:
            _kill_group(proc)
            raise
        log_tmp.seek(0)
        log_text = log_tmp.read()

    elapsed = round(time.monotonic() - t0, 1)
    tests = parse_results(step, res_dir)
    if step["kind"] == "cmd":
        tests = [{
            "id": step["name"],
            "outcome": "passed" if rc == 0 else "failed",
            "seconds": round(time.monotonic() - t0, 3),
            "detail": "" if rc == 0 else log_text[-2000:],
        }]
    counts: dict[str, int] = {}
    for t in tests:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    ok = rc == 0 and not counts.get("failed") and not counts.get("error")
    print(f"[{label}] {step['name']}: rc={rc} {counts} {elapsed}s", flush=True)
    return {
        "name": step["name"],
        "kind": step["kind"],
        "group": step["group"],
        "rc": rc,
        "ok": ok,
        "counts": counts,
        "seconds": elapsed,
        "tests": tests,
        "log_text": log_text,
    }


def _aggregate_counts(steps: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        for key, val in step["counts"].items():
            counts[key] = counts.get(key, 0) + val
    return counts


@app.function(image=core_gpu_image, gpu=DEFAULT_GPU, cpu=4.0, timeout=DEFAULT_REMOTE_TIMEOUT)
def run_core_gpu(
    tarball_bytes: bytes,
    unittest_json_src: str = "",
    suites: list[str] | None = None,
    timeout: int = 1200,
) -> str:
    """Unpack the trusted source archive and run core + non-vllm-services suites on GPU."""
    t0 = time.monotonic()
    gpu_info = _collect_gpu_info()
    tmp_root = Path(tempfile.mkdtemp(prefix="rsibox-core-", dir="/tmp"))
    try:
        src = tmp_root / "src"
        res_dir = tmp_root / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        _extract_tarball(tarball_bytes, src)
        uj_path = tmp_root / "unittest_json.py"
        if unittest_json_src:
            uj_path.write_text(unittest_json_src, encoding="utf-8")
        else:
            uj_path = src / ".github/modal-ci/unittest_json.py"

        steps = build_core_steps(src, sys.executable, res_dir, uj_path)
        if suites is not None:
            wanted = set(suites)
            steps = [s for s in steps if s["name"] in wanted]
        results = [_run_step(step, src, res_dir, timeout, "modal-core") for step in steps]
        return json.dumps({
            "group": "core",
            "gpu_info": gpu_info,
            "seconds": round(time.monotonic() - t0, 1),
            "counts": _aggregate_counts(results),
            "ok": all(r["ok"] for r in results),
            "steps": results,
        })
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@app.function(image=vllm_gpu_image, gpu=DEFAULT_GPU, cpu=4.0, timeout=DEFAULT_REMOTE_TIMEOUT)
def run_vllm_gpu(
    tarball_bytes: bytes,
    timeout: int = 1200,
    vllm_model: str = "",
) -> str:
    """Unpack the trusted source archive and run `vllm-services` pytest suite on GPU."""
    t0 = time.monotonic()
    gpu_info = _collect_gpu_info()
    tmp_root = Path(tempfile.mkdtemp(prefix="rsibox-vllm-", dir="/tmp"))
    try:
        src = tmp_root / "src"
        res_dir = tmp_root / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        _extract_tarball(tarball_bytes, src)
        step = build_vllm_step(sys.executable, res_dir, vllm_model=vllm_model)
        result = _run_step(step, src, res_dir, timeout, "modal-vllm")
        results = [result]
        return json.dumps({
            "group": "vllm",
            "gpu_info": gpu_info,
            "seconds": round(time.monotonic() - t0, 1),
            "counts": _aggregate_counts(results),
            "ok": all(r["ok"] for r in results),
            "steps": results,
        })
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _with_gpu_options(fn, gpu: str, remote_timeout: int):
    if hasattr(fn, "with_options"):
        return fn.with_options(gpu=gpu, timeout=remote_timeout)
    return fn


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _cancel_quietly(call) -> None:
    if call is not None and hasattr(call, "cancel"):
        try:
            call.cancel()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    tarball_path: str,
    output_json: str,
    gpu: str = "L4",
    timeout: int = 1200,
    vllm_model: str = "",
    suites: str = "",
):
    """Spawn `run_core_gpu` and `run_vllm_gpu` in parallel on Modal and write JSON results."""
    t0 = time.monotonic()
    tarball_bytes = Path(tarball_path).read_bytes()
    uj_file = Path(__file__).resolve().parent / "unittest_json.py"
    unittest_json_src = uj_file.read_text(encoding="utf-8") if uj_file.exists() else ""

    core_suites, want_vllm = select_suites(suites)
    want_core = core_suites is None or len(core_suites) > 0
    remote_timeout = max(1800, min(7200, timeout * 3))

    core_call = None
    vllm_call = None
    steps: list[dict] = []
    gpu_info: dict = {}
    try:
        if want_core:
            core_fn = _with_gpu_options(run_core_gpu, gpu, remote_timeout)
            core_call = core_fn.spawn(tarball_bytes, unittest_json_src, core_suites, timeout)
        if want_vllm:
            vllm_fn = _with_gpu_options(run_vllm_gpu, gpu, remote_timeout)
            vllm_call = vllm_fn.spawn(tarball_bytes, timeout, vllm_model)

        if core_call is not None:
            core_raw = core_call.get()
            core_call = None
            core_res = json.loads(core_raw) if isinstance(core_raw, str) else core_raw
            steps.extend(core_res["steps"])
            gpu_info["core"] = core_res.get("gpu_info", {})
        if vllm_call is not None:
            vllm_raw = vllm_call.get()
            vllm_call = None
            vllm_res = json.loads(vllm_raw) if isinstance(vllm_raw, str) else vllm_raw
            steps.extend(vllm_res["steps"])
            gpu_info["vllm"] = vllm_res.get("gpu_info", {})
    except BaseException:
        _cancel_quietly(core_call)
        _cancel_quietly(vllm_call)
        raise

    order = {name: idx for idx, name in enumerate(ALL_SUITE_ORDER)}
    steps.sort(key=lambda s: order.get(s["name"], len(order)))

    counts = _aggregate_counts(steps)
    ok = bool(steps) and all(step["ok"] for step in steps)
    payload = {
        "ok": ok,
        "gpu": gpu,
        "gpu_info": gpu_info,
        "seconds": round(time.monotonic() - t0, 1),
        "counts": counts,
        "steps": steps,
    }
    _write_atomic(Path(output_json), json.dumps(payload, indent=1))
    sys.exit(0 if ok else 1)
