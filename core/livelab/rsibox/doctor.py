"""Engine-agnostic read-only preflight diagnostics (`rsibox doctor --engine <name>`).

Discovers engines dynamically from `repos/*/` (keeping `toy` built-in) with zero
engine-specific code in `core/livelab/`: checks Python >= 3.11, stdlib, `RLIMIT_NOFILE`,
optional NVIDIA driver (`615+` for multi-GPU NCCL CRIU), `criu`, `cuda-checkpoint`,
`gpu_memory_service`, and discovered engine adapter/module readiness.
"""
import importlib.util, os, re, shutil, subprocess, sys
from typing import Any, Callable

MIN_PYTHON, MIN_NOFILE, MIN_CRIU_DRIVER_MAJOR = (3, 11, 0), 65536, 615
_CORE_STDLIB = ("argparse", "ast", "ctypes", "hashlib", "http.server", "json", "socket", "statistics", "subprocess", "threading")
_DRIVER_RE = re.compile(r"\b(\d{3,4})\.(\d+(?:\.\d+)?)\b")


def _finding(name: str, status: str, summary: str, remediation: str | None = None, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "summary": summary, "remediation": remediation, "details": details}

def _safe_find_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False

def _parse_driver_version(text: str) -> tuple[str, int] | None:
    match = _DRIVER_RE.search(text or "")
    return (f"{match.group(1)}.{match.group(2)}", int(match.group(1))) if match else None

def _probe_nvidia_driver(which_fn: Callable[[str], str | None]) -> str | None:
    if os.path.isfile("/proc/driver/nvidia/version"):
        try:
            with open("/proc/driver/nvidia/version", encoding="utf-8", errors="ignore") as handle:
                if parsed := _parse_driver_version(handle.read()):
                    return parsed[0]
        except OSError:
            pass
    if smi_path := which_fn("nvidia-smi"):
        try:
            res = subprocess.run([smi_path, "--query-gpu=driver_version", "--format=csv,noheader"],
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3.0)
            if res.returncode == 0 and (parsed := _parse_driver_version(res.stdout)):
                return parsed[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return None

def discover_engine_names(repo_root: str | None = None) -> list[str]:
    """Dynamically list available engine targets (`toy` plus every folder under `repos/`)."""
    from . import lab_cli
    resolved_root = lab_cli.find_repo_root(repo_root)
    names = ["toy"]
    if resolved_root and os.path.isdir(repos_dir := os.path.join(resolved_root, "repos")):
        names.extend(entry for entry in sorted(os.listdir(repos_dir))
                     if not entry.startswith(".") and entry != "__pycache__" and os.path.isdir(os.path.join(repos_dir, entry)))
    return names

def check_rlimit_nofile(limits: tuple[int, int] | None = None) -> dict[str, Any]:
    """Check `RLIMIT_NOFILE` soft/hard limits against `MIN_NOFILE` (65536)."""
    if limits is None:
        try:
            import resource
            soft_raw, hard_raw = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits = (int(soft_raw), int(hard_raw))
        except Exception:
            return _finding("rlimit_nofile", "warn", "RLIMIT_NOFILE unavailable on this platform",
                            f"Run on Linux/POSIX with open-file limit >= {MIN_NOFILE} for large VMM/IPC workloads.", soft=None, hard=None, recommended=MIN_NOFILE)
    soft, hard = int(limits[0]), int(limits[1])
    if soft < 0 or soft >= MIN_NOFILE:
        return _finding("rlimit_nofile", "ok", f"soft={'unlimited' if soft < 0 else soft}, hard={'unlimited' if hard < 0 else hard}", None, soft=soft, hard=hard, recommended=MIN_NOFILE)
    return _finding("rlimit_nofile", "warn", f"soft={soft}, hard={hard} (< {MIN_NOFILE} recommended for large VMM/IPC workloads)",
                    f"Raise open-file limit with 'ulimit -n {MIN_NOFILE}' (or configure /etc/security/limits.conf).", soft=soft, hard=hard, recommended=MIN_NOFILE)


def run_preflight(
    *, engine: str | None = None, repo_root: str | None = None,
    python_version: tuple[int, int, int] | None = None, nofile_limits: tuple[int, int] | None = None,
    driver_version: str | None = None, probe_driver: bool | None = None,
    which_fn: Callable[[str], str | None] = shutil.which, find_spec_fn: Callable[[str], bool] = _safe_find_spec,
) -> dict[str, Any]:
    """Run fast, read-only preflight diagnostics and return structured findings."""
    from . import lab_cli

    resolved_root = lab_cli.find_repo_root(repo_root)
    available_engines = discover_engine_names(resolved_root)
    if engine is None:
        selected: list[str] = []
    elif engine == "all":
        selected = list(available_engines)
    elif engine in available_engines:
        selected = [engine]
    else:
        raise ValueError(f"unknown engine {engine!r} (available: {', '.join((*available_engines, 'all'))})")

    py_ver = python_version or sys.version_info[:3]
    py_str = ".".join(str(part) for part in py_ver)
    missing_std = [module for module in _CORE_STDLIB if not find_spec_fn(module)]
    core = lab_cli._check_core_health()
    findings: list[dict[str, Any]] = [
        _finding("python_version", "ok", f"Python {py_str} (>= 3.11)", None, version=py_str) if tuple(py_ver) >= MIN_PYTHON
        else _finding("python_version", "fail", f"Python {py_str} (< 3.11 required)", "Install Python 3.11 or newer.", version=py_str),
        _finding("core_stdlib", "ok", f"{len(_CORE_STDLIB)}/{len(_CORE_STDLIB)} core stdlib modules available", None) if not missing_std
        else _finding("core_stdlib", "fail", f"missing stdlib modules: {missing_std}", "Use standard CPython 3.11+.", missing=missing_std),
        _finding("core_livelab", "ok", f"kernel {len(core['kernel']['loaded'])} modules, conformance {core['conformance']['case_count']}/11 cases", None, case_count=core["conformance"]["case_count"]) if core["ok"]
        else _finding("core_livelab", "fail", "core kernel or conformance suite failed to load", "Ensure core/livelab is on PYTHONPATH.", missing=core["kernel"]["missing"], conformance_error=core["conformance"]["error"]),
        check_rlimit_nofile(nofile_limits),
    ]
    should_probe = (engine != "toy") if probe_driver is None else probe_driver
    driver_raw = driver_version if driver_version is not None else (_probe_nvidia_driver(which_fn) if should_probe else None)
    needs_gpu = engine != "toy"
    if driver_raw is not None:
        if (parsed := _parse_driver_version(driver_raw)) is None:
            findings.append(_finding("nvidia_driver", "warn", f"unrecognized NVIDIA driver string {driver_raw!r}",
                                     f"Verify NVIDIA driver installation ({MIN_CRIU_DRIVER_MAJOR}+ for multi-GPU NCCL CRIU).", version=driver_raw, criu_nccl_ready=False))
        else:
            ver_str, major = parsed
            ok_driver = major >= MIN_CRIU_DRIVER_MAJOR
            findings.append(_finding("nvidia_driver", "ok" if ok_driver else "warn",
                                     f"NVIDIA driver {ver_str} ({'>=' if ok_driver else '<'} {MIN_CRIU_DRIVER_MAJOR}; multi-GPU NCCL CRIU {'capable' if ok_driver else f'requires {MIN_CRIU_DRIVER_MAJOR}+'})",
                                     None if ok_driver else f"Upgrade to NVIDIA driver {MIN_CRIU_DRIVER_MAJOR}+ for multi-GPU NCCL CRIU checkpoints.",
                                     version=ver_str, major=major, criu_nccl_ready=ok_driver))
    else:
        findings.append(_finding("nvidia_driver", "warn" if needs_gpu else "ok",
                                 "NVIDIA driver probe skipped for CPU-only toy engine" if not should_probe else "NVIDIA driver not detected (CPU-only host)",
                                 f"Install NVIDIA driver ({MIN_CRIU_DRIVER_MAJOR}+ for multi-GPU NCCL CRIU) on GPU nodes." if needs_gpu else None, version=None, criu_nccl_ready=False))

    for name, binary, hint in (("criu", "criu", "Install CRIU ('apt-get install criu') for process checkpoint/restore."),
                               ("cuda_checkpoint", "cuda-checkpoint", "Install NVIDIA cuda-checkpoint on PATH for CUDA context checkpoint/restore.")):
        found = bool(which_fn(binary))
        findings.append(_finding(name, "ok" if found else "warn", f"{binary} available" if found else f"{binary} not found on PATH", None if found else hint, available=found))
    gms_ok = find_spec_fn("gpu_memory_service")
    findings.append(_finding("gpu_memory_service", "ok" if gms_ok else "warn", "gpu_memory_service importable" if gms_ok else "gpu_memory_service not installed",
                             None if gms_ok else "Install gpu_memory_service to share pinned GPU weights across restarts.", available=gms_ok))

    if selected:
        discovered = {item["name"]: item for item in lab_cli.discover_labs(resolved_root)}
        for engine_name in selected:
            entry = discovered.get(engine_name)
            if engine_name == "toy":
                toy_ok = find_spec_fn("kernel.toy_lab")
                findings.append(_finding("engine:toy:adapter", "ok" if toy_ok else "fail", "built-in toy lab ready (kernel.toy_lab:ToyLab)" if toy_ok else "built-in toy lab missing",
                                         None if toy_ok else "Verify core/livelab/kernel/toy_lab.py is present.", engine="toy", builtin=True))
                continue
            if entry and not entry.get("manifest_errors") and (entry.get("toy_spec") or entry.get("real_spec")):
                findings.append(_finding(f"engine:{engine_name}:adapter", "ok", f"adapter present (toy={entry.get('toy_spec') or '-'}, real={entry.get('real_spec') or '-'})",
                                         None, engine=engine_name, toy_spec=entry.get("toy_spec"), real_spec=entry.get("real_spec")))
            elif entry and entry.get("manifest_errors"):
                findings.append(_finding(f"engine:{engine_name}:adapter", "fail", f"invalid livelab.json: {'; '.join(entry['manifest_errors'])}",
                                         f"Fix repos/{engine_name}/livelab/livelab.json schema.", engine=engine_name))
            else:
                findings.append(_finding(f"engine:{engine_name}:adapter", "fail", f"repos/{engine_name}/ exists without runnable livelab/ adapter",
                                         f"Add repos/{engine_name}/livelab/ following template/livelab/.", engine=engine_name))
            mod_ok = find_spec_fn(engine_name)
            findings.append(_finding(f"engine:{engine_name}:module", "ok" if mod_ok else "warn",
                                     f"module importable: {engine_name}" if mod_ok else f"module not installed: {engine_name}",
                                     None if mod_ok else f"Install {engine_name} in the active Python environment to run real workloads.",
                                     engine=engine_name, **({} if mod_ok else {"missing": [engine_name]})))

    counts = {status: sum(1 for item in findings if item["status"] == status) for status in ("ok", "warn", "fail")}
    core_ok = all(item["status"] == "ok" for item in findings if item["name"] in ("python_version", "core_stdlib", "core_livelab"))
    return {"ok": core_ok and counts["fail"] == 0, "strict_ok": core_ok and counts["fail"] == 0 and counts["warn"] == 0,
            "core_ok": core_ok, "engine": engine, "engines_checked": selected, "available_engines": available_engines,
            "python": py_str, "platform": sys.platform, "livelab_root": os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "repo_root": resolved_root, "findings": findings, "summary": {**counts, "total": len(findings), "core_ok": core_ok}}


def format_preflight_report(report: dict[str, Any]) -> str:
    lines = [f"engine:   {report['engine']}", f"python:   {report['python']} ({report['platform']})", f"core_ok:  {'yes' if report['core_ok'] else 'NO'}", ""]
    width = max((len(item["name"]) for item in report["findings"]), default=12)
    for item in report["findings"]:
        lines.append(f"[{item['status'].upper():<4}] {item['name']:<{width}}  {item['summary']}")
        if item.get("remediation"):
            lines.append(f"       hint: {item['remediation']}")
    summary = report["summary"]
    return "\n".join([*lines, f"\nsummary: {summary['ok']} ok, {summary['warn']} warn, {summary['fail']} fail (total {summary['total']})"])
