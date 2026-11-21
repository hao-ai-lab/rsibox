"""Unified, engine-agnostic CLI for FastRSI (`fastrsi` / `python3 -m fastrsi`).

Commands:
- `fastrsi list [--repo-root PATH] [--json]`
    Dynamically discovers the built-in `toy` lab, `template`, and every engine under
    `<repo_root>/repos/<engine>/livelab/` without hardcoding engine names or importing them.
- `fastrsi conformance <target> [--toy | --real] [--only ...] [--receipts ...] [--json]`
    Runs `core/livelab/conformance` against a discovered engine name (`--toy` by default,
    `--real` for the real-engine conformance class) or any explicit `module:Class` spec.
- `fastrsi survey ...`
    Delegates to `core/livelab/survey`.
- `fastrsi ctl (--sock PATH | --file-channel DIR | --mailbox DIR) <verb> [...]`
    Transport-agnostic JSON control client for any running live lab.
- `fastrsi doctor [--repo-root PATH] [--json]`
    Checks core `kernel` & `conformance` health, `forksafe` environment audit, `AF_UNIX`
    socket path limits, and every discovered engine lab's `LabUnderTest` API and lazy
    GPU/torch-free importability.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
import subprocess
import sys

import forksafe

from . import ipc

_CORE_KERNEL_MODULES = (
    "kernel.coordinator", "kernel.findings", "kernel.journal", "kernel.locate",
    "kernel.receipt", "kernel.session", "kernel.swap", "kernel.witness", "kernel.toy_lab",
)
_LAB_METHODS = ("install", "check", "reset", "uninstall", "status", "patches", "fresh_check", "inject", "close")
# Generic GPU/accelerator runtime packages that must stay lazy at lab-module import time.
_GPU_RUNTIME_LIBS = ("torch", "cuda", "cupy", "triton", "pynvml", "nvidia", "tensorrt", "jax", "tensorflow")
_MANIFEST_KEYS = frozenset(("toy", "real"))
_PROBE_SCRIPT = (
    "import importlib, json, sys\n"
    "spec, req_csv, gpu_csv = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "mod_name, _, cls_name = spec.partition(':')\n"
    "req = [m for m in req_csv.split(',') if m]\n"
    "gpu_set = set(g for g in gpu_csv.split(',') if g)\n"
    "before = set(sys.modules)\n"
    "out = {'ok': False, 'spec': spec, 'gpu_imports': [], 'missing_methods': [], 'has_launch_environment': False, 'error': None}\n"
    "try:\n"
    "    mod = importlib.import_module(mod_name)\n"
    "    loaded = sorted({m.split('.')[0] for m in (set(sys.modules) - before) if m.split('.')[0] in gpu_set})\n"
    "    out['gpu_imports'] = loaded\n"
    "    cls = getattr(mod, cls_name, None)\n"
    "    if cls is None:\n"
    "        out['error'] = f'class {cls_name!r} not found in {mod_name}'\n"
    "    else:\n"
    "        out['has_launch_environment'] = callable(getattr(cls, 'launch_environment', None))\n"
    "        missing = [m for m in req if not callable(getattr(cls, m, None))]\n"
    "        out['missing_methods'] = missing\n"
    "        if missing:\n"
    "            out['error'] = f'missing LabUnderTest methods: {missing}'\n"
    "        elif loaded:\n"
    "            out['error'] = f'eager GPU/runtime import(s): {loaded}'\n"
    "        else:\n"
    "            out['ok'] = True\n"
    "except BaseException as exc:\n"
    "    loaded = sorted({m.split('.')[0] for m in (set(sys.modules) - before) if m.split('.')[0] in gpu_set})\n"
    "    out['gpu_imports'] = loaded\n"
    "    out['error'] = f'{type(exc).__name__}: {exc}'\n"
    "print('FASTRSI_DOCTOR_PROBE:' + json.dumps(out))\n"
)


def find_repo_root(start: str | None = None) -> str | None:
    """Walk upward from `start` (or this package, then cwd) for a repo containing `core/livelab`."""
    seeds = [start] if start else [
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        os.getcwd(),
    ]
    for seed in seeds:
        if not seed:
            continue
        cur = os.path.abspath(seed)
        for _ in range(8):
            if os.path.isdir(os.path.join(cur, "core", "livelab")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return None


def _lab_classes_in_file(path: str) -> list[str]:
    """AST-scan `path` (no imports, no side effects) for top-level `*Lab` class names."""
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Lab")
    ]


def _is_valid_lab_spec(value: object) -> bool:
    if not isinstance(value, str) or ":" not in value:
        return False
    mod_name, sep, cls_name = value.partition(":")
    if not sep or ":" in cls_name or not mod_name or not cls_name:
        return False
    return all(part.isidentifier() for part in mod_name.split(".")) and cls_name.isidentifier()


def _load_livelab_manifest(manifest_path: str) -> tuple[dict | None, list[str]]:
    """Read and validate `<livelab_dir>/livelab.json` (`{\"toy\": spec|null, \"real\": spec|null}`)."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, [f"invalid livelab.json: {type(exc).__name__}: {exc}"]
    if not isinstance(raw, dict):
        return None, [f"invalid livelab.json: expected JSON object, got {type(raw).__name__}"]
    errors: list[str] = []
    unknown = sorted(set(raw) - _MANIFEST_KEYS)
    if unknown:
        errors.append(f"invalid livelab.json: unknown key(s) {unknown} (allowed: {sorted(_MANIFEST_KEYS)})")
    parsed: dict[str, str | None] = {"toy": None, "real": None}
    for key in ("toy", "real"):
        val = raw.get(key)
        if val is None:
            parsed[key] = None
        elif _is_valid_lab_spec(val):
            parsed[key] = val
        else:
            errors.append(f"invalid livelab.json: {key!r} must be null or 'module:Class', got {val!r}")
    return (None, errors) if errors else (parsed, [])


def _discover_livelab_dir(name: str, livelab_dir: str) -> dict:
    """Inspect one `<repo>/repos/<name>/livelab` folder via optional `livelab.json` or AST."""
    manifest_path = os.path.join(livelab_dir, "livelab.json")
    if os.path.isfile(manifest_path):
        manifest, errors = _load_livelab_manifest(manifest_path)
        entry = {
            "name": name,
            "livelab_dir": livelab_dir,
            "toy_spec": manifest["toy"] if manifest else None,
            "real_spec": manifest["real"] if manifest else None,
        }
        if errors:
            entry["manifest_errors"] = errors
        return entry

    toy_spec = None
    real_spec = None
    pkg_names = sorted(
        entry
        for entry in os.listdir(livelab_dir)
        if entry not in ("services", "__pycache__")
        and os.path.isfile(os.path.join(livelab_dir, entry, "__init__.py"))
    )
    for pkg in pkg_names:
        pkg_dir = os.path.join(livelab_dir, pkg)
        if toy_spec is None:
            toy_file = os.path.join(pkg_dir, "toy_lab.py")
            if os.path.isfile(toy_file):
                classes = _lab_classes_in_file(toy_file)
                if classes:
                    toy_spec = f"{pkg}.toy_lab:{classes[0]}"
        if real_spec is None:
            for stem in ("conformance_lab", "lab"):
                cand = os.path.join(pkg_dir, f"{stem}.py")
                if os.path.isfile(cand):
                    classes = _lab_classes_in_file(cand)
                    if classes:
                        real_spec = f"{pkg}.{stem}:{classes[0]}"
                        break
    return {
        "name": name,
        "livelab_dir": livelab_dir,
        "toy_spec": toy_spec,
        "real_spec": real_spec,
    }


def discover_labs(repo_root: str | None = None) -> list[dict]:
    """Discover all runnable labs (built-in `toy`, optional `template`, and `repos/*/livelab`)."""
    livelab_core = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    labs = [
        {
            "name": "toy",
            "livelab_dir": livelab_core,
            "toy_spec": "kernel.toy_lab:ToyLab",
            "real_spec": None,
        }
    ]
    root = find_repo_root(repo_root)
    if root is None:
        return labs
    tmpl_dir = os.path.join(root, "template", "livelab")
    if os.path.isdir(tmpl_dir):
        entry = _discover_livelab_dir("template", tmpl_dir)
        if entry["toy_spec"] is None and entry["real_spec"] is not None:
            entry["toy_spec"] = entry["real_spec"]
            entry["real_spec"] = None
        labs.append(entry)
    repos_dir = os.path.join(root, "repos")
    if os.path.isdir(repos_dir):
        for name in sorted(os.listdir(repos_dir)):
            livelab_dir = os.path.join(repos_dir, name, "livelab")
            if os.path.isdir(livelab_dir):
                labs.append(_discover_livelab_dir(name, livelab_dir))
    return labs


def resolve_lab(target: str, *, real: bool = False, repo_root: str | None = None) -> tuple[str, str | None]:
    """Resolve `<target>` to `(module:Class, extra_sys_path_or_None)`."""
    if ":" in target:
        if not _is_valid_lab_spec(target):
            raise ValueError(f"invalid lab spec {target!r} (expected 'module:Class')")
        return target, None
    by_name = {entry["name"]: entry for entry in discover_labs(repo_root)}
    entry = by_name.get(target)
    if entry is None:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"unknown lab {target!r} (known: {known}; or pass module:Class)")
    if entry.get("manifest_errors"):
        raise ValueError(f"lab {target!r}: {'; '.join(entry['manifest_errors'])}")
    spec = entry["real_spec"] if real else entry["toy_spec"]
    if not spec:
        if not real and entry["real_spec"]:
            raise ValueError(
                f"lab {target!r} has no toy conformance class; pass --real to run {entry['real_spec']}"
            )
        kind = "real" if real else "toy"
        raise ValueError(f"lab {target!r} has no {kind} conformance class")
    return spec, entry["livelab_dir"]


def _cmd_list(args) -> int:
    labs = discover_labs(args.repo_root)
    if args.json:
        print(json.dumps({"labs": labs}, indent=1))
        return 0
    w = max((len(x["name"]) for x in labs), default=4)
    print(f"{'lab':<{w}}  {'toy':<38}  real")
    for entry in labs:
        print(f"{entry['name']:<{w}}  {str(entry['toy_spec'] or '-'):<38}  {entry['real_spec'] or '-'}")
    return 0


def _cmd_conformance(args, engine_args) -> int:
    try:
        spec, extra_path = resolve_lab(args.target, real=args.real, repo_root=args.repo_root)
    except ValueError as exc:
        print(f"fastrsi conformance: {exc}", file=sys.stderr)
        return 2
    if args.path:
        extra_path = os.path.abspath(args.path)
    inserted = False
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
        inserted = True
    livelab_core = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    prev_py = os.environ.get("PYTHONPATH")
    parts = [p for p in (prev_py.split(os.pathsep) if prev_py else []) if p]
    for needed in (livelab_core, extra_path):
        if needed and needed not in parts:
            parts.insert(0, needed)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    try:
        from conformance import __main__ as conf_main

        conf_argv = ["--lab", spec]
        if args.only:
            try:
                conf_main.select_cases(args.only)
            except ValueError as exc:
                print(f"fastrsi conformance: {exc}", file=sys.stderr)
                return 2
            conf_argv += ["--only", args.only]
        if args.receipts:
            conf_argv += ["--receipts", args.receipts]
        if args.json:
            conf_argv.append("--json")
        if engine_args[:1] == ["--"]:
            engine_args = engine_args[1:]
        if engine_args:
            lab = conf_main.load(spec)
            configure = getattr(lab, "launch_environment", None)
            if not args.real or configure is None:
                print(f"fastrsi conformance: {spec} does not accept launch arguments", file=sys.stderr)
                return 2
            env = {**os.environ, **configure(engine_args)}
            env.setdefault("PYTHONUNBUFFERED", "1")
            sys.stdout.flush()
            sys.stderr.flush()
            return subprocess.run(
                [sys.executable, "-u", "-m", "conformance", *conf_argv],
                env=env,
            ).returncode
        return conf_main.main(conf_argv)
    finally:
        if prev_py is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prev_py
        if inserted and extra_path in sys.path:
            sys.path.remove(extra_path)


def _cmd_survey(rest: list[str]) -> int:
    from survey import __main__ as survey_main

    return int(survey_main.main(rest) or 0)


def _cmd_ctl(args) -> int:
    try:
        req = json.loads(args.payload) if args.payload else {}
    except ValueError as exc:
        print(f"fastrsi ctl: invalid JSON for --payload: {exc}", file=sys.stderr)
        return 2
    if not isinstance(req, dict):
        print("fastrsi ctl: --payload must be a JSON object", file=sys.stderr)
        return 2
    if args.verb:
        req[args.op_key] = args.verb
    for pairs, parse_json, label in ((args.arg, False, "--arg"), (args.json_arg, True, "--json-arg")):
        for pair in pairs:
            key, sep, val = pair.partition("=")
            if not sep:
                print(f"fastrsi ctl: {label} expects KEY={'JSON' if parse_json else 'VALUE'}, got {pair!r}", file=sys.stderr)
                return 2
            if parse_json:
                try:
                    val = json.loads(val)
                except ValueError as exc:
                    print(f"fastrsi ctl: invalid JSON for --json-arg {key!r}: {exc}", file=sys.stderr)
                    return 2
            req[key] = val
    try:
        if args.sock:
            reply = ipc.unix_json_call(args.sock, req, timeout=args.timeout)
        elif args.file_channel:
            reply = ipc.file_channel_call(args.file_channel, req, timeout=args.timeout)
        else:
            reply = ipc.mailbox_call(args.mailbox, req, timeout=args.timeout)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    if not args.keep_receipt and isinstance(reply, dict):
        reply = {k: v for k, v in reply.items() if k != "receipt"}
    print(json.dumps(reply, indent=1, default=str))
    if isinstance(reply, dict) and reply.get("outcome") in ("refused", "aborted"):
        return 2
    return 0 if isinstance(reply, dict) and reply.get("ok") else 1


def _check_core_health() -> dict:
    """Verify stdlib `kernel` modules and `conformance` cases load cleanly."""
    loaded, missing = [], []
    for mod_name in _CORE_KERNEL_MODULES:
        try:
            importlib.import_module(mod_name)
            loaded.append(mod_name)
        except Exception as exc:  # noqa: BLE001
            missing.append({"module": mod_name, "error": f"{type(exc).__name__}: {exc}"})
    conf_ok = False
    case_names: list[str] = []
    conf_error = None
    try:
        conf_cases = importlib.import_module("conformance.cases")
        case_names = list(getattr(conf_cases, "CASES", ()))
        conf_ok = len(case_names) == 11
        if not conf_ok:
            conf_error = f"expected 11 conformance cases, found {len(case_names)}"
    except Exception as exc:  # noqa: BLE001
        conf_error = f"{type(exc).__name__}: {exc}"
    return {
        "ok": not missing and conf_ok,
        "kernel": {"ok": not missing, "loaded": loaded, "missing": missing},
        "conformance": {
            "ok": conf_ok,
            "case_count": len(case_names),
            "cases": case_names,
            "error": conf_error,
        },
    }


def _probe_error(spec: str, msg: str) -> dict:
    return {
        "ok": False, "spec": spec, "gpu_imports": [], "missing_methods": [],
        "has_launch_environment": False, "error": msg,
    }


def _probe_lab_spec(livelab_dir: str, spec: str, livelab_core: str, *, timeout: float = 15.0) -> dict:
    """Import `spec` in an isolated subprocess, checking `LabUnderTest` API and lazy GPU imports."""
    py_parts = [livelab_dir] if livelab_dir == livelab_core else [livelab_dir, livelab_core]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(py_parts), "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    try:
        proc_res = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, spec, ",".join(_LAB_METHODS), ",".join(_GPU_RUNTIME_LIBS)],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _probe_error(spec, f"{type(exc).__name__}: {exc}")
    prefix = "FASTRSI_DOCTOR_PROBE:"
    for line in reversed(proc_res.stdout.splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except ValueError:
                break
    detail = (proc_res.stderr or proc_res.stdout).strip().splitlines()
    return _probe_error(spec, detail[-1] if detail else f"probe exited with code {proc_res.returncode}")


def doctor_report(repo_root: str | None = None, *, timeout: float = 15.0) -> dict:
    """Audit core kernel/conformance health, environment, and all discovered engine labs."""
    livelab_core = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    resolved_root = find_repo_root(repo_root)
    discovered = discover_labs(repo_root)
    by_name = {entry["name"]: entry for entry in discovered}
    entries = [
        {**e, "kind": "builtin" if e["name"] == "toy" else ("template" if e["name"] == "template" else "engine"), "registered": True}
        for e in discovered
    ]
    if resolved_root and os.path.isdir(repos_dir := os.path.join(resolved_root, "repos")):
        for name in sorted(os.listdir(repos_dir)):
            full = os.path.join(repos_dir, name)
            if not name.startswith(".") and name != "__pycache__" and os.path.isdir(full) and name not in by_name:
                entries.append({
                    "name": name, "kind": "engine", "registered": False,
                    "livelab_dir": os.path.join(full, "livelab"), "toy_spec": None, "real_spec": None,
                })
    jobs = [(idx, slot, entry["livelab_dir"], spec)
            for idx, entry in enumerate(entries) for slot in ("toy", "real") if (spec := entry.get(f"{slot}_spec"))]
    probe_results: dict[tuple[int, str], dict] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
            futs = {pool.submit(_probe_lab_spec, ld, sp, livelab_core, timeout=timeout): (i, sl) for i, sl, ld, sp in jobs}
            probe_results = {key: fut.result() for fut, key in futs.items()}
    lab_reports: list[dict] = []
    for idx, entry in enumerate(entries):
        toy_res, real_res = probe_results.get((idx, "toy")), probe_results.get((idx, "real"))
        errors, gpu_imports = list(entry.get("manifest_errors") or ()), []
        if not errors and not entry["registered"]:
            errors.append("missing livelab/ directory")
        elif not errors and not entry["toy_spec"] and not entry["real_spec"]:
            errors.append("no toy or real Lab class discovered")
        for slot, res in (("toy", toy_res), ("real", real_res)):
            if res is None:
                continue
            gpu_imports.extend(res.get("gpu_imports") or [])
            if not res.get("ok"):
                errors.append(f"{slot}: {res.get('error')}")
        lab_reports.append({
            "name": entry["name"], "kind": entry["kind"], "livelab_dir": entry["livelab_dir"],
            "toy_spec": entry["toy_spec"], "real_spec": entry["real_spec"],
            "toy": toy_res, "real": real_res,
            "cli_launch": None if real_res is None else bool(real_res.get("has_launch_environment")),
            "gpu_imports": sorted(set(gpu_imports)), "ok": not errors, "errors": errors,
        })
    core = _check_core_health()
    engine_reports = [item for item in lab_reports if item["kind"] == "engine"]
    return {
        "ok": bool(core["ok"] and all(item["ok"] for item in lab_reports)),
        "python": sys.version.split()[0], "platform": sys.platform,
        "livelab_root": livelab_core, "repo_root": resolved_root,
        "max_unix_sock_bytes": ipc.MAX_SOCK_PATH,
        "discovered_labs": [x["name"] for x in discovered],
        "engines": [x["name"] for x in engine_reports],
        "core": core, "labs": lab_reports, "env_findings": forksafe.audit_system_environment(),
        "summary": {
            "labs_total": len(lab_reports), "labs_ok": sum(1 for x in lab_reports if x["ok"]),
            "engines_total": len(engine_reports), "engines_ok": sum(1 for x in engine_reports if x["ok"]),
            "real_cli_launch": sum(1 for x in engine_reports if x["cli_launch"]),
            "gpu_import_violations": sum(1 for x in lab_reports if x["gpu_imports"]),
        },
    }


def _cmd_doctor(args) -> int:
    report = doctor_report(args.repo_root)
    if args.json:
        print(json.dumps(report, indent=1, default=str))
        return 0 if report["ok"] else 1
    core = report["core"]
    k_ok = len(core["kernel"]["loaded"])
    k_tot = k_ok + len(core["kernel"]["missing"])
    c_cnt = core["conformance"]["case_count"]
    print(f"python:        {report['python']} ({report['platform']})")
    print(f"livelab_root:  {report['livelab_root']}")
    print(f"repo_root:     {report['repo_root']}")
    print(f"max_sock_path: {report['max_unix_sock_bytes']} bytes")
    print(f"core:          {'ok' if core['ok'] else 'FAIL'} (kernel {k_ok}/{k_tot}, conformance {c_cnt}/11)")
    print(f"env_findings:  {len(report['env_findings'])}")
    print()
    w_name = max((len(x["name"]) for x in report["labs"]), default=4)
    print(f"{'lab':<{w_name}}  {'kind':<8}  {'toy':<5}  {'real':<5}  {'cli_launch':<10}  {'gpu_imports':<11}  status")
    for item in report["labs"]:
        toy_s = "-" if item["toy"] is None else ("ok" if item["toy"]["ok"] else "FAIL")
        real_s = "-" if item["real"] is None else ("ok" if item["real"]["ok"] else "FAIL")
        cli_s = "-" if item["cli_launch"] is None else ("yes" if item["cli_launch"] else "no")
        gpu_s = ",".join(item["gpu_imports"]) if item["gpu_imports"] else "none"
        st_s = "ok" if item["ok"] else "FAIL: " + "; ".join(item["errors"])
        print(f"{item['name']:<{w_name}}  {item['kind']:<8}  {toy_s:<5}  {real_s:<5}  {cli_s:<10}  {gpu_s:<11}  {st_s}")
    s = report["summary"]
    print(f"\nsummary: {s['labs_ok']}/{s['labs_total']} labs ok ({s['engines_ok']}/{s['engines_total']} engines), "
          f"{s['gpu_import_violations']} GPU import violation(s)")
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "survey":
        return _cmd_survey(raw_argv[1:])

    parser = argparse.ArgumentParser(
        prog="fastrsi",
        allow_abbrev=False,
        description="Unified engine-agnostic CLI for FastRSI live labs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list discovered live labs")
    p_list.add_argument("--repo-root", default=None)
    p_list.add_argument("--json", action="store_true")

    p_conf = sub.add_parser(
        "conformance", allow_abbrev=False, help="run c0-c10 conformance on a lab",
        epilog="For real labs with launch options, append [--] <engine-options> [-- <workload-args>].",
    )
    p_conf.add_argument("target", help="discovered lab name (e.g. toy) or module:Class")
    mode_grp = p_conf.add_mutually_exclusive_group()
    mode_grp.add_argument("--toy", action="store_true", help="select the CPU toy conformance class (default)")
    mode_grp.add_argument("--real", action="store_true", help="select the real-engine lab class")
    p_conf.add_argument("--path", default=None, help="extra directory to prepend to sys.path")
    p_conf.add_argument("--repo-root", default=None)
    p_conf.add_argument("--only", default="", help="comma-separated case names (e.g. c0_repeatable,c2_visible)")
    p_conf.add_argument("--receipts", default=None, help="JSONL path for transaction receipts")
    p_conf.add_argument("--json", action="store_true")

    sub.add_parser("survey", help="run static live-editability survey (passes flags through)")

    p_ctl = sub.add_parser("ctl", help="send one JSON control command to a running lab")
    transport = p_ctl.add_mutually_exclusive_group(required=True)
    transport.add_argument("--sock", help="Unix domain socket path")
    transport.add_argument("--file-channel", help="file-channel directory (ctl.json + reply/)")
    transport.add_argument("--mailbox", help="NFS mailbox directory (req-*.json + resp-*.json)")
    p_ctl.add_argument("verb", nargs="?", default=None, help="verb or op name (e.g. status)")
    p_ctl.add_argument("--op-key", default="verb", choices=("verb", "op", "cmd"))
    p_ctl.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE")
    p_ctl.add_argument("--json-arg", action="append", default=[], metavar="KEY=JSON")
    p_ctl.add_argument("--payload", default=None, help="base JSON object payload")
    p_ctl.add_argument("--timeout", type=float, default=60.0)
    p_ctl.add_argument("--keep-receipt", action="store_true")

    p_doc = sub.add_parser("doctor", help="audit environment and discovered labs")
    p_doc.add_argument("--repo-root", default=None)
    p_doc.add_argument("--json", action="store_true")

    if raw_argv[:1] == ["conformance"]:
        # Engine training options beyond the delimiter may reuse CLI option names.
        end = raw_argv.index("--") if "--" in raw_argv else len(raw_argv)
        args, extra = parser.parse_known_args(raw_argv[:end])
        return _cmd_conformance(args, extra + raw_argv[end:])
    args = parser.parse_args(raw_argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "ctl":
        return _cmd_ctl(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
