"""Unified, engine-agnostic CLI for FastRSI (`fastrsi` / `python3 -m fastrsi`).

Commands:
- `fastrsi list [--repo-root PATH] [--json]`
    Dynamically discovers the built-in `toy` lab, `template`, and every engine under
    `<repo_root>/repos/<engine>/livelab/` without hardcoding engine names or importing them.
- `fastrsi conformance <target> [--real] [--only ...] [--receipts ...] [--json]`
    Runs `core/livelab/conformance` against a discovered engine name (toy by default,
    `--real` for the real-engine conformance class) or any explicit `module:Class` spec.
- `fastrsi survey ...`
    Delegates to `core/livelab/survey`.
- `fastrsi ctl (--sock PATH | --file-channel DIR | --mailbox DIR) <verb> [...]`
    Transport-agnostic JSON control client for any running live lab.
- `fastrsi doctor [--repo-root PATH] [--json]`
    Checks Python version, shared runtime imports, `forksafe` environment audit,
    `AF_UNIX` socket path limits, and discovered labs.
"""
import argparse
import ast
import json
import os
import subprocess
import sys

import forksafe

from . import ipc


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


def _discover_livelab_dir(name: str, livelab_dir: str) -> dict:
    """Inspect one `<repo>/repos/<name>/livelab` folder via optional `livelab.json` or AST."""
    manifest_path = os.path.join(livelab_dir, "livelab.json")
    manifest = ipc.read_json(manifest_path) if os.path.isfile(manifest_path) else None
    if isinstance(manifest, dict):
        return {
            "name": name,
            "livelab_dir": livelab_dir,
            "toy_spec": manifest.get("toy"),
            "real_spec": manifest.get("real"),
        }

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
        return target, None
    by_name = {entry["name"]: entry for entry in discover_labs(repo_root)}
    entry = by_name.get(target)
    if entry is None:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"unknown lab {target!r} (known: {known}; or pass module:Class)")
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
    prev_py = os.environ.get("PYTHONPATH")
    if extra_path:
        livelab_core = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        parts = [p for p in (prev_py.split(os.pathsep) if prev_py else []) if p]
        for needed in (livelab_core, extra_path):
            if needed not in parts:
                parts.insert(0, needed)
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    try:
        from conformance import __main__ as conf_main

        conf_argv = ["--lab", spec]
        if args.only:
            conf_argv += ["--only", args.only]
        if args.receipts:
            conf_argv += ["--receipts", args.receipts]
        if args.json:
            conf_argv.append("--json")
        if engine_args:
            lab = conf_main.load(spec)
            configure = getattr(lab, "launch_environment", None)
            if not args.real or configure is None:
                print(f"fastrsi conformance: {spec} does not accept launch arguments", file=sys.stderr)
                return 2
            env = {**os.environ, **configure(engine_args)}
            return subprocess.run([sys.executable, "-m", "conformance", *conf_argv], env=env).returncode
        return conf_main.main(conf_argv)
    finally:
        if extra_path:
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
    req = json.loads(args.payload) if args.payload else {}
    if not isinstance(req, dict):
        print("fastrsi ctl: --payload must be a JSON object", file=sys.stderr)
        return 2
    if args.verb:
        req[args.op_key] = args.verb
    for pair in args.arg:
        key, sep, val = pair.partition("=")
        if not sep:
            print(f"fastrsi ctl: --arg expects KEY=VALUE, got {pair!r}", file=sys.stderr)
            return 2
        req[key] = val
    for pair in args.json_arg:
        key, sep, val = pair.partition("=")
        if not sep:
            print(f"fastrsi ctl: --json-arg expects KEY=JSON, got {pair!r}", file=sys.stderr)
            return 2
        req[key] = json.loads(val)
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


def _cmd_doctor(args) -> int:
    labs = discover_labs(args.repo_root)
    env_findings = forksafe.audit_system_environment()
    report = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "livelab_root": os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "repo_root": find_repo_root(args.repo_root),
        "max_unix_sock_bytes": ipc.MAX_SOCK_PATH,
        "discovered_labs": [x["name"] for x in labs],
        "env_findings": env_findings,
    }
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(f"python:        {report['python']} ({report['platform']})")
        print(f"livelab_root:  {report['livelab_root']}")
        print(f"repo_root:     {report['repo_root']}")
        print(f"max_sock_path: {report['max_unix_sock_bytes']} bytes")
        print(f"labs:          {', '.join(report['discovered_labs'])}")
        print(f"env_findings:  {len(env_findings)}")
    return 0


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
        epilog="For real labs with launch options, append the engine options followed by -- and its workload arguments.",
    )
    p_conf.add_argument("target", help="discovered lab name (e.g. toy) or module:Class")
    p_conf.add_argument("--real", action="store_true", help="select the real-engine lab class")
    p_conf.add_argument("--path", default=None, help="extra directory to prepend to sys.path")
    p_conf.add_argument("--repo-root", default=None)
    p_conf.add_argument("--only", default="", help="comma-separated case names")
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
