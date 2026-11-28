"""`rsibox` / `python3 -m rsibox`: serve a box, launch one, and call it.

    rsibox serve --lab SPEC|NAME [--real] --state DIR      the daemon (what launchers start)
    rsibox up --launcher local|docker|gvisor|slurm ... [-- ENGINE_ARGS...]  start a daemon, wait for its box file
    rsibox down --state DIR
    rsibox wait | status | info | events | receipts | source MODULE
    rsibox install (--file MODULE=PATH ... | --patch FILE | --diff FILE) [--accept CHECK ...]
    rsibox check [--workload JSON] | reset [--kind state|respawn] | uninstall [--id TXN]
    rsibox fresh-check [--file MODULE=PATH ... | --patch FILE | --diff FILE] [--workload JSON] [--timeout SEC]
    rsibox verify-trace PATH [PATH ...] [--expected-count N --expected-hash SHA] [--json]

Client commands read `RSIBOX` (a box file) or `RSIBOX_URL` + `RSIBOX_TOKEN`; `--box FILE`
overrides them. Exit code: 0 when the reply is ok, 2 for a refused or aborted install, else 1.
"""
import argparse
import json
import os
import re
import signal
import sys
import time

from . import box as B
from . import client as C
from . import launch as L
from . import server as S


def _resolve(lab, real):
    """`module:Class` as given, or a name `rsibox list` knows. Returns the spec."""
    if ":" in lab:
        return lab
    from . import lab_cli as fcli
    spec, extra = fcli.resolve_lab(lab, real=real)
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    return spec


def _serve(a):
    try:
        spec = _resolve(a.lab, a.real)
        lab_cls = B.load_lab(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"rsibox serve: cannot load lab {a.lab!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    state = os.path.abspath(a.state)
    os.makedirs(state, exist_ok=True)
    box = B.Box(lambda receipts: lab_cls(receipts=receipts), spec=spec, state_dir=state,
                busy_wait_s=a.busy_wait)
    daemon = S.Daemon(box, bind=a.bind, port=a.port, token=os.environ.get("RSIBOX_TOKEN"), state_dir=state)

    def on_signal(*_):
        box.close()
        daemon.stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    daemon.start()
    print(f"rsibox: listening on {daemon.url}, box file {os.path.join(state, S.BOX_FILE)}", flush=True)
    box.start()
    daemon.wait()
    box.close()
    return 0


def _pairs(items, what):
    out = {}
    for item in items or ():
        key, sep, val = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"rsibox: {what} expects NAME=VALUE, got {item!r}")
        out[key] = val
    return out


def _up(a, lab_args=()):
    spec = {"launcher": a.launcher, "lab": a.lab, "real": a.real, "state": a.state, "python": a.python,
            "repo": a.repo, "env": _pairs(a.env, "--env"), "bind": a.bind, "port": a.port,
            "image": a.image, "gpus": a.gpus, "runtime": a.runtime, "mounts": a.mount,
            "docker_args": a.docker_arg, "nested_runsc": a.nested_runsc, "runsc": a.runsc,
            "allow_unsupported_driver": a.allow_unsupported_driver, "sbatch_args": a.sbatch_arg,
            "setup": a.setup, "lab_args": list(lab_args)}
    if a.launcher == "docker" and not a.image:
        raise SystemExit("rsibox up: --launcher docker needs --image")
    if a.dry_run:
        what = L.plan(spec)
        print(what["script"] if what["script"] else json.dumps(what, indent=1))
        return 0
    record = L.up(spec, wait_s=a.wait)
    print(json.dumps(record, indent=1))
    return 0 if record["ok"] else 1


def _client(a):
    try:
        return C.Client.from_box_file(a.box) if a.box else C.Client.from_env()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"rsibox: {exc}")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _call(a):
    box, cmd = _client(a), a.command
    if cmd == "wait":
        reply = box.wait(timeout=a.timeout)
    elif cmd == "install":
        if a.patch:
            patch = json.loads(_read(a.patch))
        elif a.diff:
            patch = {"diff": _read(a.diff)}
        elif a.file:
            patch = {"files": {mod: _read(path) for mod, path in _pairs(a.file, "--file").items()}}
        else:
            raise SystemExit("rsibox install: give --file MODULE=PATH, --patch FILE or --diff FILE")
        if a.accept:
            patch["accept"] = a.accept
        patch.setdefault("edit_time", time.time())
        reply = box.install(patch)
    elif cmd == "check":
        reply = box.check(json.loads(a.workload) if a.workload else None)
    elif cmd in ("fresh-check", "fresh_check"):
        if sum(bool(x) for x in (a.patch, a.diff, a.file)) > 1:
            raise SystemExit("rsibox fresh-check: give at most one of --file, --patch, or --diff")
        try:
            if a.patch:
                patch = json.loads(_read(a.patch))
            elif a.diff:
                patch = {"diff": _read(a.diff)}
            elif a.file:
                patch = {"files": {mod: _read(path) for mod, path in _pairs(a.file, "--file").items()}}
            else:
                patch = None
            workload = json.loads(a.workload) if a.workload else None
        except (OSError, ValueError) as exc:
            raise SystemExit(f"rsibox fresh-check: {exc}")
        reply = box.fresh_check(patch, workload=workload, timeout=a.timeout)
    elif cmd == "reset":
        reply = box.reset(a.kind)
    elif cmd == "uninstall":
        reply = box.uninstall(a.id)
    elif cmd == "events":
        reply = box.events(a.since)
    elif cmd == "source":
        reply = box.source(a.module)
        if reply.get("ok") and not a.json:
            sys.stdout.write(reply["text"])
            return 0
    else:
        reply = getattr(box, cmd)()
    if not getattr(a, "keep_receipt", True):
        reply = {k: v for k, v in reply.items() if k != "receipt"}
    print(json.dumps(reply, indent=1, default=str))
    if reply.get("outcome") in ("refused", "aborted"):
        return 2
    return 0 if reply.get("ok") else 1


def _verify_trace(a, ap):
    has_count = a.expected_count is not None
    has_hash = a.expected_hash is not None
    if has_count != has_hash:
        ap.error("--expected-count and --expected-hash must be given together")
    expected_head = None
    if has_count:
        if len(a.paths) != 1:
            ap.error("--expected-count/--expected-hash require a single trace path")
        if a.expected_count < 0:
            ap.error("--expected-count must be >= 0")
        expected_hash = a.expected_hash.strip()
        if not expected_hash:
            ap.error("--expected-hash must be non-empty")
        if expected_hash != "genesis":
            if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                ap.error("--expected-hash must be 'genesis' or a 64-character hex SHA-256 digest")
            expected_hash = expected_hash.lower()
        if a.expected_count == 0 and expected_hash != "genesis":
            ap.error("--expected-count 0 requires --expected-hash genesis")
        if a.expected_count > 0 and expected_hash == "genesis":
            ap.error("--expected-hash genesis requires --expected-count 0")
        expected_head = (a.expected_count, expected_hash)

    from .pool import verify_trace

    results = []
    for path in a.paths:
        res = verify_trace(path, expected_head=expected_head)
        head = list(res["head"]) if "head" in res else None
        results.append({
            "path": path,
            "ok": bool(res["ok"]),
            "count": int(res.get("count", 0)),
            "head": head,
            "line": res.get("line"),
            "error": res.get("error"),
        })
    passed = sum(1 for item in results if item["ok"])
    failed = len(results) - passed
    total_records = sum(item["count"] for item in results)
    summary = {
        "ok": failed == 0,
        "checked": len(results),
        "passed": passed,
        "failed": failed,
        "total_records": total_records,
        "results": results,
    }
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        for item in results:
            if item["ok"]:
                head = item["head"] or [0, "genesis"]
                digest = head[1] if head[1] == "genesis" else f"{head[1][:12]}..."
                print(f"OK   {item['path']} ({item['count']} records, head={head[0]}:{digest})")
            else:
                loc = f"line {item['line']}: " if item["line"] is not None else ""
                print(f"FAIL {item['path']} ({item['count']} verified; {loc}{item['error']})")
        if len(results) > 1:
            print(f"Summary: {passed}/{len(results)} traces OK ({total_records} total records)")
    return 0 if summary["ok"] else 1


_LAB_COMMANDS = frozenset(("list", "conformance", "survey", "ctl", "doctor"))


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _LAB_COMMANDS:
        from . import lab_cli
        return lab_cli.main(raw)
    if raw and raw[0] == "promote":
        from . import promote
        return promote.main(raw[1:])
    up_lab_args = []
    parse_argv = raw
    if raw and raw[0] == "up" and "--" in raw:
        sep = raw.index("--")
        parse_argv = raw[:sep]
        up_lab_args = raw[sep + 1:]
    ap = argparse.ArgumentParser(
        prog="rsibox",
        description="A live lab as a service.",
        epilog="Lab commands: list, conformance, survey, ctl, doctor, promote (run 'rsibox <cmd> --help').",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def lab_args(p):
        p.add_argument("--lab", required=True, help="module:Class, or a name `rsibox list` knows")
        p.add_argument("--real", action="store_true", help="for a name: the real-engine lab, not the toy")
        p.add_argument("--state", required=True, help="folder for the box file, events and receipts")
        p.add_argument("--bind", default=None)
        p.add_argument("--port", type=int, default=0)

    p = sub.add_parser("serve", help="run the box daemon in the foreground")
    lab_args(p)
    p.set_defaults(bind="127.0.0.1")
    p.add_argument("--busy-wait", type=float, default=5.0)

    p = sub.add_parser(
        "up",
        help="start a daemon with a launcher and wait for its box file",
        epilog="Pass engine/lab CLI flags after '--', e.g.: rsibox up ... -- Qwen/Qwen3-8B -tp 2 --max-model-len 8192",
    )
    lab_args(p)
    p.add_argument("--launcher", required=True, choices=("local", "docker", "gvisor", "slurm"))
    p.add_argument("--python", default=None)
    p.add_argument("--repo", default=None, help="the rsibox checkout (default: this one)")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the box file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--image")
    p.add_argument("--gpus")
    p.add_argument("--runtime", help="Docker runtime, e.g. runsc")
    p.add_argument("--mount", action="append", default=[], metavar="SRC:DST[:ro]")
    p.add_argument("--docker-arg", action="append", default=[])
    p.add_argument("--nested-runsc", metavar="PATH", help="docker: run the daemon under this runsc inside a privileged container")
    p.add_argument("--runsc", default=None, help="gvisor: the runsc binary")
    p.add_argument("--allow-unsupported-driver", action="store_true")
    p.add_argument("--sbatch-arg", action="append", default=[])
    p.add_argument("--setup", default=None, help="slurm: shell line run before the daemon")

    p = sub.add_parser("down", help="close a box and remove what `up` started")
    p.add_argument("--state", required=True)

    def client_cmd(name, help_text, aliases=()):
        q = sub.add_parser(name, help=help_text, aliases=aliases)
        q.add_argument("--box", default=None, help="box file (default: $RSIBOX)")
        return q

    client_cmd("wait", "block until the engine is warm").add_argument("--timeout", type=float, default=1800.0)
    for name in ("status", "info", "receipts"):
        client_cmd(name, f"GET /v1/{name}")
    client_cmd("events", "calls so far").add_argument("--since", type=int, default=0)
    q = client_cmd("source", "print the live text of a module")
    q.add_argument("module")
    q.add_argument("--json", action="store_true")
    q = client_cmd("install", "install a patch")
    q.add_argument("--file", action="append", default=[], metavar="MODULE=PATH")
    q.add_argument("--patch")
    q.add_argument("--diff")
    q.add_argument("--accept", action="append", default=[])
    q.add_argument("--keep-receipt", action="store_true")
    q = client_cmd("check", "run the workload in the warm engine")
    q.add_argument("--workload", help="JSON")
    q = client_cmd(
        "fresh-check",
        "run the workload in a fresh subprocess from base plus optional patch",
        aliases=("fresh_check",),
    )
    q.add_argument("--file", action="append", default=[], metavar="MODULE=PATH")
    q.add_argument("--patch")
    q.add_argument("--diff")
    q.add_argument("--workload", help="JSON")
    q.add_argument("--timeout", type=float, default=None, help="HTTP request timeout in seconds")
    client_cmd("reset", "reset volatile state").add_argument("--kind", default="state")
    client_cmd("uninstall", "undo the newest edit, or down to --id").add_argument("--id")

    vt = sub.add_parser("verify-trace", help="verify SHA-256 hash-chained SandboxPool JSONL trace file(s)")
    vt.add_argument("paths", nargs="+", metavar="PATH", help="trace JSONL file path(s)")
    vt.add_argument("--expected-count", type=int, default=None, help="expected record count (requires --expected-hash)")
    vt.add_argument("--expected-hash", default=None, help="expected tail SHA-256 hash (requires --expected-count)")
    vt.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")

    a = ap.parse_args(parse_argv)
    if a.command == "serve":
        return _serve(a)
    if a.command == "up":
        return _up(a, up_lab_args)
    if a.command == "down":
        reply = L.down(a.state)
        print(json.dumps(reply, indent=1, default=str))
        return 0 if reply["ok"] else 1
    if a.command == "verify-trace":
        return _verify_trace(a, vt)
    return _call(a)


if __name__ == "__main__":
    sys.exit(main())
