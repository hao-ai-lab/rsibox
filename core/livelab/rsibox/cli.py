"""`rsibox` / `python3 -m rsibox`: serve a box, launch one, and call it.

    rsibox serve --lab SPEC|NAME [--real] --state DIR      the daemon (what launchers start)
    rsibox up --launcher local|docker|gvisor|slurm ...     start a daemon, wait for its box file
    rsibox down --state DIR
    rsibox wait | status | info | events | receipts | source MODULE
    rsibox install (--file MODULE=PATH ... | --patch FILE | --diff FILE) [--accept CHECK ...]
    rsibox check [--workload JSON] | reset [--kind state|respawn] | uninstall [--id TXN]

Client commands read `RSIBOX` (a box file) or `RSIBOX_URL` + `RSIBOX_TOKEN`; `--box FILE`
overrides them. Exit code: 0 when the reply is ok, 2 for a refused or aborted install, else 1.
"""
import argparse
import json
import os
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


def _up(a):
    spec = {"launcher": a.launcher, "lab": a.lab, "real": a.real, "state": a.state, "python": a.python,
            "repo": a.repo, "env": _pairs(a.env, "--env"), "bind": a.bind, "port": a.port,
            "image": a.image, "gpus": a.gpus, "runtime": a.runtime, "mounts": a.mount,
            "docker_args": a.docker_arg, "nested_runsc": a.nested_runsc, "runsc": a.runsc,
            "allow_unsupported_driver": a.allow_unsupported_driver, "sbatch_args": a.sbatch_arg,
            "setup": a.setup}
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


_LAB_COMMANDS = frozenset(("list", "conformance", "survey", "ctl", "doctor"))


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _LAB_COMMANDS:
        from . import lab_cli
        return lab_cli.main(raw)
    ap = argparse.ArgumentParser(
        prog="rsibox",
        description="A live lab as a service.",
        epilog="Lab commands: list, conformance, survey, ctl, doctor (run 'rsibox <cmd> --help').",
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

    p = sub.add_parser("up", help="start a daemon with a launcher and wait for its box file")
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

    def client_cmd(name, help_text):
        q = sub.add_parser(name, help=help_text)
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
    client_cmd("reset", "reset volatile state").add_argument("--kind", default="state")
    client_cmd("uninstall", "undo the newest edit, or down to --id").add_argument("--id")

    a = ap.parse_args(argv)
    if a.command == "serve":
        return _serve(a)
    if a.command == "up":
        return _up(a)
    if a.command == "down":
        reply = L.down(a.state)
        print(json.dumps(reply, indent=1, default=str))
        return 0 if reply["ok"] else 1
    return _call(a)


if __name__ == "__main__":
    sys.exit(main())
