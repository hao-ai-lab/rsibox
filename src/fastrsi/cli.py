"""Command line. One JSON object per command; exit 0 on success, 1 on failure, 2 on refusal."""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from fastrsi import __version__, edits
from fastrsi.core import EDITED, POISONED, Session
from fastrsi.engines import load

HOME = ".fastrsi"
CONFIG = os.path.join(HOME, "engine.json")


def emit(record, code=0):
    print(json.dumps(record, indent=1))
    return code


def open_session():
    if not os.path.exists(CONFIG):
        raise SystemExit("no engine here; run: fastrsi start <engine>")
    with open(CONFIG) as handle:
        config = json.load(handle)
    engine = load(config["engine"])(os.path.join(HOME, "engine"), **config["options"])
    return Session(engine, os.path.join(HOME, "session"))


def parse_options(pairs):
    return dict(pair.split("=", 1) for pair in pairs)


def outcome_code(receipt):
    return {"installed": 0, "undone": 0, "refused": 2}.get(receipt["outcome"], 1)


def cmd_start(args):
    os.makedirs(HOME, exist_ok=True)
    engine = load(args.engine)(os.path.join(HOME, "engine"), **parse_options(args.option))
    engine.start()
    with open(CONFIG, "w") as handle:
        json.dump({"engine": args.engine, "options": parse_options(args.option)}, handle)
    return emit({"engine": args.engine, "state": Session(engine, f"{HOME}/session").state})


def cmd_install(args):
    session = open_session()
    if session.health() == POISONED:
        return emit({"state": POISONED, "reason": session.data["reason"]}, 1)
    edit = edits.from_patch(session.engine.root(), args.patch)
    receipt = session.install(edit)
    return emit(receipt, outcome_code(receipt))


def cmd_undo(args):
    receipt = open_session().undo()
    return emit(receipt, outcome_code(receipt))


def cmd_status(args):
    session = open_session()
    state = session.health()
    record = {"state": state, "reason": session.data["reason"], "edit": session.data["edit"],
              "installs": session.data["installs"]}
    if state == EDITED:
        record["witness"] = session.witness()
    return emit(record, 1 if state == POISONED else 0)


def cmd_exercise(args):
    session = open_session()
    result = session.exercise()
    if result is None:
        return emit({"state": POISONED, "reason": session.data["reason"]}, 1)
    return emit(result)


def cmd_restart(args):
    return emit(open_session().restart())


def cmd_stop(args):
    open_session().engine.stop()
    os.unlink(CONFIG)
    return emit({"stopped": True})


def cmd_conformance(args):
    from fastrsi.conformance import Suite

    with tempfile.TemporaryDirectory() as work:
        workdir = args.output or os.path.join(work, "conformance")
        record = Suite(load(args.engine), workdir, **parse_options(args.option)).run()
    return emit(record, 0 if record["passed"] else 1)


def cmd_overhead(args):
    from fastrsi import overhead

    with tempfile.TemporaryDirectory() as work:
        record = overhead.run(load(args.engine), args.output or os.path.join(work, "overhead"),
                              parse_options(args.option), rounds=args.rounds,
                              repeats=args.repeats, cycles=args.cycles)
    return emit(record)


def cmd_demo(args):
    from fastrsi.engines.toy import Toy

    with tempfile.TemporaryDirectory() as work:
        engine = Toy(os.path.join(work, "toy"))
        engine.start()
        try:
            session = Session(engine, os.path.join(work, "session"))
            spec = engine.spec()
            before = session.exercise()["digest"]
            edit = edits.replace(engine.root(), *spec.visible)
            receipt = session.install(edit)
            print(f"installed  {receipt['edit'][:6]}  processes={receipt['processes']}")
            after = session.exercise()["digest"]
            hits = sum(1 for count in session.witness()["toymodel.py:forward"] if count)
            print(f"witness    ran in {hits} of {receipt['processes']} processes")
            print(f"output     {'changed' if after != before else 'unchanged'}")
            session.undo()
            engine.reset()
            same = session.exercise()["digest"] == before
            print(f"undone     base identity proven, output {'restored' if same else 'DIFFERS'}")
            bad = session.install(edits.replace(engine.root(), *spec.refused["signature"]))
            print(f"refused    {bad['reason']}: restart needed")
        finally:
            engine.stop()
    return 0


def cmd_check(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return subprocess.run(command).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fastrsi")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="name", required=True)

    def add(name, function, *arguments):
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        for flags, options in arguments:
            sub.add_argument(*flags, **options)

    option = (("-o", "--option"), {"action": "append", "default": []})
    output = (("--output",), {})
    engine = (("engine",), {})
    add("demo", cmd_demo)
    add("start", cmd_start, engine, option)
    add("install", cmd_install, (("patch",), {}))
    add("undo", cmd_undo)
    add("status", cmd_status)
    add("exercise", cmd_exercise)
    add("restart", cmd_restart)
    add("stop", cmd_stop)
    add("check", cmd_check, (("command",), {"nargs": argparse.REMAINDER}))
    add("conformance", cmd_conformance, engine, option, output)
    add("overhead", cmd_overhead, engine, option, output,
        (("--rounds",), {"type": int, "default": 3}),
        (("--repeats",), {"type": int, "default": 20}),
        (("--cycles",), {"type": int, "default": 20}))
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
