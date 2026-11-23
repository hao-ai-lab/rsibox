"""Client for a running Megatron lab: one JSON request per call over its Unix socket.

    python -m rsibox_megatron.client --sock PATH status        (or --mailbox DIR)
    python -m rsibox_megatron.client --sock PATH install --patch patch.json
    python -m rsibox_megatron.client --sock PATH check --steps 4
    python -m rsibox_megatron.client --sock PATH uninstall [--id TXN]
    python -m rsibox_megatron.client --sock PATH repl --code 'print(rank)'

A request that gets no answer within `timeout` raises TimeoutError: the lab may be hung in a
step, so the caller must treat it as poisoned and respawn (the lab cannot answer for itself).
"""
import argparse
import json
import sys

from rsibox import ipc


def call(sock, req, timeout=600):
    return ipc.unix_json_call(sock, req, timeout=timeout)


def call_mailbox(root, req, timeout=600, poll_s=0.02):
    """The same request through a mailbox directory (gap.Mailbox), from another host."""
    return ipc.mailbox_call(root, req, timeout=timeout, poll_s=poll_s)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sock")
    p.add_argument("--mailbox")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("verb")
    p.add_argument("--patch")
    p.add_argument("--steps", type=int)
    p.add_argument("--id")
    p.add_argument("--code")
    p.add_argument("--kind", default="state")
    a = p.parse_args(argv)
    req = {"verb": a.verb}
    if a.patch:
        req["patch"] = json.load(open(a.patch))
    if a.steps:
        req["steps"] = a.steps
    if a.id:
        req["id"] = a.id
    if a.code:
        req["code"] = a.code
    if a.verb == "reset":
        req["kind"] = a.kind
    r = call_mailbox(a.mailbox, req, a.timeout) if a.mailbox else call(a.sock, req, a.timeout)
    r.pop("receipt", None)
    json.dump(r, sys.stdout, indent=1, default=str)
    print()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
