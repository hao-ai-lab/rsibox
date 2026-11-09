"""Client for a running Megatron lab: one JSON request per call over its Unix socket.

    python -m fastrsi_megatron.client --sock PATH status        (or --mailbox DIR)
    python -m fastrsi_megatron.client --sock PATH install --patch patch.json
    python -m fastrsi_megatron.client --sock PATH check --steps 4
    python -m fastrsi_megatron.client --sock PATH uninstall [--id TXN]
    python -m fastrsi_megatron.client --sock PATH repl --code 'print(rank)'

A request that gets no answer within `timeout` raises TimeoutError: the lab may be hung in a
step, so the caller must treat it as poisoned and respawn (the lab cannot answer for itself).
"""
import argparse
import json
import socket
import sys


def call(sock, req, timeout=600):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock)
        s.sendall((json.dumps(req) + "\n").encode())
        f = s.makefile("rb")
        line = f.readline()
    except socket.timeout as e:
        raise TimeoutError(f"no answer to {req.get('verb')} in {timeout} s") from e
    finally:
        s.close()
    if not line:
        raise ConnectionError(f"lab closed the connection during {req.get('verb')}")
    return json.loads(line)


def call_mailbox(root, req, timeout=600, poll_s=0.02):
    """The same request through a mailbox directory (gap.Mailbox), from another host."""
    import os
    import time
    import uuid
    rid = uuid.uuid4().hex[:12]
    tmp, path = os.path.join(root, f".req-{rid}.tmp"), os.path.join(root, f"req-{rid}.json")
    with open(tmp, "w") as f:
        json.dump(req, f)
    os.rename(tmp, path)
    resp, t0 = os.path.join(root, f"resp-{rid}.json"), time.time()
    while time.time() - t0 < timeout:
        os.listdir(root)                      # revalidates the directory on NFS
        if os.path.exists(resp):
            with open(resp) as f:
                r = json.load(f)
            os.unlink(resp)
            return r
        time.sleep(poll_s)
    raise TimeoutError(f"no answer to {req.get('verb')} in {timeout} s")


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
