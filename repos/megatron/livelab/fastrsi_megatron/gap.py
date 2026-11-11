"""The step gap: how the host on rank 0 reaches every rank, and how clients reach the host.

Megatron runs one process per GPU, all in the same train loop. The lab enters the gap before a
train step on every rank. There, rank 0 hosts the kernel's coordinator and the other ranks wait
for its commands:

    round(msg)  rank 0 broadcasts msg on a gloo group; every rank (rank 0 too) handles it; the
                replies are gathered on rank 0, in rank order.

Gloo, not NCCL: the rounds carry Python objects, run on the CPU, and never touch the training
streams. A client that times out or disconnects before its reply costs only that reply: reply()
returns False and the host serves the next request (the ranks never see it). The ranks wait for
the next message on a group with a long timeout, since they wait there while the lab is idle; the
replies come back on a second group whose timeout bounds a round (FASTRSI_ROUND_TIMEOUT_S), so a
rank that dies or hangs in a round fails it on rank 0 (RoundFailed), which poisons the lab. A
client's own timeout (see client.py) is shorter and poisons it on the client's side first.

Clients reach rank 0 through a Unix socket (one JSON line in, one JSON line out; the path must
be under 108 bytes) or, from another host, a mailbox directory on a shared file system:
`req-<id>.json` in, `resp-<id>.json` out, each written to a temporary name and renamed. Our
shared NFS showed a file written on a login host to the node in 26-34 ms (2026-11-07 probe).
"""
import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import timedelta


class RoundFailed(RuntimeError):
    """A round did not complete on every rank (a rank died or did not answer within the round
    timeout). The group is not used again: every later round raises this at once."""


class Ranks:
    """The gloo groups and the round. Ranks wait for rank 0's next message on `group`, whose
    timeout is long because they wait there while the lab is idle; the replies come back on
    `active`, whose timeout (FASTRSI_ROUND_TIMEOUT_S, default 900 s) bounds a round once every
    rank has the message: it must exceed the longest request (a check of FASTRSI_CHECK_STEPS_MAX
    steps). A rank that dies or hangs in a round then fails it on rank 0 instead of blocking."""

    def __init__(self, timeout_s=7 * 24 * 3600, round_timeout_s=None):
        import torch.distributed as dist
        if round_timeout_s is None:
            round_timeout_s = float(os.environ.get("FASTRSI_ROUND_TIMEOUT_S", "900"))
        self.dist = dist
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))
        self.active = dist.new_group(backend="gloo", timeout=timedelta(seconds=round_timeout_s))
        self.round_timeout_s = round_timeout_s
        self.failed = None

    def bcast(self, msg=None):
        box = [msg]
        self.dist.broadcast_object_list(box, src=0, group=self.group)
        return box[0]

    def gather(self, reply):
        out = [None] * self.world if self.rank == 0 else None
        self.dist.gather_object(reply, out, dst=0, group=self.active)
        return out

    def round(self, msg, handle):
        """Rank 0 only: every rank handles msg; returns the replies in rank order. Raises
        RoundFailed when a rank does not answer (and for every round after that one)."""
        if self.failed:
            raise RoundFailed(f"an earlier round failed: {self.failed}")
        try:
            self.bcast(msg)
            return self.gather(handle(msg))
        except RuntimeError as e:               # gloo: timeout, peer gone
            self.failed = f"{msg.get('op')}: {type(e).__name__}: {str(e)[:300]}"
            raise RoundFailed(self.failed) from e


class Server:
    """Rank 0: a thread accepts clients and queues (request, reply function) for the gap loop,
    which runs every request in the main thread, between two train steps."""

    def __init__(self, path):
        self.path = path
        self.q = queue.Queue()
        if os.path.exists(path):
            os.unlink(path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(8)
        threading.Thread(target=self._accept, daemon=True, name="fastrsi-server").start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:                     # closed by close()
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        done = threading.Event()

        def reply(obj):
            """Send obj; False when the client is gone (timed out, disconnected): the lab logs it
            and serves on. Never raises."""
            try:
                f.write((json.dumps(obj, default=str) + "\n").encode())
                f.flush()
                return True
            except (OSError, ValueError) as e:
                print(f"[fastrsi] client gone before the reply ({type(e).__name__}: {e}); "
                      "serving on", file=sys.stderr, flush=True)
                return False
            finally:
                for c in (f, conn):
                    try:
                        c.close()
                    except OSError:
                        pass
                done.set()
        try:
            f = conn.makefile("rwb")
            line = f.readline()
        except OSError:
            conn.close()
            return
        if not line:
            conn.close()
            return
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError(f"a request is a JSON object, not {type(req).__name__}")
        except ValueError as e:
            reply({"ok": False, "error": f"bad request: {e}"})
            return
        self.q.put((req, reply))
        done.wait()

    def next(self, timeout=None):
        return self.q.get(timeout=timeout)

    def close(self):
        try:
            self.sock.close()
            os.unlink(self.path)
        except OSError:
            pass


class Mailbox:
    """Rank 0: a thread polls `root` for requests and queues them on `q` like Server."""

    def __init__(self, root, q, poll_s=0.02):
        self.root, self.q, self.poll_s = root, q, poll_s
        os.makedirs(root, exist_ok=True)
        threading.Thread(target=self._run, daemon=True, name="fastrsi-mailbox").start()

    def _run(self):
        while True:
            try:
                names = sorted(n for n in os.listdir(self.root)
                               if n.startswith("req-") and n.endswith(".json"))
            except OSError:
                names = []
            for n in names:
                rid = n[4:-5]
                src, taken = os.path.join(self.root, n), os.path.join(self.root, n + ".taken")
                try:
                    os.rename(src, taken)
                    with open(taken) as f:
                        req = json.load(f)
                except (OSError, ValueError) as e:
                    req = {"verb": "bad", "error": str(e)}
                self.q.put((req, self._replier(rid)))
            time.sleep(self.poll_s)

    def _replier(self, rid):
        def reply(obj):
            tmp = os.path.join(self.root, f".resp-{rid}.tmp")
            try:
                with open(tmp, "w") as f:
                    json.dump(obj, f, default=str)
                os.rename(tmp, os.path.join(self.root, f"resp-{rid}.json"))
                return True
            except OSError as e:
                print(f"[fastrsi] mailbox reply {rid} not written ({e}); serving on",
                      file=sys.stderr, flush=True)
                return False
        return reply
