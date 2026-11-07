"""The step gap: how the host on rank 0 reaches every rank, and how clients reach the host.

Megatron runs one process per GPU, all in the same train loop. The lab enters the gap before a
train step on every rank. There, rank 0 hosts the kernel's coordinator and the other ranks wait
for its commands:

    round(msg)  rank 0 broadcasts msg on a gloo group; every rank (rank 0 too) handles it; the
                replies are gathered on rank 0, in rank order.

Gloo, not NCCL: the rounds carry Python objects, run on the CPU, and never touch the training
streams. The group's timeout is long because ranks wait in it while the lab is idle; a hang is
caught by the client's own timeout (see client.py), which poisons the lab.

Clients reach rank 0 through a Unix socket (one JSON line in, one JSON line out; the path must
be under 108 bytes) or, from another host, a mailbox directory on a shared file system:
`req-<id>.json` in, `resp-<id>.json` out, each written to a temporary name and renamed. Our
shared NFS showed a file written on a login host to the node in 26-34 ms (2026-11-07 probe).
"""
import json
import os
import queue
import socket
import threading
import time
from datetime import timedelta


class Ranks:
    """The gloo group and the round."""

    def __init__(self, timeout_s=7 * 24 * 3600):
        import torch.distributed as dist
        self.dist = dist
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))

    def bcast(self, msg=None):
        box = [msg]
        self.dist.broadcast_object_list(box, src=0, group=self.group)
        return box[0]

    def gather(self, reply):
        out = [None] * self.world if self.rank == 0 else None
        self.dist.gather_object(reply, out, dst=0, group=self.group)
        return out

    def round(self, msg, handle):
        """Rank 0 only: every rank handles msg; returns the replies in rank order."""
        self.bcast(msg)
        return self.gather(handle(msg))


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
            conn, _ = self.sock.accept()
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        f = conn.makefile("rwb")
        line = f.readline()
        if not line:
            conn.close()
            return
        done = threading.Event()

        def reply(obj):
            try:
                f.write((json.dumps(obj, default=str) + "\n").encode())
                f.flush()
            finally:
                conn.close()
                done.set()
        try:
            req = json.loads(line)
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
            with open(tmp, "w") as f:
                json.dump(obj, f, default=str)
            os.rename(tmp, os.path.join(self.root, f"resp-{rid}.json"))
        return reply
