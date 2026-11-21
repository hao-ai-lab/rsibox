"""The step gap: how the host on rank 0 reaches every rank, and how clients reach the host.

Megatron runs one process per GPU, all in the same train loop. The lab enters the gap before a
train step on every rank. There, rank 0 hosts the kernel's coordinator and the other ranks wait
for its commands:

    round(msg)  rank 0 sends msg to every rank on a gloo group; every rank (rank 0 too) handles
                it; the replies come back to rank 0, in rank order.

Gloo, not NCCL: the rounds carry Python objects, run on the CPU, and never touch the training
streams. A client that times out or disconnects before its reply costs only that reply: reply()
returns False and the host serves the next request (the ranks never see it). The ranks wait for
the next message, and send their replies, on a group with a long timeout, since they wait there
while the lab is idle or while rank 0 is still busy. Rank 0 never blocks in a collective: it
waits for its sends and receives with a deadline, at most FASTRSI_ROUND_TIMEOUT_S after its own
part, so a rank that dies or hangs in a round fails it on rank 0 (RoundFailed), which poisons
the lab (see Ranks for what is and is not bounded). A client's own timeout (see client.py) may be
shorter and poison it on the client's side first.

Clients reach rank 0 through a Unix socket (one JSON line in, one JSON line out; the path must
be under 108 bytes) or, from another host, a mailbox directory on a shared file system:
`req-<id>.json` in, `resp-<id>.json` out, each written to a temporary name and renamed. Our
shared NFS showed a file written on a login host to the node in 26-34 ms (2026-11-07 probe).
"""
import ctypes
import os
import pickle
import time
from datetime import timedelta

from fastrsi import ipc


class RoundFailed(RuntimeError):
    """A round did not complete on every rank (a rank died, a send or receive failed, or a rank
    did not answer in time). The group is not used again: every later round raises this at once."""


BYTES_PER_READ = 1 << 30
TAG_MSG_SIZE, TAG_MSG, TAG_REPLY_SIZE, TAG_REPLY = 1, 2, 3, 4
# (work, tensor) of every send and receive rank 0 posted in a round that failed: kept at module
# level so both live as long as the process even if Ranks is deleted, since gloo may still write
# a late reply into the tensor.
_KEPT_BUFFERS = []


def _encode(obj):
    """obj -> (int64 size tensor, uint8 payload tensor) of its pickle."""
    import torch
    data = pickle.dumps(obj)
    return (torch.tensor([len(data)], dtype=torch.int64),
            torch.frombuffer(bytearray(data), dtype=torch.uint8))


def _decode(payload):
    """The object in a uint8 payload tensor (read through ctypes, in bounded reads: no numpy)."""
    address, length = payload.data_ptr(), payload.numel()
    data = b"".join(ctypes.string_at(address + offset, min(BYTES_PER_READ, length - offset))
                    for offset in range(0, length, BYTES_PER_READ))
    return pickle.loads(data)


class Ranks:
    """The gloo group and the round, as point-to-point messages between rank 0 and each rank.

    The other ranks block in recv() and send() on a group with a long timeout: they wait there
    while the lab is idle, and a rank that answers before a slower rank 0 just waits. Rank 0 never
    blocks in a collective: each of its sends and receives is waited for with a deadline. It
    waits at most FASTRSI_ROUND_TIMEOUT_S (default 900 s) for its sends, and at most that long
    after its own part for the replies; a rank that has not answered by then (dead, hung, or
    slower than rank 0 by more than the timeout), or a send or receive that fails, fails the
    round (RoundFailed). Not bounded: rank 0's own part of the round (its own train steps)."""

    def __init__(self, timeout_s=7 * 24 * 3600, round_timeout_s=None):
        import torch
        import torch.distributed as dist
        if round_timeout_s is None:
            round_timeout_s = float(os.environ.get("FASTRSI_ROUND_TIMEOUT_S", "900"))
        self.torch, self.dist = torch, dist
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))
        self.round_timeout_s = round_timeout_s
        self.failed = None

    # -- the other ranks ---------------------------------------------------------------
    def _recv(self, tag_size, tag_payload, source=0):
        size = self.torch.zeros(1, dtype=self.torch.int64)
        self.dist.recv(size, src=source, group=self.group, tag=tag_size)
        payload = self.torch.empty(int(size.item()), dtype=self.torch.uint8)
        self.dist.recv(payload, src=source, group=self.group, tag=tag_payload)
        return _decode(payload)

    def bcast(self, msg=None):
        """Ranks other than 0: wait for rank 0's next message. After a round rank 0 gave up on,
        this or gather() may block until the job is respawned, or raise (transport-dependent);
        the lab's serve loop then exits the rank non-zero (hook.LOST_ROUND_EXIT)."""
        return self._recv(TAG_MSG_SIZE, TAG_MSG)

    def gather(self, reply):
        """Ranks other than 0: send this round's reply to rank 0."""
        size, payload = _encode(reply)
        self.dist.send(size, dst=0, group=self.group, tag=TAG_REPLY_SIZE)
        self.dist.send(payload, dst=0, group=self.group, tag=TAG_REPLY)

    # -- rank 0 --------------------------------------------------------------------------
    def _wait(self, work, deadline, what):
        """Wait for one send or receive until the deadline (gloo's wait takes a timeout)."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise RoundFailed(f"{what}: not done within {self.round_timeout_s} s")
        try:
            work.wait(timeout=timedelta(seconds=left))
        except Exception as error:              # gloo: timeout, peer gone
            raise RoundFailed(f"{what}: {type(error).__name__}: {str(error)[:300]}") from error

    def round(self, msg, handle):
        """Rank 0 only: every rank handles msg; returns the replies in rank order. Raises
        RoundFailed when a rank does not answer in time or a send or receive fails (and for
        every round after that one)."""
        if self.failed:
            raise RoundFailed(f"an earlier round failed: {self.failed}")
        posted = []                             # (work, tensor) of this round

        def post(work, tensor):
            posted.append((work, tensor))
            return work
        try:
            others = range(1, self.world)
            size, payload = _encode(msg)
            deadline = time.monotonic() + self.round_timeout_s
            for tag, tensor in ((TAG_MSG_SIZE, size), (TAG_MSG, payload)):
                sends = [post(self.dist.isend(tensor, dst=rank, group=self.group, tag=tag), tensor)
                         for rank in others]
                for rank, work in zip(others, sends):
                    self._wait(work, deadline, f"send to rank {rank}")
            sizes = {}
            for rank in others:                 # posted before rank 0's own part
                reply_size = self.torch.zeros(1, dtype=self.torch.int64)
                sizes[rank] = (reply_size, post(self.dist.irecv(
                    reply_size, src=rank, group=self.group, tag=TAG_REPLY_SIZE), reply_size))
            own = handle(msg)
            deadline = time.monotonic() + self.round_timeout_s
            replies = [own]
            for rank in others:
                reply_size, work = sizes[rank]
                self._wait(work, deadline, f"reply of rank {rank}")
                reply = self.torch.empty(int(reply_size.item()), dtype=self.torch.uint8)
                self._wait(post(self.dist.irecv(reply, src=rank, group=self.group,
                                                tag=TAG_REPLY), reply),
                           deadline, f"reply of rank {rank}")
                replies.append(_decode(reply))
            return replies
        except Exception as error:              # any failure of a send or receive poisons
            _KEPT_BUFFERS.extend(posted)        # never freed: a late reply may still land
            self.failed = f"{msg.get('op')}: " + (
                str(error) if isinstance(error, RoundFailed)
                else f"{type(error).__name__}: {str(error)[:300]}")
            raise RoundFailed(self.failed) from error


class Server(ipc.UnixJsonServer):
    """Rank 0: a thread accepts clients and queues (request, reply function) for the gap loop,
    which runs every request in the main thread, between two train steps."""

    def __init__(self, path):
        super().__init__(path, handler=None, backlog=8, mode=None, read_timeout_s=None)


class Mailbox(ipc.MailboxServer):
    """Rank 0: a thread polls `root` for requests and queues them on `q` like Server."""

    def __init__(self, root, q, poll_s=0.02):
        super().__init__(root, q, poll_s=poll_s)
