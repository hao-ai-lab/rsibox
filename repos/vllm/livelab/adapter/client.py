"""Send one command to the engine-core host over its file channel and return the reply.

    python3 -m adapter.client <dir> '{"op": "status", "txn": "..."}'

The reply is the kernel's result (with its receipt record) plus a `host` block: seconds the op
took in the engine core, steps held for pause keep, and step durations during the op.
"""
import json
import os
import sys

if not __package__:
    _livelab = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
    if _livelab not in sys.path:
        sys.path.insert(0, _livelab)

from rsibox import ipc  # noqa: E402


def current_seq(root):
    """Return the highest file-channel sequence number visible under `root`, or 0."""
    root_path = os.fspath(root)
    previous = ipc.read_json(os.path.join(root_path, "ctl.json"))
    previous_seq = (
        int(previous["seq"])
        if isinstance(previous, dict) and isinstance(previous.get("seq"), int)
        else 0
    )
    reply_dir = os.path.join(root_path, "reply")
    existing = [
        int(filename[:-5])
        for filename in (os.listdir(reply_dir) if os.path.isdir(reply_dir) else ())
        if filename.endswith(".json") and filename[:-5].isdigit()
    ]
    return max(previous_seq, max(existing, default=0))


def send(root, cmd, timeout=300.0, *, min_seq=None, alive_fn=None):
    return ipc.file_channel_call(
        root, cmd, timeout=timeout, poll_s=0.005, min_seq=min_seq, alive_fn=alive_fn,
    )


if __name__ == "__main__":
    print(json.dumps(send(sys.argv[1], json.loads(sys.argv[2])), indent=1))
