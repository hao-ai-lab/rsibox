"""Send one command to the engine-core host over its file channel and return the reply.

    python3 -m adapter.client <dir> '{"op": "status", "txn": "..."}'

The reply is the kernel's result (with its receipt record) plus a `host` block: seconds the op
took in the engine core, steps held for pause keep, and step durations during the op.
"""
import json
import os
import sys

if not __package__:
    _livelab = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
    if _livelab not in sys.path:
        sys.path.insert(0, _livelab)

from fastrsi import ipc  # noqa: E402


def send(root, cmd, timeout=300.0):
    return ipc.file_channel_call(root, cmd, timeout=timeout, poll_s=0.005)


if __name__ == "__main__":
    print(json.dumps(send(sys.argv[1], json.loads(sys.argv[2])), indent=1))
