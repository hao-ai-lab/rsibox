"""Send one command to the engine-core host over its file channel and return the reply.

    python3 -m kernel_vllm.client <dir> '{"op": "status", "txn": "..."}'

The reply is the kernel's result (with its receipt record) plus a `host` block: seconds the op
took in the engine core, steps held for pause keep, and step durations during the op.
"""
import json
import os
import sys
import time


def send(root, cmd, timeout=300.0):
    ctl = os.path.join(root, "ctl.json")
    try:
        with open(ctl) as f:
            seq = json.load(f)["seq"] + 1
    except (OSError, ValueError, KeyError):
        seq = 1
    cmd = {**cmd, "seq": seq}
    t = time.time()
    with open(ctl + ".tmp", "w") as f:
        json.dump(cmd, f)
    os.replace(ctl + ".tmp", ctl)
    rp = os.path.join(root, "reply", f"{seq}.json")
    while not os.path.exists(rp):
        if time.time() - t > timeout:
            return {"ok": False, "error": f"no reply in {timeout} s", "seq": seq}
        time.sleep(0.005)
    with open(rp) as f:
        r = json.load(f)
    r["client_round_trip_s"] = round(time.time() - t, 4)
    return r


if __name__ == "__main__":
    print(json.dumps(send(sys.argv[1], json.loads(sys.argv[2])), indent=1))
