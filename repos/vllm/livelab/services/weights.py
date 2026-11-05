"""Weight service: keep the model's weights in GPU memory across engine restarts.

Runs `vllm preload`, which starts one background process per GPU holding that GPU's share of the
(already quantized) weights. Engines started with `--load-format ipc_cache` attach to that GPU
memory directly instead of reading the checkpoint from disk. Start it once; it outlives every
engine.

Usage:
  python3 weights.py start --model M --tp 4 --max-model-len L --log preload.log [-- extra args]
  python3 weights.py wait --log preload.log
"""
import argparse, os, re, subprocess, sys, time

READY = re.compile(r"(weight cache daemon|daemons?) .*(ready|serving)", re.I)
FAILED = re.compile(r"(?:Traceback|Error:).*")


def start(model, tp, max_model_len, log, extra=(), vllm="vllm", env=None):
    cmd = [vllm, "preload", "--model", model, "--tensor-parallel-size", str(tp),
           "--max-model-len", str(max_model_len), *extra]
    out = open(log, "w")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env or os.environ.copy(),
                         start_new_session=True)
    return p.pid, cmd


def wait(log, timeout=3600, poll=5, pid=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pid is not None and not os.path.exists(f"/proc/{pid}"):
            return {"ready": False, "error": f"preload process {pid} exited"}
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        if READY.search(text):
            return {"ready": True, "seconds": round(time.time() - t0, 1)}
        if FAILED.search(text):
            return {"ready": False, "error": FAILED.search(text).group(0)}
        time.sleep(poll)
    return {"ready": False, "error": "timeout"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["start", "wait"])
    p.add_argument("--model"); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int); p.add_argument("--log", required=True)
    a, extra = p.parse_known_args()
    if a.cmd == "start":
        print(start(a.model, a.tp, a.max_model_len, a.log, [x for x in extra if x != "--"]))
    else:
        print(wait(a.log))
