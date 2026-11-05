"""Engine service: start `vllm serve` on resident weights, warm caches and the startup hooks,
then report where its startup time went.

What it composes:
  weights   --load-format ipc_cache        (needs services/weights.py running)
  caches    startup/caches.py env           (one persistent cache root: compile once)
  graphmem  startup/ sitecustomize hook     (FASTRSI_GRAPHMEM=1)
  lab       --worker-extension-cls fastrsi_lab_ext.LabExtension, VLLM_SERVER_DEV_MODE=1,
            bound to 127.0.0.1 only (dev mode lets any caller run code in the engine)

Usage:
  python3 engine.py start --model M --tp 4 --max-model-len L --cache-root C --log serve.log \
      [--graphmem] [--hooks mod1,mod2] [-- extra vllm serve args]
  python3 engine.py report --log serve.log --cache-root C
Writes only --log, <log>.cache_before.json and files under --cache-root.
"""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # repos/vllm/livelab
sys.path.insert(0, str(HERE / "startup"))
import caches, phases  # noqa: E402


def start(a, extra):
    env = os.environ.copy()
    env.update(caches.env(a.cache_root))
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "startup"), str(HERE)] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if a.graphmem:
        env["FASTRSI_GRAPHMEM"] = "1"
        env["FASTRSI_GRAPHMEM_DIR"] = str(Path(a.cache_root) / "graphmem")
    if a.hooks:
        env["FASTRSI_HOOKS"] = a.hooks
    cmd = [a.vllm, "serve", a.model, "--load-format", a.load_format, "--host", "127.0.0.1",
           "--port", str(a.port), "--tensor-parallel-size", str(a.tp),
           "--max-model-len", str(a.max_model_len),
           "--worker-extension-cls", "fastrsi_lab_ext.LabExtension", *extra]
    if a.load_format == "ipc_cache":
        # By default ipc_cache quietly reads weights from disk when the preload process is missing
        # or does not match. With fallback off the engine refuses to start instead, so every
        # engine that runs is known to use the preloaded weights; nothing to check at run time.
        cmd += ["--model-loader-extra-config", json.dumps({"fallback": False})]
    before = caches.snapshot(a.cache_root)
    Path(a.log + ".cache_before.json").write_text(json.dumps(before))
    p = subprocess.Popen(cmd, stdout=open(a.log, "w"), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    return {"pid": p.pid, "cmd": cmd}


FAILED = ("Engine core initialization failed", "Traceback (most recent call last)")


def wait_healthy(port, pid, log, timeout=3600):
    """Seconds until /health answers 200, or None as soon as the server exits or its log shows
    a failed start (the API process can outlive a dead engine core, so the pid alone is not
    enough)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not os.path.exists(f"/proc/{pid}"):
            return None
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        if any(s in text for s in FAILED):
            return None
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(5)
    return None


def report(a):
    out = {"phases": phases.parse(a.log)}
    before = json.loads(Path(a.log + ".cache_before.json").read_text())
    out["cache_growth"] = caches.diff(before, caches.snapshot(a.cache_root))
    text = open(a.log, errors="replace").read()
    out["graphmem_reused"] = text.count("fastrsi graphmem: reused")
    # No weights-from-IPC check: start() runs ipc_cache with fallback off (see there).
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["start", "report"])
    p.add_argument("--model"); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache-root", required=True); p.add_argument("--log", required=True)
    p.add_argument("--load-format", default="ipc_cache"); p.add_argument("--vllm", default="vllm")
    p.add_argument("--graphmem", action="store_true"); p.add_argument("--hooks", default="")
    a, extra = p.parse_known_args()
    extra = [x for x in extra if x != "--"]
    if a.cmd == "start":
        r = start(a, extra)
        r["healthy_after_s"] = wait_healthy(a.port, r["pid"], a.log)
        print(json.dumps(r))
    else:
        print(json.dumps(report(a), indent=1))
