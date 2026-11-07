"""Engine launcher: start `vllm serve`, directly or forked from the zygote (zygote.py), wait until
it answers /health, and report whether its processes were forked or spawned.

Usage:
  python3 engine.py start --model M --port P --cache-root C --log serve.log \
      [--tp N] [--max-model-len L] [--fork] [--zygote SOCK] [-- extra vllm serve args]
  python3 engine.py zygote --python <venv python> --zygote SOCK --cache-root C --log z.log \
      [--fork] [--zygote-modules FILE]
  python3 engine.py start ... --zygote SOCK --zygote-check   # would the zygote fork it now?

`start` prints one JSON line: {"pid", "cmd", "t_launch", "healthy_after_s", ...}. pid leads a new
session (setsid): stop the engine by that pid or its session, never by the caller's group.

Contract:
- engine_env() is the one environment of every engine process. The zygote is started with it
  too, so a start that matches the zygote's settings passes the zygote's "env" check.
- Weights load from disk (`--load-format auto`) unless the caller asks for another format.
- The server binds 127.0.0.1 only.
- healthy_after_s is seconds from launch until /health answers 200, or null as soon as the
  server exits or its log shows a failed start. A port that already answers is a finding and
  nothing starts, so the number is always about this engine.
- With --fork, fork_check() reports whether vLLM fell back to spawning its processes.

Writes only --log and files under --cache-root (and the zygote's socket).

Not here (other boundaries, later PRs): the weights preload daemon and its `ipc_cache` load
format; the startup hooks (graph memory, KV plan, overlay, autotune key, forksafe); the lab
worker extension and dev mode. Pass their flags after `--` or set their variables in the
environment until they land.
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zygote  # noqa: E402

# One folder per compiler/autotune cache under --cache-root, so later starts with the same
# config reuse compiled kernels, and concurrent runs with different roots never share a JIT
# library mid-run.
CACHE_LAYOUT = {
    "VLLM_CACHE_ROOT": "vllm",                  # torch.compile artifacts, autotune winners
    "TRITON_CACHE_DIR": "triton",               # Triton kernels
    "TORCHINDUCTOR_CACHE_DIR": "inductor",      # Inductor
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",  # FlashInfer JIT modules
    "TILELANG_CACHE_DIR": "tilelang",           # TileLang kernels
    "CUDA_CACHE_PATH": "cuda",                  # driver PTX JIT cache
}


def cache_env(root):
    root = Path(root).resolve()
    return {k: str(root / sub) for k, sub in CACHE_LAYOUT.items()}


def engine_env(a, base=None):
    """The environment every engine process runs with, and the zygote's (start_zygote)."""
    env = dict(os.environ if base is None else base)
    env.update(cache_env(a.cache_root))
    if a.fork:
        # The engine core and workers are forked, not spawned: they inherit the parent's
        # imports. vLLM falls back to spawn when the parent has initialized CUDA; fork_check()
        # reports that. TODO(startup boundary): forksafe keeps CUDA uninitialized in the API
        # server so the fork actually happens on servers that probe the GPU arch at import.
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        # CuTe DSL probes the GPU arch with cuInit at import (from flashinfer), which makes a
        # forked child's CUDA fail. Give it the arch it would detect, from nvidia-smi, which
        # does not touch the CUDA driver.
        arch = cute_dsl_arch(env.get("CUDA_VISIBLE_DEVICES"))
        if arch:
            env.setdefault("CUTE_DSL_ARCH", arch)
    return env


def cute_dsl_arch(visible):
    """CuTe DSL's arch name (its detect_gpu_arch formula) for the visible GPUs, or None."""
    q = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    if visible:
        q += ["-i", visible]
    try:
        caps = set(subprocess.check_output(q, text=True, timeout=30).split())
    except (OSError, subprocess.SubprocessError) as e:
        caps, why = set(), f"nvidia-smi failed: {e!r}"
    else:
        why = f"visible GPUs differ in compute capability {sorted(caps)}"
    if len(caps) != 1:
        print(json.dumps({"finding": {
            "check": "cute_dsl_arch", "observed": why, "expected": "one compute capability",
            "why": "CUTE_DSL_ARCH stays unset, so CuTe DSL probes the GPU with cuInit and "
                   "forked workers cannot use CUDA",
            "actions": ["set CUTE_DSL_ARCH yourself", "start without --fork"]}}),
            file=sys.stderr)
        return None
    major, minor = (int(x) for x in caps.pop().split("."))
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def port_in_use(port, host="127.0.0.1"):
    """True when something already listens on host:port (a connect succeeds). Not a bind test:
    a bind also fails on a privileged port or a TIME_WAIT socket, where nothing answers."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        return s.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def serve_cmd(a, extra):
    """The `vllm serve` argv. --load-format defaults to `auto` (safetensors from disk).
    TODO(weights boundary): `ipc_cache` needs the preload daemon, which is not in this folder
    yet; with no daemon a start on it waits for one (seen: a start that never finished)."""
    cmd = [a.vllm, "serve", a.model, "--load-format", a.load_format, "--host", "127.0.0.1",
           "--port", str(a.port), "--tensor-parallel-size", str(a.tp)]
    if a.max_model_len:
        cmd += ["--max-model-len", str(a.max_model_len)]
    return cmd + list(extra)


def _accept(a):
    return [x for x in a.zygote_accept.split(",") if x]


def start(a, extra):
    """Launch the server; return {"pid", "cmd", "t_launch", ...} or exit 2 with a finding."""
    env = engine_env(a)
    cmd = serve_cmd(a, extra)
    if a.zygote and a.zygote_check:
        # Ask only: would the zygote fork this server now? (callers check before stopping the
        # engine that runs, so a refusal leaves that engine up)
        reply = zygote.check(a.zygote, env, accept=_accept(a))
        print(json.dumps({"zygote_check": reply}))
        sys.exit(0 if reply.get("ok") else 2)
    if port_in_use(a.port) and not a.accept_port_in_use:
        # Another server on this port answers /health at once: wait_healthy would report the
        # new engine healthy after ~0.01 s while it dies on bind.
        print(json.dumps({"finding": {
            "check": "port_in_use", "observed": f"127.0.0.1:{a.port} already has a listener",
            "expected": "a free port", "why": "the health check would read the other server's "
            "/health as this start's; the new server then fails to bind",
            "actions": ["pick a free port (cost: none)", "--accept-port-in-use: start anyway "
                        "(the health result is then not about this engine)"],
            "risk": "a start reported healthy that never ran"}, "note": "not started"}, indent=1))
        sys.exit(2)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    t_launch = time.time()
    if a.zygote:
        # Empty the log first: the forked child truncates it only once it runs, and until then
        # wait_healthy would read the previous start's text (a Traceback there reads as a
        # failed start, and the new engine is left running unwatched).
        open(a.log, "w").close()
        reply = zygote.request(a.zygote, cmd, env, os.path.abspath(a.log), accept=_accept(a))
        if not reply.get("ok"):
            print(json.dumps({"zygote_findings": reply.get("findings"), "error": reply.get("error"),
                              "note": "not started; act on the findings or pass "
                                      "--zygote-accept"}, indent=1))
            sys.exit(2)
        out = {"pid": reply["pid"], "cmd": cmd, "t_launch": t_launch, "zygote": a.zygote,
               "zygote_check_seconds": reply.get("check_seconds"),
               "zygote_accepted_findings": reply.get("accepted_findings", [])}
        if reply.get("refreshed"):
            out["zygote_refreshed"] = reply["refreshed"]
        return out
    with open(a.log, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
    return {"pid": p.pid, "cmd": cmd, "t_launch": t_launch}


def start_zygote(a):
    """Start zygote.py detached, with exactly the environment start() gives the engine."""
    env = engine_env(a)
    err = zygote.sock_path_error(a.zygote)
    if err:
        return {"ready": False, "error": err}
    cmd = [a.python, str(Path(__file__).resolve().parent / "zygote.py"), "serve", "--sock",
           a.zygote]
    if a.zygote_modules:
        cmd += ["--modules", a.zygote_modules]
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
    return {"pid": p.pid, "cmd": cmd, **zygote.wait_ready(a.zygote, p.pid)}


FAILED = ("Engine core initialization failed", "Traceback (most recent call last)")


def wait_healthy(port, pid, log, timeout=3600, t0=None, poll=0.25):
    """Seconds from t0 (default: now) until /health answers 200, or None as soon as the server
    exits (a zombie counts: a direct start is our unreaped child) or its log shows a failed
    start (the API process can outlive a dead engine core, so the pid alone is not enough)."""
    t0 = time.time() if t0 is None else t0
    while time.time() - t0 < timeout:
        if zygote._exited(pid):
            return None
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        if any(s in text for s in FAILED):
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return round(time.time() - t0, 2)
        except Exception:
            pass
        time.sleep(poll)
    return None


SPAWN_FORCED = "We must use the `spawn` multiprocessing start method"
SPAWNED = "fastrsi forksafe: spawned interpreter"


def fork_check(log):
    """With --fork, every process after the API server must be forked. vLLM logs SPAWN_FORCED
    (with its reason) when it falls back to spawn. SPAWNED lines come from the forksafe startup
    hook, one per spawned interpreter; without that hook their count is always 0, so
    `spawned_probe` says whether the count means anything."""
    text = open(log, errors="replace").read()
    return {"spawn_forced": SPAWN_FORCED in text, "spawned_interpreters": text.count(SPAWNED),
            "spawned_probe": os.environ.get("FASTRSI_FORKSAFE") == "1"}


def fork_finding(fc):
    """None when the fork applied, else the finding (checks advise; the server still runs)."""
    if not fc["spawn_forced"] and fc["spawned_interpreters"] == 0:
        return None
    return {"check": "fork", "observed": fc,
            "expected": "no spawned interpreter after the API server",
            "why": "vLLM fell back to spawn: each process imports vLLM again (~11-13 s)",
            "actions": ["find what initialized CUDA in the parent: the reason after "
                        f"\"{SPAWN_FORCED}\" in the log", "accept: the server is correct, "
                        "only slower"],
            "risk": "none for correctness; startup is not faster"}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["start", "zygote"])
    p.add_argument("--model"); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache-root", required=True); p.add_argument("--log", required=True)
    p.add_argument("--load-format", default="auto",
                   help="vLLM load format (default auto: weights from disk)")
    p.add_argument("--vllm", default="vllm", help="the vllm executable")
    p.add_argument("--fork", action="store_true",
                   help="fork the engine core and workers instead of spawning them")
    p.add_argument("--zygote", default="", help="start: fork from the zygote at this socket; "
                   "zygote: start one listening there (log: --log)")
    p.add_argument("--zygote-check", action="store_true",
                   help="start: only ask the zygote whether it would fork this server; print "
                        "{\"zygote_check\": reply} and exit (0 ok, 2 refused)")
    p.add_argument("--zygote-accept", default="",
                   help="start: accept these zygote findings and fork anyway (env,stale)")
    p.add_argument("--zygote-modules", default="", help="zygote: file with modules to pre-import")
    p.add_argument("--python", default=sys.executable, help="zygote: the vLLM venv's python")
    p.add_argument("--accept-port-in-use", action="store_true",
                   help="start: launch although the port already has a listener")
    p.add_argument("--timeout", type=float, default=3600, help="start: /health wait, seconds")
    a, extra = p.parse_known_args(argv)
    return a, [x for x in extra if x != "--"]


def main(argv=None):
    a, extra = parse(argv)
    if a.cmd == "zygote":
        r = start_zygote(a)
        print(json.dumps(r))
        return 0 if r.get("ready") else 1
    r = start(a, extra)
    r["healthy_after_s"] = wait_healthy(a.port, r["pid"], a.log, timeout=a.timeout,
                                        t0=r["t_launch"])
    if a.fork or a.zygote:
        r["fork_check"] = fork_check(a.log)
        r["fork_finding"] = fork_finding(r["fork_check"])
    print(json.dumps(r))
    return 0 if r["healthy_after_s"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
