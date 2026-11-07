"""Engine launcher: start `vllm serve`, directly or forked from the zygote (zygote.py), wait until
it answers /health, and report whether its processes were forked or spawned.

Usage:
  python3 engine.py start --cache-root C --log serve.log [--fork] [--zygote SOCK] \
      [--weights] [--kvplan[=DIR]] MODEL [any `vllm serve` flag: --port P -tp N ...]
  python3 engine.py zygote --python <venv python> --zygote SOCK --cache-root C --log z.log \
      [--fork] [--zygote-modules FILE]
  python3 engine.py start ... --zygote SOCK --zygote-check   # would the zygote fork it now?

`start` prints one JSON line: {"pid", "cmd", "t_launch", "healthy_after_s", "vllm_args", ...}.
pid leads a new session (setsid): stop the engine by that pid or its session, never by the
caller's group.

Contract:
- Two parsers, no overlap. The launcher's own parser holds only launcher flags (below, parse());
  every other argument is the `vllm serve` argv and passes through verbatim, in order (a `--`
  separator is dropped). vLLM's own parser is the one source of truth for it: the launcher reads
  back the model, port, host, tensor parallel size, load format, eager mode and worker class
  with vllm_args.py. With --zygote the zygote parses (~0.2 s, vLLM imported already); without
  it the launcher imports vLLM's parser itself (~5 s), after setting engine_env(), so vLLM's
  import-time reads of the environment match the engine's. It has to run under the vLLM venv's
  python then.
- A start needs the model in the argv (positional or --model): without one vLLM would serve its
  default model, and that is a finding. An argv vLLM rejects is a finding too, with vLLM's
  message; nothing starts in either case.
- engine_env() is the one environment of every engine process. The zygote is started with it
  too, so a start that matches the zygote's settings passes the zygote's "env" check.
- Weights load with vLLM's default load format (`auto`: from disk) unless the argv asks for
  another one.
- A holder flag adds the vLLM flag its holder needs, unless the argv set that flag itself
  ("explicit": given in the argv, even at vLLM's default). An explicit value the holder cannot
  use is a finding naming both values, and nothing starts:
  - --weights appends `--load-format ipc_cache` (and the loader config, unless the argv has
    `--model-loader-extra-config`); an explicit `--load-format` other than ipc_cache conflicts,
    and so does an explicit loader config that leaves vLLM's disk fallback on (`"fallback"`
    true or missing: vLLM's default is true), since --weights promises no disk fallback.
    The daemon folder is the explicit config's socket_dir when the argv has a config (that is
    what vLLM dials), else --weights-socket-dir; both given and different is a finding.
    With ipc_cache (from --weights or the argv) the weight cache daemon of weights.py is probed
    first: when none answers for a GPU of this engine, the launcher prints a weights_daemon
    finding (exit 2) instead of a start that waits 300 s.
  - --kvplan appends `--worker-cls kvplan.KVPlanWorker`; an explicit other `--worker-cls`
    conflicts. It reuses the KV plan of an earlier start with the same config (kvplan.py), so
    the profile run of startup step 4c is skipped; the report counts reuses and findings.
    Give a folder as `--kvplan=DIR`: a bare `--kvplan` followed by the model takes the model
    as the folder (the missing model is then a finding).
- The server binds 127.0.0.1 only: the launcher appends `--host 127.0.0.1`, and an argv that
  sets another host is a finding.
- healthy_after_s is seconds from launch until /health answers 200, or null as soon as the
  server exits or its log shows a failed start. A port that already answers is a finding and
  nothing starts, so the number is about this engine; with --accept-port-in-use it starts
  anyway, and the number may describe another server.
- With --fork, fork_check() reports whether vLLM fell back to spawning its processes.

Writes only --log, files under --cache-root, the --kvplan=DIR folder when given, and the
zygote's socket.

Not here: the zygote installs forksafe.py (zygote.py); the lab worker extension is
adapter/worker_ext.py. The other startup hooks (graph memory, overlay, autotune key) and dev
mode are not built. Pass their vLLM flags in the argv or set their variables in the
environment.
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvplan, vllm_args, weights, zygote  # noqa: E402

KVPLAN_WORKER = "kvplan.KVPlanWorker"

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
    if a.kvplan is not None:
        # KVPlanWorker is imported by name in every worker: this folder goes on the path.
        here = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join([here] + ([env["PYTHONPATH"]]
                                                      if env.get("PYTHONPATH") else []))
        env["FASTRSI_KVPLAN_DIR"] = kvplan_dir(a)
    if a.fork:
        # The engine core and workers are forked, not spawned: they inherit the parent's
        # imports. vLLM falls back to spawn when the parent has initialized CUDA; fork_check()
        # reports that. A server forked from the zygote inherits forksafe.py, which keeps
        # CUDA uninitialized in the API server when a library probes the GPU arch at import
        # (FlashInfer at TP>1). TODO(startup boundary): the same for a direct start.
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


def read_vllm_args(a, vargv, env):
    """vllm_args.parse() of the `vllm serve` argv: by the zygote with --zygote, else here, after
    setting env, since importing vLLM reads it. Adds "via"; {"ok": False, "error"} on failure."""
    if a.zygote:
        return {**zygote.parse(a.zygote, vargv, accept=_accept(a)), "via": "zygote"}
    os.environ.update(env)
    try:
        r = vllm_args.parse(vargv)
    except ImportError as e:
        r = {"ok": False, "error": f"vLLM's parser is not importable by {sys.executable} ({e!r}): "
                                   "run the launcher with the vLLM venv's python, or use --zygote"}
    return {**r, "via": "direct"}


def kvplan_dir(a):
    return str(Path(a.kvplan or Path(a.cache_root) / "kvplan").resolve())


def load_format(a, v):
    """The load format the server will use: the argv's if explicit, else ipc_cache with
    --weights, else vLLM's default."""
    if a.weights and "load_format" not in v["explicit"]:
        return "ipc_cache"
    return v["args"]["load_format"]


def explicit_loader_config(v):
    """The argv's --model-loader-extra-config as a dict, or None when the argv has none."""
    if "model_loader_extra_config" not in v["explicit"]:
        return None
    cfg = v["args"].get("model_loader_extra_config")
    return cfg if isinstance(cfg, dict) else {}


def socket_dir(a, v):
    """The daemon socket folder this start dials. With an explicit --model-loader-extra-config
    vLLM gets that config unchanged, so its socket_dir (else vLLM's default); otherwise
    --weights-socket-dir (else vLLM's default), which the launcher's loader config names."""
    cfg = explicit_loader_config(v)
    if cfg is not None:
        return cfg.get("socket_dir") or weights.default_socket_dir()
    return a.weights_socket_dir or weights.default_socket_dir()


def argv_findings(a, v):
    """Findings about the parsed `vllm serve` argv that stop the start (each names both values
    or what is missing)."""
    if not v["ok"] and v.get("findings"):        # the zygote refused to parse (stale)
        return v["findings"]
    if not v["ok"] and v.get("zygote_error"):    # vLLM never judged the argv
        return [{"check": "zygote_parse", "observed": v.get("error"), "expected": "the "
                 "zygote's parse child reads the argv with vLLM's parser", "why": "the child "
                 "failed before vLLM's parser ran (an edited module that does not import, or a "
                 "refresh that failed); a forked server would fail the same way",
                 "actions": ["fix the edited module (the error above)", "start without "
                             "--zygote to see the full traceback", "restart the zygote"]}]
    if not v["ok"]:
        return [{"check": "vllm_args", "observed": v.get("error"), "expected": "an argv that "
                 "`vllm serve` accepts", "why": "the server would exit on it",
                 "actions": ["fix the vLLM flags (vLLM's message above)"]}]
    args, explicit, found = v["args"], v["explicit"], []
    if "model" not in explicit:
        found.append({"check": "vllm_model", "observed": f"no model in the argv; vLLM's default "
                      f"is {args['model']}", "expected": "the model, positional or --model",
                      "why": "the server would load vLLM's default model, not the one meant "
                      "(a flag that takes an optional value can swallow a positional model)",
                      "actions": ["pass the model as the first vLLM argument"]})
    if "host" in explicit and args["host"] != "127.0.0.1":
        found.append({"check": "vllm_host", "observed": f"--host {args['host']}",
                      "expected": "--host 127.0.0.1 (the launcher's)", "why": "lab engines "
                      "bind 127.0.0.1 only", "actions": ["drop --host"]})
    if a.weights and "load_format" in explicit and args["load_format"] != "ipc_cache":
        found.append({"check": "weights_load_format", "observed": f"--weights and the argv's "
                      f"--load-format {args['load_format']}", "expected": "--load-format "
                      "ipc_cache, or none (--weights adds it)", "why": "--weights maps the "
                      "daemon's weights over IPC; the argv asks vLLM to load them another way",
                      "actions": ["drop --load-format", "drop --weights"]})
    cfg = explicit_loader_config(v)
    if a.weights and cfg is not None and cfg.get("fallback", True) is not False:
        found.append({"check": "weights_fallback", "observed": "--weights and the argv's "
                      "--model-loader-extra-config with fallback " + (json.dumps(cfg["fallback"])
                      if "fallback" in cfg else "unset (vLLM's default: true)"),
                      "expected": '"fallback": false in that config, or no config (--weights '
                      "adds one)", "why": "with the fallback on, an engine that cannot map the "
                      "daemon's weights reads the checkpoint from disk silently; --weights "
                      "promises resident weights or a failed load",
                      "actions": ['add "fallback": false to the config',
                                  "drop --model-loader-extra-config", "drop --weights"]})
    if (a.weights_socket_dir and cfg is not None and load_format(a, v) == "ipc_cache"
            and os.path.abspath(a.weights_socket_dir) != os.path.abspath(socket_dir(a, v))):
        found.append({"check": "weights_socket_dir", "observed": f"--weights-socket-dir "
                      f"{a.weights_socket_dir} and the argv's loader config, whose socket "
                      f"folder is {socket_dir(a, v)}", "expected": "one folder", "why": "vLLM "
                      "dials the config's folder; the daemon probe would check the other one",
                      "actions": ["drop --weights-socket-dir", "set socket_dir in the config "
                                  "to the same folder"]})
    if a.kvplan is not None and "worker_cls" in explicit and args["worker_cls"] != KVPLAN_WORKER:
        found.append({"check": "kvplan_worker_cls", "observed": f"--kvplan and the argv's "
                      f"--worker-cls {args['worker_cls']}", "expected": f"--worker-cls "
                      f"{KVPLAN_WORKER}, or none (--kvplan adds it)", "why": "the KV plan lives "
                      "in that worker class; with another one every start profiles",
                      "actions": ["drop --worker-cls", "drop --kvplan"],
                      "risk": "none for correctness; startup not faster"})
    return found


def serve_cmd(a, vargv, v):
    """The `vllm serve` argv: the caller's, verbatim, then what the launcher and the holders
    need that the caller did not set (host, load format and loader config, worker class).
    Flags go after the caller's: `vllm serve` wants the model first when --served-model-name
    is given."""
    cmd, explicit = [a.vllm, "serve", *vargv], v["explicit"]
    if "host" not in explicit:
        cmd += ["--host", "127.0.0.1"]
    if a.weights and "load_format" not in explicit:
        cmd += ["--load-format", "ipc_cache"]
    if load_format(a, v) == "ipc_cache" and "model_loader_extra_config" not in explicit:
        cmd += ["--model-loader-extra-config", json.dumps(weights.loader_config(socket_dir(a, v)))]
    if a.kvplan is not None and "worker_cls" not in explicit:
        cmd += ["--worker-cls", KVPLAN_WORKER]
    return cmd


def preflight(a, v, env):
    """Holder checks that stop this start before anything runs (each names its override), and
    the report fields of the checks that ran."""
    found, rep = [], {}
    if load_format(a, v) == "ipc_cache":
        pr = weights.probe(socket_dir(a, v), env.get("CUDA_VISIBLE_DEVICES"),
                           v["args"]["tensor_parallel_size"])
        rep["weights_probe"] = pr
        if not pr["ok"] and not a.weights_accept_missing:
            found.append(weights.daemon_finding(pr))
    return found, rep


def kvplan_report(log):
    """The KV plan's log lines (one per rank per start): counts, seconds and findings."""
    rep = {"reused": 0, "measured": 0, "profiled": 0, "seconds": [], "findings": []}
    for line in open(log, errors="replace"):
        i = line.find(kvplan.TAG)
        j = line.find("{", i)
        if i < 0 or j < 0:
            continue
        try:
            ev = json.loads(line[j:])
        except ValueError:
            continue
        if line[i:j].startswith(kvplan.TAG + " finding"):
            rep["findings"].append(ev)
        elif ev.get("event") in ("reused", "measured", "profiled"):
            rep[ev["event"]] += 1
            rep["seconds"].append(ev.get("seconds"))
    return rep


def _accept(a):
    return [x for x in a.zygote_accept.split(",") if x]


def start(a, vargv):
    """Launch the server; return {"pid", "cmd", "t_launch", "port", "vllm_args", ...} or exit 2
    with a finding."""
    env = engine_env(a)
    if a.zygote and a.zygote_check:
        # Ask only: would the zygote fork this server now? (callers check before stopping the
        # engine that runs, so a refusal leaves that engine up)
        reply = zygote.check(a.zygote, env, accept=_accept(a))
        print(json.dumps({"zygote_check": reply}))
        sys.exit(0 if reply.get("ok") else 2)
    v = read_vllm_args(a, vargv, env)
    found, pre = argv_findings(a, v), {}
    if not found:
        found, pre = preflight(a, v, env)
    if found:
        print(json.dumps({"findings": found, **pre, "vllm_args": v, "note": "not started"}))
        sys.exit(2)
    cmd, port = serve_cmd(a, vargv, v), v["args"]["port"]
    rep = {"port": port, "vllm_args": v, **pre}
    if port_in_use(port) and not a.accept_port_in_use:
        # Another server on this port answers /health at once: wait_healthy would report the
        # new engine healthy after ~0.01 s while it dies on bind.
        print(json.dumps({"finding": {
            "check": "port_in_use", "observed": f"127.0.0.1:{port} already has a listener",
            "expected": "a free port", "why": "the health check would read the other server's "
            "/health as this start's; the new server then fails to bind",
            "actions": ["pick a free port (cost: none)", "--accept-port-in-use: start anyway "
                        "(the health result is then not about this engine)"],
            "risk": "a start reported healthy that never ran"}, "note": "not started"}))
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
                                      "--zygote-accept"}))
            sys.exit(2)
        out = {"pid": reply["pid"], "cmd": cmd, "t_launch": t_launch, **rep, "zygote": a.zygote,
               "zygote_check_seconds": reply.get("check_seconds"),
               "zygote_accepted_findings": reply.get("accepted_findings", [])}
        if reply.get("refreshed"):
            out["zygote_refreshed"] = reply["refreshed"]
        return out
    with open(a.log, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
    return {"pid": p.pid, "cmd": cmd, "t_launch": t_launch, **rep}


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


def fork_check(log):
    """With --fork, every process after the API server must be forked. vLLM logs SPAWN_FORCED
    (with its reason) when it falls back to spawn; that line is the only spawn signal read."""
    text = open(log, errors="replace").read()
    return {"spawn_forced": SPAWN_FORCED in text}


def fork_finding(fc):
    """None when the fork applied, else the finding (checks advise; the server still runs)."""
    if not fc["spawn_forced"]:
        return None
    return {"check": "fork", "observed": fc,
            "expected": "no spawned interpreter after the API server",
            "why": "vLLM fell back to spawn: each process imports vLLM again (~11-13 s)",
            "actions": ["find what initialized CUDA in the parent: the reason after "
                        f"\"{SPAWN_FORCED}\" in the log", "accept: the server is correct, "
                        "only slower"],
            "risk": "none for correctness; startup is not faster"}


def launcher_parser():
    """Only launcher flags live here; none of them is a `vllm serve` flag (test_vllm_args.py
    checks this against vLLM's parser), and abbreviations are off, so a vLLM flag never lands
    here by prefix."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("cmd", choices=["start", "zygote"])
    p.add_argument("--cache-root", required=True, help="folder for the launcher's caches")
    p.add_argument("--log", required=True, help="the server's log file")
    p.add_argument("--vllm", default="vllm", help="start: the vllm executable")
    p.add_argument("--weights", action="store_true",
                   help="start: map the weights held by the daemon of weights.py (adds "
                        "--load-format ipc_cache, no disk fallback); off by default")
    p.add_argument("--weights-socket-dir", default="",
                   help="the daemon's --socket-dir (default: vLLM's default folder)")
    p.add_argument("--weights-accept-missing", action="store_true",
                   help="start: launch although no daemon answers (it then waits up to 300 s)")
    p.add_argument("--kvplan", nargs="?", const="", default=None, metavar="DIR",
                   help="reuse the KV plan of an earlier start with the same config (adds "
                        "--worker-cls kvplan.KVPlanWorker; plans in DIR, default "
                        "<cache-root>/kvplan; give DIR as --kvplan=DIR); off by default")
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
    return p


def parse(argv=None):
    """(launcher args, `vllm serve` argv: everything the launcher's parser does not know)."""
    a, vargv = launcher_parser().parse_known_args(argv)
    return a, [x for x in vargv if x != "--"]


def main(argv=None):
    a, vargv = parse(argv)
    if a.cmd == "zygote":
        r = start_zygote(a)
        print(json.dumps(r))
        return 0 if r.get("ready") else 1
    r = start(a, vargv)
    r["healthy_after_s"] = wait_healthy(r["port"], r["pid"], a.log, timeout=a.timeout,
                                        t0=r["t_launch"])
    if a.kvplan is not None:
        r["kvplan"] = kvplan_report(a.log)
    if a.fork or a.zygote:
        r["fork_check"] = fork_check(a.log)
        r["fork_finding"] = fork_finding(r["fork_check"])
    print(json.dumps(r))
    return 0 if r["healthy_after_s"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
