"""Engine launcher: start or reconfigure `vllm serve`, directly or forked from the zygote
(zygote.py), wait until it answers /health, and report whether its processes were forked or
spawned.

Usage:
  python3 engine.py start --cache-root C --log serve.log [--fork] [--zygote SOCK] \
      [--weights] [--kvplan[=DIR]] MODEL [any `vllm serve` flag: --port P -tp N ...]
  python3 engine.py reconfigure --pid PID --cache-root C --log serve2.log \
      [any `vllm serve` flag override: --max-model-len L2 --no-enforce-eager ...]
  python3 engine.py zygote --python <venv python> --zygote SOCK --cache-root C --log z.log \
      [--fork] [--zygote-modules FILE]
  python3 engine.py start ... --zygote SOCK --zygote-check   # would the zygote fork it now?
  python3 engine.py pool {copy,demo} ...   # weight lease / NCCL seed / standby pool
      (weight_scaling_colocate.py; its own flags, see that file)

`start` and `reconfigure` print one JSON line: {"pid", "cmd", "t_launch", "healthy_after_s",
"vllm_args", ...} (plus "reconfig" on `reconfigure`). pid leads a new session (setsid): stop
the engine by that pid or its session, never by the caller's group.

Contract:
- Two parsers, no overlap. The launcher's own parser holds only launcher flags (below, parse());
  every other argument is the `vllm serve` argv and passes through verbatim, in order (a `--`
  separator is dropped). vLLM's own parser is the one source of truth for it: the launcher reads
  back the model, port, host, tensor parallel size, load format, eager mode, worker class and
  reconfiguration fields with vllm_args.py. With --zygote the zygote parses (~0.2 s, vLLM
  imported already); without it the launcher imports vLLM's parser itself (~5 s), after setting
  engine_env(), so vLLM's import-time reads of the environment match the engine's. It has to run
  under the vLLM venv's python then.
- A start needs the model in the argv (positional or --model): without one vLLM would serve its
  default model, and that is a finding. An argv vLLM rejects is a finding too, with vLLM's
  message; nothing starts in either case.
- `reconfigure --pid PID` reads the running engine's recorded `vllm serve` argv and launcher
  settings from `<cache-root>/engines/<PID>.json` (or `/proc/<PID>/cmdline`), applies the given
  flag overrides via `vllm_args.override()`, and classifies the delta into 4 tiers (see
  `repos/vllm/kb/reconfiguration_matrix.md`), then applies it one of two ways and reports which
  (`reconfig.path`):
  - `in_process_live`: every changed flag has a live consumer (live_reconfig.LIVE_FIELDS:
    `--max-num-seqs`, `--max-num-batched-tokens`, each at most its boot value). The launcher
    sends it to the engine core's live agent (started by engine.py in every engine it starts),
    which sets the scheduler's limits between two steps and reads them back. PID keeps
    running; only an acknowledged change updates the recorded state.
  - `restart`: everything else (`--max-model-len`, `--enable-prefix-caching`, `--enforce-eager`,
    CUDA graph flags, `--gpu-memory-utilization`, ...), and any live change the agent does not
    acknowledge: PID is stopped (verified) and the engine starts with the new argv; the
    report's `restart_reason` says why it was not live. With `--weights`, resident CUDA IPC
    weights make the restart warm (zero disk reload).
  Tier-2 deltas (`WeightCacheKey` sharding, `subtier='tier2a_reshard'`, or weight format,
  `'tier2b_weight_format'`) are refused before stopping PID unless `--weights-accept-reconfig`.
  Every preflight check runs BEFORE mutating or stopping PID; a refusal exits 2 and leaves PID
  running untouched.
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
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvplan, live_reconfig, vllm_args, weights, zygote  # noqa: E402

LIVE_SITE = str(Path(__file__).resolve().parent / "live_site")

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
    # The live reconfiguration agent (live_reconfig.py) in the engine core: its sitecustomize
    # goes on the path ahead of site-packages, and each launch gets its own control folder.
    if not getattr(a, "live_dir", None):
        a.live_dir = str(Path(a.cache_root).resolve() / "engines" / "live" / uuid.uuid4().hex)
    env["FASTRSI_LIVE_RECONFIG_DIR"] = a.live_dir
    paths = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and p != LIVE_SITE]
    env["PYTHONPATH"] = os.pathsep.join([LIVE_SITE] + paths)
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
    if a.fork or getattr(a, "zygote", ""):
        # CuTe DSL probes the GPU arch with cuInit at import (from flashinfer), which makes a
        # forked child's CUDA fail; the zygote forks every server, so it needs this too. Give
        # it the arch it would detect, from nvidia-smi, which does not touch the CUDA driver.
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
            "actions": ["set CUTE_DSL_ARCH yourself", "start without --fork or --zygote"]}}),
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


def reconfig_findings(launcher_args, parsed_vllm, reconfig):
    """Findings when a `reconfigure` with resident weights (`ipc_cache`) changes a Tier-2 field
    in `WeightCacheKey` (sharding or post-processed weight format)."""
    if not reconfig or reconfig["weights_compatible"]:
        return []
    if (load_format(launcher_args, parsed_vllm) != "ipc_cache"
            or getattr(launcher_args, "weights_accept_reconfig", False)):
        return []
    weight_key_diff = {
        field_name: reconfig["changed"][field_name]
        for field_name in reconfig["weight_key_changed"]
    }
    return [{
        "check": "weights_reconfig_tier2",
        "observed": f"{reconfig['tier']} change on WeightCacheKey fields: {weight_key_diff}",
        "expected": "Tier 1 engine flags only (max_model_len, enforce_eager, compilation_config, "
                    "max_num_batched_tokens, max_num_seqs, gpu_memory_utilization, "
                    "enable_prefix_caching, ...), or a restarted weights.py daemon matching the "
                    "new WeightCacheKey",
        "why": "weights.py holds post-processed, rank-sharded GPU tensors keyed by "
               "WeightCacheKey; IpcModelLoader rejects a mismatched WeightCacheKey with "
               "CacheConfigMismatchError after the old engine is stopped",
        "actions": [
            "restart weights.py with the new sharding/format flags, then pass "
            "--weights-accept-reconfig",
            "drop --weights / --load-format ipc_cache to load from disk",
            "keep WeightCacheKey flags unchanged for zero-copy IPC reconfiguration",
        ],
        "risk": "stopping the running engine for an ipc_cache start that fails at load_model",
    }]


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


def _engine_state_path(cache_root, pid):
    return Path(cache_root).resolve() / "engines" / f"{pid}.json"


def save_engine_state(launcher_args, pid, port, vllm_argv, command, parsed_vllm):
    """Record the launched engine's `vllm serve` argv and launcher settings under --cache-root so
    `engine.py reconfigure --pid <pid>` can inspect and override it (including zygote forks whose
    `/proc/<pid>/cmdline` stays the zygote's)."""
    state_path = _engine_state_path(launcher_args.cache_root, pid)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": pid,
        "port": port,
        "vargv": list(vllm_argv),
        "cmd": list(command),
        "vllm_args": parsed_vllm,
        "live_dir": getattr(launcher_args, "live_dir", None),
        "launcher": {
            "vllm": launcher_args.vllm,
            "weights": bool(launcher_args.weights),
            "weights_socket_dir": launcher_args.weights_socket_dir,
            "kvplan": launcher_args.kvplan,
            "fork": bool(launcher_args.fork),
            "zygote": launcher_args.zygote,
        },
    }
    temp_path = state_path.with_suffix(f".{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(record))
    temp_path.replace(state_path)


def _strip_injected_flags(argv):
    """Strip launcher-appended holder flags from a raw `/proc/<pid>/cmdline` `vllm serve` tail."""
    cleaned_argv = []
    index = 0
    while index < len(argv):
        if index + 1 < len(argv) and argv[index] == "--host" and argv[index + 1] == "127.0.0.1":
            index += 2
        elif (index + 1 < len(argv)
              and argv[index] == "--load-format"
              and argv[index + 1] == "ipc_cache"):
            index += 2
        elif index + 1 < len(argv) and argv[index] == "--model-loader-extra-config":
            index += 2
        elif (index + 1 < len(argv)
              and argv[index] == "--worker-cls"
              and argv[index + 1] == KVPLAN_WORKER):
            index += 2
        else:
            cleaned_argv.append(argv[index])
            index += 1
    return cleaned_argv


def load_engine_state(cache_root, pid):
    """Load the state of engine `pid` from `<cache-root>/engines/<pid>.json`, falling back to
    `/proc/<pid>/cmdline` when started directly without a state file."""
    state_path = _engine_state_path(cache_root, pid)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (OSError, ValueError):
            pass
    try:
        raw_tokens = [
            token for token in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            .split("\0") if token
        ]
    except OSError:
        return None
    if "serve" not in raw_tokens:
        return None
    serve_index = raw_tokens.index("serve")
    serve_tail = raw_tokens[serve_index + 1:]
    vllm_bin = raw_tokens[serve_index - 1] if serve_index >= 1 else "vllm"
    weights_on = (
        "--load-format" in serve_tail
        and serve_tail[serve_tail.index("--load-format") + 1:
                       serve_tail.index("--load-format") + 2] == ["ipc_cache"]
    )
    kvplan_on = (
        "--worker-cls" in serve_tail
        and serve_tail[serve_tail.index("--worker-cls") + 1:
                       serve_tail.index("--worker-cls") + 2] == [KVPLAN_WORKER]
    )
    socket_dir_val = ""
    if "--model-loader-extra-config" in serve_tail:
        try:
            loader_config = json.loads(
                serve_tail[serve_tail.index("--model-loader-extra-config") + 1]
            )
            socket_dir_val = loader_config.get("socket_dir") or ""
        except Exception:
            pass
    return {
        "pid": pid,
        "port": None,
        "vargv": _strip_injected_flags(serve_tail),
        "cmd": raw_tokens[serve_index - 1:] if serve_index >= 1 else raw_tokens,
        "vllm_args": None,
        "launcher": {
            "vllm": vllm_bin,
            "weights": weights_on,
            "weights_socket_dir": socket_dir_val,
            "kvplan": "" if kvplan_on else None,
            "fork": False,
            "zygote": "",
        },
    }


def _descendants(pid):
    children_by_parent, descendants = {}, []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return descendants
    for entry in entries:
        if entry.isdigit():
            try:
                parent_pid = int(
                    open(f"/proc/{entry}/stat").read().rsplit(")", 1)[1].split()[1]
                )
                children_by_parent.setdefault(parent_pid, []).append(int(entry))
            except Exception:
                pass
    stack = [pid]
    while stack:
        for child_pid in children_by_parent.get(stack.pop(), []):
            descendants.append(child_pid)
            stack.append(child_pid)
    return descendants


def _start_time(pid):
    """The process start time (clock ticks since boot, /proc/<pid>/stat field 22), or None when
    pid is gone. A pid with another start time is another process (pid reuse)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _same_process(pid, start_time):
    return start_time is not None and _start_time(pid) == start_time


def _signal_if_same(pid, start_time, sig):
    """Signal pid only if it is still the process recorded with start_time."""
    if not _same_process(pid, start_time):
        return False
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def stop_engine(pid, port=None, grace=10.0, kill_wait=5.0):
    """Stop engine `pid` and its descendants: SIGTERM, up to `grace` s, then SIGKILL, up to
    `kill_wait` s more. Each process is identified by (pid, start time) when first seen and is
    signalled only while it still has that start time, so a reused pid is never hit.
    Returns {"stopped", "seconds", "alive", "port_open"}: stopped is True only when every
    recorded process is gone (or a zombie) and `port` no longer answers; it is never assumed."""
    t0 = time.time()
    if zygote._exited(pid):
        return {"stopped": False, "seconds": 0.0, "alive": [], "port_open": None,
                "note": f"pid {pid} was not running"}
    procs = {p: _start_time(p) for p in [pid] + _descendants(pid)}
    procs = {p: st for p, st in procs.items() if st is not None}

    def alive():
        return sorted(p for p, st in procs.items() if _same_process(p, st) and not zygote._exited(p))

    def done():
        return not alive() and (port is None or not port_in_use(port))

    try:
        if os.getpgid(pid) == pid and _same_process(pid, procs.get(pid)):
            os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for p, st in procs.items():
        _signal_if_same(p, st, signal.SIGTERM)
    deadline = t0 + grace
    while time.time() < deadline and not done():
        time.sleep(0.05)
    if not done():
        for p in alive():
            _signal_if_same(p, procs[p], signal.SIGKILL)
        deadline = time.time() + kill_wait
        while time.time() < deadline and not done():
            time.sleep(0.05)
    left = alive()
    port_open = port_in_use(port) if port is not None else None
    return {"stopped": not left and not port_open, "seconds": round(time.time() - t0, 3),
            "alive": left, "port_open": port_open}


def _launch_process(launcher_args, vllm_argv, parsed_vllm, env, preflight_report,
                    extra_report=None):
    """Spawn or fork the validated `vllm serve` command and record its state."""
    command, port = (
        serve_cmd(launcher_args, vllm_argv, parsed_vllm),
        parsed_vllm["args"]["port"],
    )
    report = {
        "port": port,
        "vllm_args": parsed_vllm,
        **preflight_report,
        **(extra_report or {}),
    }
    Path(launcher_args.log).parent.mkdir(parents=True, exist_ok=True)
    t_launch = time.time()
    if launcher_args.zygote:
        # Empty the log first: the forked child truncates it only once it runs, and until then
        # wait_healthy would read the previous start's text (a Traceback there reads as a
        # failed start, and the new engine is left running unwatched).
        open(launcher_args.log, "w").close()
        reply = zygote.request(
            launcher_args.zygote,
            command,
            env,
            os.path.abspath(launcher_args.log),
            accept=_accept(launcher_args),
        )
        if not reply.get("ok"):
            print(json.dumps({
                "zygote_findings": reply.get("findings"),
                "error": reply.get("error"),
                **({"reconfig": extra_report["reconfig"]}
                   if extra_report and "reconfig" in extra_report else {}),
                "note": "not started; act on the findings or pass --zygote-accept",
            }))
            sys.exit(2)
        result = {
            "pid": reply["pid"],
            "cmd": command,
            "t_launch": t_launch,
            **report,
            "zygote": launcher_args.zygote,
            "zygote_check_seconds": reply.get("check_seconds"),
            "zygote_accepted_findings": reply.get("accepted_findings", []),
        }
        if reply.get("refreshed"):
            result["zygote_refreshed"] = reply["refreshed"]
        save_engine_state(launcher_args, result["pid"], port, vllm_argv, command, parsed_vllm)
        return result
    with open(launcher_args.log, "w") as log_file:
        proc = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    save_engine_state(launcher_args, proc.pid, port, vllm_argv, command, parsed_vllm)
    return {"pid": proc.pid, "cmd": command, "t_launch": t_launch, **report}


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
    port = v["args"]["port"]
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
    return _launch_process(a, vargv, v, env, pre)


def _inherit_launcher(launcher_args, saved_launcher, raw_argv):
    """Inherit launcher flags from the running engine's recorded state when not explicitly
    given on the `reconfigure` command line."""
    if not saved_launcher:
        return
    if "--vllm" not in raw_argv and saved_launcher.get("vllm"):
        launcher_args.vllm = saved_launcher["vllm"]
    if "--weights" not in raw_argv and saved_launcher.get("weights"):
        launcher_args.weights = True
    if "--weights-socket-dir" not in raw_argv and saved_launcher.get("weights_socket_dir"):
        launcher_args.weights_socket_dir = saved_launcher["weights_socket_dir"]
    if not any(token == "--kvplan" or token.startswith("--kvplan=") for token in raw_argv):
        if saved_launcher.get("kvplan") is not None:
            launcher_args.kvplan = saved_launcher["kvplan"]
    if "--fork" not in raw_argv and saved_launcher.get("fork"):
        launcher_args.fork = True
    if "--zygote" not in raw_argv and saved_launcher.get("zygote"):
        launcher_args.zygote = saved_launcher["zygote"]


LIVE_ACK_TIMEOUT_S = 15.0


def live_reconfig_plan(state, reconfig):
    """Can this delta be applied inside the running engine? (values, None) when the engine's
    live agent is up and every changed field is one it can set within the engine's boot limits;
    else (None, reason). Pure checks: nothing is sent."""
    if not reconfig or not reconfig.get("in_process") or not reconfig.get("changed"):
        return None, "the delta has fields without a live consumer"
    values = {f: new for f, (_, new) in reconfig["changed"].items()}
    live_dir = state.get("live_dir")
    ready = live_reconfig.read_ready(live_dir) if live_dir else None
    if not ready:
        return None, ("no live agent answered for this engine (started without engine.py's "
                      "live agent, or it failed to start: see the engine log)")
    if ready.get("api_pid") != state.get("pid"):
        return None, (f"the live agent belongs to api pid {ready.get('api_pid')}, not "
                      f"{state.get('pid')}")
    if zygote._exited(int(ready.get("pid") or 0)):
        return None, f"the live agent's engine core (pid {ready.get('pid')}) is gone"
    problems = live_reconfig.check_values(values, ready.get("boot") or {})
    if problems:
        return None, "; ".join(problems)
    return values, None


def apply_in_process_reconfig(launcher_args, state, new_vargv, new_parsed, preflight_report,
                              reconfig, values):
    """Apply `values` inside the running engine through its live agent and wait for the
    agent's acknowledgement. Only an acknowledged change updates the recorded engine state.
    Returns (report, None) on success, (None, reason) when the engine did not confirm."""
    t0 = time.perf_counter()
    pid = launcher_args.pid
    ack = live_reconfig.request(state["live_dir"], "set", timeout=LIVE_ACK_TIMEOUT_S,
                                values=values)
    apply_s = round(time.perf_counter() - t0, 6)
    if not ack.get("ok"):
        return None, f"live agent refused or did not answer: {ack.get('error')}"
    after = ack.get("after") or {}
    confirmed = (("max_num_seqs" not in values
                  or after.get("target_max_num_seqs") == values["max_num_seqs"])
                 and ("max_num_batched_tokens" not in values
                      or after.get("max_num_scheduled_tokens") == values["max_num_batched_tokens"]))
    if not confirmed:
        return None, f"live agent answered but the scheduler reads back {after}"
    command = serve_cmd(launcher_args, new_vargv, new_parsed)
    port = new_parsed["args"]["port"]
    launcher_args.live_dir = state["live_dir"]
    save_engine_state(launcher_args, pid, port, new_vargv, command, new_parsed)
    return {
        "pid": pid, "cmd": command, "t_launch": time.time(), "port": port,
        "vllm_args": new_parsed, **preflight_report,
        "reconfig": {**reconfig, "path": "in_process_live", "restarted": False,
                     "replaced_pid": None, "apply_seconds": apply_s, "live_ack": ack},
    }, None


def reconfigure(launcher_args, patch_vargv, raw_argv=()):
    """Reconfigure a running engine (`--pid PID`) with `patch_vargv` overrides:
    1. Verify `launcher_args.pid` is alive and read its base `vllm serve` argv and launcher config.
    2. Merge `patch_vargv` onto `base_vargv` via `vllm_args.override()` and classify the delta
       into `in_process_instant`, `in_process_graph`, `warm_weights_ipc`, or
       `live_reshard_or_cold`.
    3. Run all preflight checks (vLLM parser, WeightCacheKey compatibility, daemon socket probe,
       zygote check, port check) BEFORE mutating or stopping `launcher_args.pid`. Any refusal
       exits 2 with `launcher_args.pid` still running.
    4. Apply live (`path: in_process_live`) when the engine's live agent can set every changed
       flag and acknowledges it; otherwise stop `launcher_args.pid` (verified: a stop that
       leaves a process or the port behind is a finding, exit 2, nothing started) and launch
       the reconfigured engine (`path: restart`, `restart_reason`)."""
    if not launcher_args.pid or zygote._exited(launcher_args.pid):
        print(json.dumps({"findings": [{
            "check": "reconfigure_pid",
            "observed": (f"pid {launcher_args.pid} is not running"
                         if launcher_args.pid else "--pid not given"),
            "expected": "the pid of a running `vllm serve` started by engine.py",
            "why": "reconfigure reads the running engine's flags and replaces it only after "
                   "preflight checks pass",
            "actions": ["pass --pid <running engine pid>", "use `engine.py start` instead"],
        }], "note": "not started"}))
        sys.exit(2)
    state = load_engine_state(launcher_args.cache_root, launcher_args.pid)
    if state is None:
        print(json.dumps({"findings": [{
            "check": "reconfigure_pid",
            "observed": f"cannot read `vllm serve` state for pid {launcher_args.pid}",
            "expected": (f"state file under {launcher_args.cache_root}/engines/"
                         f"{launcher_args.pid}.json or `vllm serve` in "
                         f"/proc/{launcher_args.pid}/cmdline"),
            "why": "reconfigure needs the running engine's current flags to apply overrides",
            "actions": ["use the same --cache-root as `engine.py start`",
                        "use `engine.py start` with the full argv instead"],
        }], "note": "not started"}))
        sys.exit(2)
    _inherit_launcher(launcher_args, state.get("launcher"), raw_argv)
    env = engine_env(launcher_args)
    old_vargv = state["vargv"]
    new_vargv = vllm_args.override(old_vargv, patch_vargv)
    old_parsed = state.get("vllm_args")
    if not old_parsed or not old_parsed.get("ok"):
        old_parsed = read_vllm_args(launcher_args, old_vargv, env)
    new_parsed = read_vllm_args(launcher_args, new_vargv, env)
    reconfig = vllm_args.classify_reconfig(old_parsed, new_parsed) if (
        old_parsed.get("ok") and new_parsed.get("ok")
    ) else None
    findings, preflight_report = argv_findings(launcher_args, new_parsed), {}
    if not findings and reconfig is not None:
        findings = reconfig_findings(launcher_args, new_parsed, reconfig)
    if not findings:
        findings, preflight_report = preflight(launcher_args, new_parsed, env)
    if findings:
        print(json.dumps({"findings": findings, **preflight_report, "vllm_args": new_parsed,
                          **({"reconfig": reconfig} if reconfig else {}),
                          "note": "not started; old engine left running"}))
        sys.exit(2)
    old_port = state.get("port") or (old_parsed.get("args") or {}).get("port")
    new_port = new_parsed["args"]["port"]
    if reconfig is not None and reconfig.get("tier") == "noop":
        print(json.dumps({"reconfig": {**reconfig, "path": "noop", "restarted": False},
                          "pid": launcher_args.pid, "note": "nothing changed"}))
        sys.exit(0)
    live_values, live_reason = (None, "--reconfigure-restart given") if getattr(
        launcher_args, "reconfigure_restart", False) else (
        (None, "the port changes") if new_port != old_port else
        live_reconfig_plan(state, reconfig))
    can_apply_in_process = live_values is not None
    if launcher_args.zygote and not can_apply_in_process:
        zygote_reply = zygote.check(launcher_args.zygote, env, accept=_accept(launcher_args))
        if not zygote_reply.get("ok"):
            print(json.dumps({"zygote_findings": zygote_reply.get("findings"),
                              "error": zygote_reply.get("error"), "reconfig": reconfig,
                              "note": "not started; old engine left running"}))
            sys.exit(2)
        if launcher_args.zygote_check:
            print(json.dumps({"zygote_check": zygote_reply, "reconfig": reconfig,
                              "vllm_args": new_parsed}))
            sys.exit(0)
    elif launcher_args.zygote_check:
        print(json.dumps({"reconfig": reconfig, "vllm_args": new_parsed}))
        sys.exit(0)
    if new_port != old_port and port_in_use(new_port) and not launcher_args.accept_port_in_use:
        print(json.dumps({"finding": {
            "check": "port_in_use", "observed": f"127.0.0.1:{new_port} already has a listener",
            "expected": "a free port (or the replaced engine's own port)",
            "why": "the health check would read the other server's /health as this start's",
            "actions": ["keep the replaced engine's port", "pick a free port",
                        "--accept-port-in-use: start anyway"],
            "risk": "a start reported healthy that never ran"},
            "reconfig": reconfig, "note": "not started; old engine left running"}))
        sys.exit(2)
    if can_apply_in_process:
        report, live_reason = apply_in_process_reconfig(
            launcher_args, state, new_vargv, new_parsed, preflight_report, reconfig,
            live_values)
        if report is not None:
            return report
        if launcher_args.zygote:   # the restart below forks: check before stopping the engine
            zygote_reply = zygote.check(launcher_args.zygote, env, accept=_accept(launcher_args))
            if not zygote_reply.get("ok"):
                print(json.dumps({"zygote_findings": zygote_reply.get("findings"),
                                  "error": zygote_reply.get("error"), "reconfig": reconfig,
                                  "live_fallback_reason": live_reason,
                                  "note": "not started; old engine left running"}))
                sys.exit(2)
    t_stop = time.perf_counter()
    stop_info = stop_engine(launcher_args.pid, port=old_port)
    if not stop_info["stopped"]:
        print(json.dumps({"findings": [{
            "check": "stop_engine", "observed": stop_info,
            "expected": "every process of the old engine gone and its port closed",
            "why": "a new engine next to a live old one would share its GPUs and port",
            "actions": ["stop the listed pids yourself, then start the engine"]}],
            "reconfig": reconfig, "note": "not started; old engine may still be running"}))
        sys.exit(2)
    reconfig_report = {**(reconfig or {}), "path": "restart", "restarted": True,
                       "replaced_pid": launcher_args.pid,
                       "restart_reason": live_reason,
                       "stop_seconds": round(time.perf_counter() - t_stop, 6)}
    return _launch_process(
        launcher_args,
        new_vargv,
        new_parsed,
        env,
        preflight_report,
        extra_report={"reconfig": reconfig_report},
    )


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
        text = open(log, errors="replace").read() if log and os.path.exists(log) else ""
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
    p.add_argument("cmd", choices=["start", "reconfigure", "zygote"])
    p.add_argument("--pid", type=int, default=0,
                   help="reconfigure: pid of the running engine to reconfigure")
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
    p.add_argument("--weights-accept-reconfig", action="store_true",
                   help="reconfigure: proceed with ipc_cache although a Tier-2 WeightCacheKey "
                        "field changed (use after restarting weights.py for the new config)")
    p.add_argument("--reconfigure-restart", action="store_true",
                   help="reconfigure: force a warm worker restart even when the flag delta can "
                        "be applied in-process (in_process_instant / in_process_graph)")
    p.add_argument("--kvplan", nargs="?", const="", default=None, metavar="DIR",
                   help="reuse the KV plan of an earlier start with the same config (adds "
                        "--worker-cls kvplan.KVPlanWorker; plans in DIR, default "
                        "<cache-root>/kvplan; give DIR as --kvplan=DIR); off by default")
    p.add_argument("--fork", action="store_true",
                   help="fork the engine core and workers instead of spawning them")
    p.add_argument("--zygote", default="", help="start: fork from the zygote at this socket; "
                   "zygote: start one listening there (log: --log)")
    p.add_argument("--zygote-check", action="store_true",
                   help="start/reconfigure: only check whether the start/reconfigure would "
                        "pass preflight and exit (0 ok, 2 refused)")
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
    parsed_args, vargv = launcher_parser().parse_known_args(argv)
    return parsed_args, [token for token in vargv if token != "--"]


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["pool"]:
        # The standby pool has its own parser (holders, engines and a router of its own).
        import weight_scaling_colocate
        return weight_scaling_colocate.main(raw_argv[1:])
    parsed_args, vargv = parse(raw_argv)
    if parsed_args.cmd == "zygote":
        result_payload = start_zygote(parsed_args)
        print(json.dumps(result_payload))
        return 0 if result_payload.get("ready") else 1
    result_payload = (
        reconfigure(parsed_args, vargv, raw_argv)
        if parsed_args.cmd == "reconfigure"
        else start(parsed_args, vargv)
    )
    live = (result_payload.get("reconfig") or {}).get("path") == "in_process_live"
    # An in-process change starts nothing: --log is not this engine's fresh log (an old
    # traceback there would read as a failed start), so only /health is checked.
    result_payload["healthy_after_s"] = wait_healthy(
        result_payload["port"], result_payload["pid"], None if live else parsed_args.log,
        timeout=parsed_args.timeout, t0=result_payload["t_launch"],
    )
    if parsed_args.kvplan is not None and not live:
        result_payload["kvplan"] = kvplan_report(parsed_args.log)
    if (parsed_args.fork or parsed_args.zygote) and not live:
        result_payload["fork_check"] = fork_check(parsed_args.log)
        result_payload["fork_finding"] = fork_finding(result_payload["fork_check"])
    print(json.dumps(result_payload))
    return 0 if result_payload["healthy_after_s"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
