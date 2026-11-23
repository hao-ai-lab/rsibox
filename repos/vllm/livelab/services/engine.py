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
    `--max-num-seqs`, `--max-num-batched-tokens`, each at most its boot value, and `--scheduling-policy`). The launcher
    sends it to the engine core's live agent (started by engine.py in every engine it starts),
    which sets the scheduler's limits between two steps and reads them back. PID keeps
    running; only an acknowledged change updates the recorded state.
  - `restart`: everything else (`--max-model-len`, `--enable-prefix-caching`, `--enforce-eager`,
    CUDA graph flags, `--gpu-memory-utilization`, ...), and any live change the agent does not
    acknowledge: PID is stopped (verified) and the engine starts with the new argv; the
    report's `restart_reason` says why it was not live. With `--weights`, resident CUDA IPC
    weights make the restart warm (zero disk reload).
  Tier-2 deltas (`WeightCacheKey` sharding, `subtier='tier2a_reshard'`, or weight format,
  `'tier2b_weight_format'`) are refused before stopping PID unless `--weights-accept-reconfig`,
  with one exception, `path: reshard`: on an engine with resident weights (ipc_cache), a delta
  whose only weight-key change is the MoE expert sharding (`--enable-expert-parallel`,
  `--expert-placement-strategy`: TP <-> EP on the same GPUs) is made live. Before PID stops,
  `reshard_live.py ensure` makes the target engine's holder group exist (socket folder
  `reshard_live.group_dir`: `<dir>.ep` for EP, `<dir>` without; a group already there with the
  target flags is reused, e.g. the way back) or starts it without reading the checkpoint
  (`--load-format dummy`) and fills it GPU to GPU from the running engine's group. Only then is
  PID stopped and the engine restarted on the target group; any failure is a finding (exit 2)
  with PID running and the half-started group stopped. The source group is left resident (the
  way back costs no reshard).
  Every preflight check runs BEFORE mutating or stopping PID; a refusal exits 2 and leaves PID
  running untouched.
- Stopping PID (stop_engine) is drain-aware: SIGTERM goes to the API server only, which aborts
  its in-flight requests (answering them), stops its engine core and workers, and closes the
  port; only what it leaves behind is signalled by the launcher, and SIGKILL comes after the
  grace. The report's `reconfig.stop` has the phases (leader exit, tree exit, port closed).
  The engine's state file records the API server's start time, so a stop after it died still
  finds the members of its session, zygote-forked ones included, and never a reused pid.
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
import argparse, contextlib, json, os, signal, subprocess, sys, time, urllib.request, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corelab  # noqa: F401,E402  (core/livelab on sys.path)
import kvplan, live_reconfig, vllm_args, weights, zygote  # noqa: E402
import forksafe  # noqa: E402
from rsibox import harness, ipc, proc  # noqa: E402  (stdlib only)
from rsibox.warm import (  # noqa: E402  (stdlib only)
    DynamoGmsStore,
    GMS_VLLM_LOAD_FORMAT,
    GMS_VLLM_WORKER_CLS,
)

_finding = harness.plain_finding


def _exit_json(payload, code=2):
    print(json.dumps(payload))
    sys.exit(code)


def _port_in_use_finding(port, *, expected="a free port", why="the health check would read the other server's /health as this start's; the new server then fails to bind", actions=None):
    return _finding(
        "port_in_use", f"127.0.0.1:{port} already has a listener", expected,
        why=why,
        actions=actions or ["pick a free port (cost: none)",
                            "--accept-port-in-use: start anyway (the health result is then not about this engine)"],
        risk="a start reported healthy that never ran",
    )

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


def _apply_gms_env(a, env):
    """Apply Dynamo GMS environment variables without shadowing LIVE_SITE / farm_dir."""
    store = gms_store(a)
    env["GMS_SOCKET_DIR"] = store.socket_dir
    if store.scratch_kv:
        env["DYN_GMS_SCRATCH_KV_ENABLED"] = "1"
    if store.extra_pythonpath:
        cur = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        for entry in store.extra_pythonpath:
            if entry not in cur:
                cur.append(entry)
        env["PYTHONPATH"] = os.pathsep.join(cur)


def engine_env(a, base=None):
    """The environment every engine process runs with, and the zygote's (start_zygote)."""
    env = dict(os.environ if base is None else base)
    env.update(cache_env(a.cache_root))
    # The live reconfiguration agent (live_reconfig.py) in the engine core: its sitecustomize
    # goes on the path ahead of site-packages, and each launch gets its own control folder.
    if not getattr(a, "live_dir", None):
        a.live_dir = str(Path(a.cache_root).resolve() / "engines" / "live" / uuid.uuid4().hex)
    env["RSIBOX_LIVE_RECONFIG_DIR"] = a.live_dir
    paths = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and p != LIVE_SITE]
    env["PYTHONPATH"] = os.pathsep.join([LIVE_SITE] + paths)
    if a.kvplan is not None:
        # KVPlanWorker is imported by name in every worker: this folder goes on the path.
        here = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join([here] + ([env["PYTHONPATH"]]
                                                      if env.get("PYTHONPATH") else []))
        env["RSIBOX_KVPLAN_DIR"] = kvplan_dir(a)
    explicit_backend = getattr(a, "weights_backend", "") or ""
    if explicit_backend == "dynamo_gms" or (
        not explicit_backend
        and not getattr(a, "weights", False)
        and (
            getattr(a, "gms_socket_dir", "")
            or getattr(a, "gms_scratch_kv", False)
            or getattr(a, "gms_pythonpath", "")
        )
    ):
        _apply_gms_env(a, env)
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
        print(json.dumps({"finding": _finding(
            "cute_dsl_arch", why, "one compute capability",
            why="CUTE_DSL_ARCH stays unset, so CuTe DSL probes the GPU with cuInit and forked workers cannot use CUDA",
            actions=["set CUTE_DSL_ARCH yourself", "start without --fork or --zygote"],
        )}), file=sys.stderr)
        return None
    major, minor = (int(x) for x in caps.pop().split("."))
    return forksafe.cute_dsl_arch((major, minor))


port_in_use = proc.port_in_use


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


def weights_backend(a, v=None):
    """Resolved resident-weights backend: 'dynamo_gms', 'ipc_cache', or 'none'."""
    explicit_backend = getattr(a, "weights_backend", "") or ""
    if explicit_backend:
        return explicit_backend
    if getattr(a, "weights", False):
        return "ipc_cache"
    if v and v.get("ok"):
        lf = (v.get("args") or {}).get("load_format")
        if lf == GMS_VLLM_LOAD_FORMAT:
            return "dynamo_gms"
        if lf == "ipc_cache":
            return "ipc_cache"
    return "none"


def gms_store(a):
    """Build the `DynamoGmsStore` configured by launcher flags."""
    sock_dir = getattr(a, "gms_socket_dir", "") or getattr(a, "weights_socket_dir", "") or None
    extra_py = [p for p in (getattr(a, "gms_pythonpath", "") or "").split(os.pathsep) if p]
    return DynamoGmsStore(
        socket_dir=sock_dir,
        read_only=bool(getattr(a, "gms_read_only", False)),
        scratch_kv=bool(getattr(a, "gms_scratch_kv", False)),
        extra_pythonpath=extra_py,
    )


def load_format(a, v):
    """The load format the server will use: the argv's if explicit, else the selected
    weights backend's format (`ipc_cache` or `gms`), else vLLM's default."""
    if "load_format" not in v["explicit"]:
        backend = weights_backend(a, v)
        if backend == "ipc_cache":
            return "ipc_cache"
        if backend == "dynamo_gms":
            return GMS_VLLM_LOAD_FORMAT
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


def value_finding(parsed):
    """A `vllm_value` finding when a live scheduler field (live_reconfig.LIVE_FIELDS) has a
    value no engine runs with, else None."""
    args = parsed.get("args") or {}
    values = {k: args[k] for k in live_reconfig.LIVE_FIELDS if args.get(k) is not None}
    problems = live_reconfig.check_values(values, {})
    if not problems:
        return None
    return _finding("vllm_value", problems, "valid scheduler limits and policy",
                    why="the restart would start a server that cannot run, after stopping this one",
                    actions=["correct the reported scheduler value"])


def argv_findings(a, v):
    """Findings about the parsed `vllm serve` argv that stop the start."""
    if not v["ok"]:
        if v.get("findings"):
            return v["findings"]
        if v.get("zygote_error"):
            return [_finding(
                "zygote_parse", v.get("error"),
                "the zygote's parse child reads the argv with vLLM's parser",
                why="the child failed before vLLM's parser ran (an edited module that does not import, or a refresh that failed); a forked server would fail the same way",
                actions=["fix the edited module (the error above)", "start without --zygote to see the full traceback", "restart the zygote"],
            )]
        return [_finding("vllm_args", v.get("error"), "an argv that `vllm serve` accepts",
                         why="the server would exit on it", actions=["fix the vLLM flags (vLLM's message above)"])]
    args, explicit, found = v["args"], v["explicit"], []
    if "model" not in explicit:
        found.append(_finding(
            "vllm_model", f"no model in the argv; vLLM's default is {args['model']}",
            "the model, positional or --model",
            why="the server would load vLLM's default model, not the one meant (a flag that takes an optional value can swallow a positional model)",
            actions=["pass the model as the first vLLM argument"],
        ))
    if "host" in explicit and args["host"] != "127.0.0.1":
        found.append(_finding("vllm_host", f"--host {args['host']}", "--host 127.0.0.1 (the launcher's)",
                              why="lab engines bind 127.0.0.1 only", actions=["drop --host"]))
    want_ipc = bool(a.weights or getattr(a, "weights_backend", "") == "ipc_cache")
    if want_ipc and "load_format" in explicit and args["load_format"] != "ipc_cache":
        found.append(_finding(
            "weights_load_format", f"--weights and the argv's --load-format {args['load_format']}",
            "--load-format ipc_cache, or none (--weights adds it)",
            why="--weights maps the daemon's weights over IPC; the argv asks vLLM to load them another way",
            actions=["drop --load-format", "drop --weights"],
        ))
    cfg = explicit_loader_config(v)
    if want_ipc and cfg is not None and cfg.get("fallback", True) is not False:
        fb = json.dumps(cfg["fallback"]) if "fallback" in cfg else "unset (vLLM's default: true)"
        found.append(_finding(
            "weights_fallback", f"--weights and the argv's --model-loader-extra-config with fallback {fb}",
            '"fallback": false in that config, or no config (--weights adds one)',
            why="with the fallback on, an engine that cannot map the daemon's weights reads the checkpoint from disk silently; --weights promises resident weights or a failed load",
            actions=['add "fallback": false to the config', "drop --model-loader-extra-config", "drop --weights"],
        ))
    if (a.weights_socket_dir and cfg is not None and load_format(a, v) == "ipc_cache"
            and os.path.abspath(a.weights_socket_dir) != os.path.abspath(socket_dir(a, v))):
        found.append(_finding(
            "weights_socket_dir",
            f"--weights-socket-dir {a.weights_socket_dir} and the argv's loader config, whose socket folder is {socket_dir(a, v)}",
            "one folder",
            why="vLLM dials the config's folder; the daemon probe would check the other one",
            actions=["drop --weights-socket-dir", "set socket_dir in the config to the same folder"],
        ))
    if getattr(a, "weights", False) and getattr(a, "weights_backend", "") in ("none", "dynamo_gms"):
        found.append(_finding(
            "weights_backend_conflict",
            f"--weights (ipc_cache) and --weights-backend {a.weights_backend}",
            "one resident weights backend",
            why="--weights selects vLLM's ipc_cache daemon; it conflicts with another --weights-backend",
            actions=["drop --weights", f"or drop --weights-backend {a.weights_backend}"],
        ))
    backend = weights_backend(a, v)
    if backend == "dynamo_gms":
        delta = gms_store(a).launch_delta(
            engine="vllm",
            explicit_flags=set(explicit),
            explicit_values=args,
        )
        found.extend(delta.findings)
        if a.kvplan is not None:
            found.append(_finding(
                "kvplan_gms_worker_cls",
                f"--kvplan ({KVPLAN_WORKER}) and --weights-backend dynamo_gms ({GMS_VLLM_WORKER_CLS})",
                "either --kvplan or --weights-backend dynamo_gms (both set --worker-cls)",
                why="vLLM accepts a single --worker-cls; Dynamo GMS requires GMSWorker while --kvplan uses KVPlanWorker "
                    "(note: RSIBox live editing uses --worker-extension-cls and composes cleanly with GMSWorker)",
                actions=["drop --kvplan when using --weights-backend dynamo_gms", "or use --weights-backend ipc_cache"],
            ))
    if a.kvplan is not None and "worker_cls" in explicit and args["worker_cls"] != KVPLAN_WORKER:
        found.append(_finding(
            "kvplan_worker_cls", f"--kvplan and the argv's --worker-cls {args['worker_cls']}",
            f"--worker-cls {KVPLAN_WORKER}, or none (--kvplan adds it)",
            why="the KV plan lives in that worker class; with another one every start profiles",
            actions=["drop --worker-cls", "drop --kvplan"], risk="none for correctness; startup not faster",
        ))
    return found


def reconfig_findings(launcher_args, parsed_vllm, reconfig):
    """Findings when a `reconfigure` with resident weights (`ipc_cache` or `dynamo_gms`) changes a Tier-2 WeightCacheKey field."""
    backend = weights_backend(launcher_args, parsed_vllm)
    if (not reconfig or reconfig["weights_compatible"]
            or backend not in ("ipc_cache", "dynamo_gms")
            or getattr(launcher_args, "weights_accept_reconfig", False)):
        return []
    diff = {k: reconfig["changed"][k] for k in reconfig["weight_key_changed"]}
    if backend == "dynamo_gms":
        return [_finding(
            "weights_reconfig_tier2",
            f"{reconfig['tier']} change on WeightCacheKey fields: {diff}",
            "Tier 1 engine flags only (max_model_len, enforce_eager, compilation_config, "
            "max_num_batched_tokens, max_num_seqs, gpu_memory_utilization, enable_prefix_caching, ...), "
            "or a fresh Dynamo GMS server matching the new model/sharding/dtype",
            why="Dynamo GMS holds committed GPU allocations for the original model/sharding/dtype layout; "
                "relaunching against mismatched resident weights fails or corrupts layout validation after the old engine stops",
            actions=["restart the GMS server(s) for the new model/sharding/dtype, then pass --weights-accept-reconfig",
                     "pass --weights-backend none to load from disk",
                     "keep WeightCacheKey flags unchanged for zero-disk GMS reconfiguration"],
            risk="stopping the running engine for a dynamo_gms start that fails at load_model",
        )]
    return [_finding(
        "weights_reconfig_tier2",
        f"{reconfig['tier']} change on WeightCacheKey fields: {diff}",
        "Tier 1 engine flags only (max_model_len, enforce_eager, compilation_config, "
        "max_num_batched_tokens, max_num_seqs, gpu_memory_utilization, enable_prefix_caching, ...), "
        "or a restarted weights.py daemon matching the new WeightCacheKey",
        why="weights.py holds post-processed, rank-sharded GPU tensors keyed by WeightCacheKey; "
            "IpcModelLoader rejects a mismatched WeightCacheKey with CacheConfigMismatchError after the old engine is stopped",
        actions=["restart weights.py with the new sharding/format flags, then pass --weights-accept-reconfig",
                 "drop --weights / --load-format ipc_cache to load from disk",
                 "keep WeightCacheKey flags unchanged for zero-copy IPC reconfiguration"],
        risk="stopping the running engine for an ipc_cache start that fails at load_model",
    )]


def serve_cmd(a, vargv, v):
    """The `vllm serve` argv: the caller's, verbatim, then what the launcher and the holders
    need that the caller did not set (host, load format and loader config, worker class).
    Flags go after the caller's: `vllm serve` wants the model first when --served-model-name
    is given."""
    cmd, explicit = [a.vllm, "serve", *vargv], v["explicit"]
    if "host" not in explicit:
        cmd += ["--host", "127.0.0.1"]
    backend = weights_backend(a, v)
    if backend == "dynamo_gms":
        delta = gms_store(a).launch_delta(
            engine="vllm",
            explicit_flags=set(explicit),
            explicit_values=v["args"],
        )
        cmd += list(delta.args)
    else:
        if (a.weights or backend == "ipc_cache") and "load_format" not in explicit:
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
    backend = weights_backend(a, v)
    if backend == "dynamo_gms":
        pr = gms_store(a).probe(
            visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
            tp=v["args"]["tensor_parallel_size"],
        )
        rep["gms_probe"] = pr.to_dict()
        if not pr.ok and not a.weights_accept_missing:
            found.append(pr.finding)
    elif load_format(a, v) == "ipc_cache":
        pr = weights.probe(socket_dir(a, v), env.get("CUDA_VISIBLE_DEVICES"),
                           v["args"]["tensor_parallel_size"])
        rep["weights_probe"] = pr
        if not pr["ok"] and not a.weights_accept_missing:
            found.append(weights.daemon_finding(pr))
    return found, rep


def kvplan_report(log):
    """The KV plan's log lines (one per rank per start): counts, seconds and findings."""
    rep = {"reused": 0, "measured": 0, "profiled": 0, "seconds": [], "findings": []}
    with open(log, errors="replace") as fh:
        for line in fh:
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
    record = {
        "pid": pid,
        "start_time": _start_time(pid),     # tells this process from a later one with its pid
        "port": port,
        "vargv": list(vllm_argv),
        "cmd": list(command),
        "vllm_args": parsed_vllm,
        "live_dir": getattr(launcher_args, "live_dir", None),
        "log": os.path.abspath(launcher_args.log),
        "launcher": {
            "vllm": launcher_args.vllm,
            "weights": bool(launcher_args.weights),
            "weights_backend": getattr(launcher_args, "weights_backend", ""),
            "weights_socket_dir": launcher_args.weights_socket_dir,
            "gms_socket_dir": getattr(launcher_args, "gms_socket_dir", ""),
            "gms_read_only": bool(getattr(launcher_args, "gms_read_only", False)),
            "gms_scratch_kv": bool(getattr(launcher_args, "gms_scratch_kv", False)),
            "gms_pythonpath": getattr(launcher_args, "gms_pythonpath", ""),
            "kvplan": launcher_args.kvplan,
            "fork": bool(launcher_args.fork),
            "zygote": launcher_args.zygote,
        },
    }
    ipc.atomic_write_json(state_path, record)


_INJECTED_FLAGS = (
    ("--host", "127.0.0.1"),
    ("--load-format", "ipc_cache"),
    ("--load-format", GMS_VLLM_LOAD_FORMAT),
    ("--model-loader-extra-config", None),
    ("--worker-cls", KVPLAN_WORKER),
    ("--worker-cls", GMS_VLLM_WORKER_CLS),
)


def _strip_injected_flags(argv):
    """Strip launcher-appended holder flags from a raw `/proc/<pid>/cmdline` `vllm serve` tail."""
    cleaned, i = [], 0
    while i < len(argv):
        if i + 1 < len(argv) and any(argv[i] == f and (v is None or argv[i + 1] == v) for f, v in _INJECTED_FLAGS):
            i += 2
        else:
            cleaned.append(argv[i])
            i += 1
    return cleaned


def _flag_value(tokens, flag):
    try:
        idx = tokens.index(flag)
    except ValueError:
        return None
    return tokens[idx + 1] if idx + 1 < len(tokens) else None


def _proc_environ(pid, proc_root="/proc"):
    """Best-effort parse of `/proc/<pid>/environ` into a str->str dict."""
    out = {}
    try:
        raw = Path(f"{proc_root}/{pid}/environ").read_bytes()
    except OSError:
        return out
    for item in raw.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def load_engine_state(cache_root, pid, proc_root="/proc"):
    """Load engine `pid` state from `<cache-root>/engines/<pid>.json`, falling back to `/proc/<pid>/cmdline`."""
    if (state := ipc.read_json(_engine_state_path(cache_root, pid))) is not None:
        return state
    try:
        raw_tokens = [t for t in Path(f"{proc_root}/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0") if t]
    except OSError:
        return None
    if "serve" not in raw_tokens:
        return None
    idx = raw_tokens.index("serve")
    tail = raw_tokens[idx + 1:]
    cfg_dict = {}
    if (raw_cfg := _flag_value(tail, "--model-loader-extra-config")):
        with contextlib.suppress(Exception):
            parsed_cfg = json.loads(raw_cfg)
            if isinstance(parsed_cfg, dict):
                cfg_dict = parsed_cfg
    lf = _flag_value(tail, "--load-format")
    wcls = _flag_value(tail, "--worker-cls")
    is_gms = lf == GMS_VLLM_LOAD_FORMAT or wcls == GMS_VLLM_WORKER_CLS
    is_ipc = lf == "ipc_cache"
    penv = _proc_environ(pid, proc_root=proc_root) if is_gms else {}
    scratch_raw = (penv.get("DYN_GMS_SCRATCH_KV_ENABLED") or "").strip().lower()
    return {
        "pid": pid, "port": None, "vargv": _strip_injected_flags(tail),
        "cmd": raw_tokens[idx - 1:] if idx >= 1 else raw_tokens, "vllm_args": None,
        "launcher": {
            "vllm": raw_tokens[idx - 1] if idx >= 1 else "vllm",
            "weights": is_ipc,
            "weights_backend": "dynamo_gms" if is_gms else ("ipc_cache" if is_ipc else ""),
            "weights_socket_dir": cfg_dict.get("socket_dir") or "",
            "gms_socket_dir": penv.get("GMS_SOCKET_DIR", "") if is_gms else "",
            "gms_read_only": bool(cfg_dict.get("gms_read_only", False)) if is_gms else False,
            "gms_scratch_kv": scratch_raw in ("1", "true", "yes") if is_gms else False,
            "gms_pythonpath": "",
            "kvplan": "" if wcls == KVPLAN_WORKER else None,
            "fork": False, "zygote": "",
        },
    }


def _stat(pid, proc_root="/proc"):
    """(ppid, pgid, sid, start time) from /proc/<pid>/stat, or None when pid is gone. The start
    time (clock ticks since boot, field 22) tells a reused pid from the process first seen."""
    try:
        with open(f"{proc_root}/{pid}/stat") as stat_file:
            fields = stat_file.read().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid, proc_root="/proc"):
    """The processes below pid (ppid links), plus every process in pid's process group or
    session and theirs: a worker whose parent died is reparented (ppid 1) but stays in the
    engine's session. A pid equal to a live group or session id is never reused."""
    children_by_parent, descendants, members = {}, [], []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return descendants
    for entry in entries:
        if entry.isdigit():
            st = _stat(entry, proc_root)
            if st is None:
                continue
            parent_pid, group_id, session_id, _ = st
            children_by_parent.setdefault(parent_pid, []).append(int(entry))
            if int(entry) != pid and pid in (group_id, session_id):
                members.append(int(entry))
    stack = [pid] + members
    seen = {pid}
    while stack:
        current = stack.pop()
        if current != pid and current not in seen:
            seen.add(current)
            descendants.append(current)
        for child_pid in children_by_parent.get(current, []):
            if child_pid not in seen:
                stack.append(child_pid)
    return descendants


MARKER_ENV = zygote.MARKER_ENV   # engine_env(): one folder per launch, inherited


def _marker_member_verified(member_pid, marker, proc_root="/proc"):
    """True when `<marker>/.members/<member_pid>` recorded this process's exact start time
    (covers zygote-forked trees whose `/proc/<pid>/environ` stays the zygote's)."""
    if not marker:
        return False
    member_file = os.path.join(marker, zygote.MARKER_MEMBERS_DIR, str(member_pid))
    try:
        with open(member_file, encoding="utf-8") as fh:
            recorded_start = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    st = _stat(member_pid, proc_root)
    return st is not None and st[3] == recorded_start


def verified_members(pids, marker, proc_root="/proc", session=None, since=None):
    """The pids that belong to one launch, never this process: those whose exec-time
    environment has MARKER_ENV == marker, those recorded under `<marker>/.members/<pid>` with
    matching `/proc/<pid>/stat` start time (set on fork by `zygote._child`), and, when the
    launch recorded its leader's start time `since`, the members of the leader's session
    `session` that started at or after it. Empty when none of those ties a member to the
    engine."""
    want = f"{MARKER_ENV}={marker}".encode() if marker else None
    verified = []
    for member_pid in pids:
        if member_pid == os.getpid():
            continue
        if want is not None:
            try:
                with open(f"{proc_root}/{member_pid}/environ", "rb") as environ_file:
                    if want in environ_file.read().split(b"\0"):
                        verified.append(member_pid)
                        continue
            except OSError:
                pass
            if _marker_member_verified(member_pid, marker, proc_root=proc_root):
                verified.append(member_pid)
                continue
        if session is not None and since is not None:
            st = _stat(member_pid, proc_root)
            if st is not None and st[2] == session and st[3] >= since:
                verified.append(member_pid)
    return verified


def _start_time(pid):
    """The process start time (clock ticks since boot, /proc/<pid>/stat field 22), or None when
    pid is gone. A pid with another start time is another process (pid reuse)."""
    st = _stat(pid)
    return None if st is None else st[3]


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


def stop_engine(pid, port=None, grace=10.0, kill_wait=5.0, marker=None, start_time=None):
    """Stop engine `pid` (the `vllm serve` API server, leader of its session) and every process
    below it (_descendants: its process tree, group and session), drain-aware: SIGTERM goes to
    the leader only, and vLLM's own shutdown stops the rest in order (the API server aborts
    its in-flight requests, answering them, stops the engine core and its workers, then closes
    the port). Signalling every process at once, as before, races that order: the engine core
    and workers died before the aborts reached the API server, whose request handlers then
    waited forever, so uvicorn never closed its connections and only the SIGKILL after
    `grace` ended it (10.1 s under client load on B200, against 1.0 s leader-only).
    - When the leader exits and leaves members behind, those get SIGTERM at once (nothing
      will stop them now); anything alive after `grace` s gets SIGKILL, up to `kill_wait` s.
    - When `pid` exited before the stop, or another process has it now (its start time is not
      `start_time`, the leader's recorded at launch), only verified members are stopped
      (verified_members: the launch `marker`, or the leader's session and start time). Without
      either, nothing is signalled, and the report says so.
    Each process is identified by (pid, start time) when first seen and is signalled only
    while it still has that start time, so a reused pid is never hit.
    Returns {"stopped", "seconds", "alive", "port_open", "how", "escalated", "tree",
    "leader_exit_s", "tree_exit_s", "port_closed_s"}: stopped is True only when every recorded
    process is gone (or a zombie) and `port` no longer answers; it is never assumed."""
    t0 = time.time()
    below = [member_pid for member_pid in _descendants(pid) if member_pid != os.getpid()]
    leader_start = _start_time(pid) if not zygote._exited(pid) else None
    reused = leader_start is not None and start_time is not None and leader_start != start_time
    if leader_start is None or reused:
        verified = verified_members(below, marker, session=None if reused else pid,
                                    since=start_time)
        if not verified:
            note = (f"pid {pid} is another process now (start time {leader_start}, recorded "
                    f"{start_time})" if reused else
                    f"pid {pid} was not running" if not below else
                    f"pid {pid} exited; {len(below)} member(s) of its group or session not "
                    "verified as this engine's (no launch marker or start time): none signalled")
            return {"stopped": False, "seconds": 0.0, "alive": [], "port_open": None,
                    "note": note, "unverified": below}
        below, leader = verified, None
    else:
        leader = pid
    procs = {proc_pid: _start_time(proc_pid) for proc_pid in ([leader] if leader else []) + below}
    procs = {p: st for p, st in procs.items() if st is not None}
    marks = {"leader_exit_s": None, "tree_exit_s": None, "port_closed_s": None}

    def alive():
        return sorted(p for p, st in procs.items() if _same_process(p, st) and not zygote._exited(p))

    def done():
        now, left = round(time.time() - t0, 3), alive()
        if leader and marks["leader_exit_s"] is None and leader not in left:
            marks["leader_exit_s"] = now
        if marks["tree_exit_s"] is None and not left:
            marks["tree_exit_s"] = now
        if port is not None and marks["port_closed_s"] is None and not port_in_use(port):
            marks["port_closed_s"] = now
        return not left and (port is None or marks["port_closed_s"] is not None)

    orphans_signalled = False
    if leader:
        _signal_if_same(leader, procs.get(leader), signal.SIGTERM)
    else:                        # no API server left to stop them in order
        for p, st in procs.items():
            _signal_if_same(p, st, signal.SIGTERM)
        orphans_signalled = True
    deadline = t0 + grace
    while time.time() < deadline and not done():
        if not orphans_signalled and marks["leader_exit_s"] is not None:
            for p in alive():    # the leader exited without them
                _signal_if_same(p, procs[p], signal.SIGTERM)
            orphans_signalled = True
        time.sleep(0.02)
    escalated = not done()
    if escalated:
        for p in alive():
            _signal_if_same(p, procs[p], signal.SIGKILL)
        deadline = time.time() + kill_wait
        while time.time() < deadline and not done():
            time.sleep(0.02)
    left = alive()
    port_open = port_in_use(port) if port is not None else None
    return {"stopped": not left and not port_open, "seconds": round(time.time() - t0, 3),
            "alive": left, "port_open": port_open, "how": "leader" if leader else "members",
            "escalated": escalated, "tree": len(procs), **marks}


def _launch_process(launcher_args, vllm_argv, parsed_vllm, env, preflight_report, extra_report=None):
    """Spawn or fork the validated `vllm serve` command and record its state."""
    command, port = serve_cmd(launcher_args, vllm_argv, parsed_vllm), parsed_vllm["args"]["port"]
    report = {"port": port, "vllm_args": parsed_vllm, **preflight_report, **(extra_report or {})}
    t_launch = time.time()
    if launcher_args.zygote:
        Path(launcher_args.log).parent.mkdir(parents=True, exist_ok=True)
        open(launcher_args.log, "w").close()
        reply = zygote.request(launcher_args.zygote, command, env, os.path.abspath(launcher_args.log),
                               accept=_accept(launcher_args))
        if not reply.get("ok"):
            _exit_json({
                "zygote_findings": reply.get("findings"), "error": reply.get("error"),
                **({"reconfig": extra_report["reconfig"]} if extra_report and "reconfig" in extra_report else {}),
                "note": "not started; act on the findings or pass --zygote-accept",
            })
        result = {
            "pid": reply["pid"], "cmd": command, "t_launch": t_launch, **report,
            "zygote": launcher_args.zygote, "zygote_check_seconds": reply.get("check_seconds"),
            "zygote_accepted_findings": reply.get("accepted_findings", []),
            **({"zygote_refreshed": reply["refreshed"]} if reply.get("refreshed") else {}),
        }
        save_engine_state(launcher_args, result["pid"], port, vllm_argv, command, parsed_vllm)
        return result
    p = corelab.spawn_logged(command, launcher_args.log, env=env)
    save_engine_state(launcher_args, p.pid, port, vllm_argv, command, parsed_vllm)
    return {"pid": p.pid, "cmd": command, "t_launch": t_launch, **report}


def start(a, vargv):
    """Launch the server; return {"pid", "cmd", "t_launch", "port", "vllm_args", ...} or exit 2."""
    env = engine_env(a)
    if a.zygote and a.zygote_check:
        reply = zygote.check(a.zygote, env, accept=_accept(a))
        _exit_json({"zygote_check": reply}, code=0 if reply.get("ok") else 2)
    v = read_vllm_args(a, vargv, env)
    if weights_backend(a, v) == "dynamo_gms":
        _apply_gms_env(a, env)
    found, pre = argv_findings(a, v), {}
    if not found:
        found, pre = preflight(a, v, env)
    if found:
        _exit_json({"findings": found, **pre, "vllm_args": v, "note": "not started"})
    port = v["args"]["port"]
    if port_in_use(port) and not a.accept_port_in_use:
        _exit_json({"finding": _port_in_use_finding(port), "note": "not started"})
    return _launch_process(a, vargv, v, env, pre)


_INHERIT_FLAGS = (
    ("--vllm", "vllm", False),
    ("--weights-socket-dir", "weights_socket_dir", False),
    ("--fork", "fork", True),
    ("--zygote", "zygote", False),
)

_GMS_INHERIT_FLAGS = (
    ("--gms-socket-dir", "gms_socket_dir", False),
    ("--gms-read-only", "gms_read_only", True),
    ("--gms-scratch-kv", "gms_scratch_kv", True),
    ("--gms-pythonpath", "gms_pythonpath", False),
)


def _has_cli_flag(tokens, flag):
    return any(tok == flag or tok.startswith(flag + "=") for tok in tokens)


def _inherit_launcher(launcher_args, saved_launcher, raw_argv):
    """Inherit launcher flags from the running engine's recorded state when not passed on CLI."""
    if not saved_launcher:
        return
    tokens = raw_argv[:raw_argv.index("--")] if "--" in raw_argv else list(raw_argv)
    explicit_wb = _has_cli_flag(tokens, "--weights-backend")
    explicit_w = _has_cli_flag(tokens, "--weights")
    if not explicit_wb and not explicit_w:
        if saved_launcher.get("weights"):
            launcher_args.weights = True
        if saved_launcher.get("weights_backend"):
            launcher_args.weights_backend = saved_launcher["weights_backend"]
    for flag, attr, is_bool in _INHERIT_FLAGS:
        if not _has_cli_flag(tokens, flag) and saved_launcher.get(attr):
            setattr(launcher_args, attr, True if is_bool else saved_launcher[attr])
    resolved_wb = weights_backend(launcher_args)
    if resolved_wb == "dynamo_gms" or (resolved_wb == "none" and not explicit_wb and not explicit_w):
        for flag, attr, is_bool in _GMS_INHERIT_FLAGS:
            if not _has_cli_flag(tokens, flag) and saved_launcher.get(attr):
                setattr(launcher_args, attr, True if is_bool else saved_launcher[attr])
    if (not _has_cli_flag(tokens, "--kvplan")
            and saved_launcher.get("kvplan") is not None
            and weights_backend(launcher_args) != "dynamo_gms"):
        launcher_args.kvplan = saved_launcher["kvplan"]


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
    if "scheduling_policy" in values:
        confirmed = confirmed and after.get("scheduling_policy") == values["scheduling_policy"]
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


def reshard_route(launcher_args, new_parsed, reconfig):
    """True when this reconfiguration goes through a holder-to-holder reshard: resident weights
    (ipc_cache), the weight key changes only in MoE expert sharding, and no
    --weights-accept-reconfig (which means: the caller restarted the daemons itself)."""
    import reshard_live     # no torch, no vLLM at import
    return bool(reconfig and not reconfig["weights_compatible"]
                and new_parsed.get("ok")
                and load_format(launcher_args, new_parsed) == "ipc_cache"
                and explicit_loader_config(new_parsed) is None
                and not getattr(launcher_args, "weights_accept_reconfig", False)
                and reshard_live.reshardable(reconfig))


def vllm_python(launcher_args):
    """The python next to --vllm (the venv the engine runs in), else this one."""
    candidate = Path(launcher_args.vllm).with_name("python")
    return str(candidate) if candidate.exists() else sys.executable


def reshard_target(launcher_args, state, old_vargv, new_vargv, old_parsed, new_parsed, env,
                   reconfig):
    """Run `reshard_live.py ensure` for the target engine (see the module contract); on success
    point the launcher at the target group and return its report, else exit 2 with PID up."""
    import reshard_live
    src_dir = socket_dir(launcher_args, old_parsed)
    dst_dir = reshard_live.group_dir(src_dir, new_parsed["args"])
    cmd = [vllm_python(launcher_args), str(Path(__file__).resolve().parent / "reshard_live.py"),
           "ensure", "--src-socket-dir", src_dir, "--src-vargv", json.dumps(old_vargv),
           "--src-log", state.get("log") or "", "--dst-socket-dir", dst_dir,
           "--dst-vargv", json.dumps(new_vargv), "--vllm", launcher_args.vllm,
           "--state-dir", str(Path(launcher_args.cache_root).resolve() / "holders")]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=launcher_args.timeout)
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, err, code = e.stdout or "", f"timeout after {launcher_args.timeout} s", -1
    report = corelab.last_json_line(out)
    report["seconds"] = round(time.perf_counter() - t0, 3)
    if code != 0 or not report.get("target"):
        _exit_json({
            "findings": report.get("findings") or [_finding(
                "weights_reshard", f"reshard_live.py ensure exited {code}: {(err or '')[-1500:]}",
                f"the target group at {dst_dir} holding the engine's weights",
                why="the reconfigured engine maps that group; without it nothing can start",
                actions=["read the report and the group log", "--weights-accept-reconfig after starting the target group yourself"],
            )],
            "reshard": report, "reconfig": reconfig, "note": "not started; old engine left running",
        })
    launcher_args.weights_socket_dir = dst_dir
    return report


def _check_zygote_for_reconfig(launcher_args, env, reconfig, **extra):
    reply = zygote.check(launcher_args.zygote, env, accept=_accept(launcher_args))
    if not reply.get("ok"):
        _exit_json({"zygote_findings": reply.get("findings"), "error": reply.get("error"),
                    "reconfig": reconfig, **extra, "note": "not started; old engine left running"})
    return reply


def reconfigure(launcher_args, patch_vargv, raw_argv=()):
    """Reconfigure a running engine (`--pid PID`) with `patch_vargv` overrides (preflight first,
    then live in-process apply, holder-to-holder reshard, or verified stop + restart)."""
    if not launcher_args.pid or zygote._exited(launcher_args.pid):
        _exit_json({"findings": [_finding(
            "reconfigure_pid",
            f"pid {launcher_args.pid} is not running" if launcher_args.pid else "--pid not given",
            "the pid of a running `vllm serve` started by engine.py",
            why="reconfigure reads the running engine's flags and replaces it only after preflight checks pass",
            actions=["pass --pid <running engine pid>", "use `engine.py start` instead"],
        )], "note": "not started"})
    state = load_engine_state(launcher_args.cache_root, launcher_args.pid)
    if state is None:
        _exit_json({"findings": [_finding(
            "reconfigure_pid", f"cannot read `vllm serve` state for pid {launcher_args.pid}",
            f"state file under {launcher_args.cache_root}/engines/{launcher_args.pid}.json or `vllm serve` in /proc/{launcher_args.pid}/cmdline",
            why="reconfigure needs the running engine's current flags to apply overrides",
            actions=["use the same --cache-root as `engine.py start`", "use `engine.py start` with the full argv instead"],
        )], "note": "not started"})
    _inherit_launcher(launcher_args, state.get("launcher"), raw_argv)
    env, old_vargv = engine_env(launcher_args), state["vargv"]
    new_vargv = vllm_args.override(old_vargv, patch_vargv)
    old_parsed = state.get("vllm_args")
    if not old_parsed or not old_parsed.get("ok"):
        old_parsed = read_vllm_args(launcher_args, old_vargv, env)
    new_parsed = read_vllm_args(launcher_args, new_vargv, env)
    if weights_backend(launcher_args, new_parsed) == "dynamo_gms":
        _apply_gms_env(launcher_args, env)
    reconfig = vllm_args.classify_reconfig(old_parsed, new_parsed) if (old_parsed.get("ok") and new_parsed.get("ok")) else None
    findings, preflight_report = argv_findings(launcher_args, new_parsed), {}
    if not findings and new_parsed.get("ok"):
        findings = [f for f in (vllm_args.override_finding(patch_vargv, old_parsed, new_parsed) if old_parsed.get("ok") else None,
                                value_finding(new_parsed)) if f]
    reshard = reshard_route(launcher_args, new_parsed, reconfig)
    if not findings and reconfig is not None and not reshard:
        findings = reconfig_findings(launcher_args, new_parsed, reconfig)
    if not findings:
        findings, preflight_report = preflight(launcher_args, new_parsed, env)
    if findings:
        _exit_json({"findings": findings, **preflight_report, "vllm_args": new_parsed,
                    **({"reconfig": reconfig} if reconfig else {}), "note": "not started; old engine left running"})
    old_port = state.get("port") or (old_parsed.get("args") or {}).get("port")
    new_port = new_parsed["args"]["port"]
    if reconfig is not None and reconfig.get("tier") == "noop":
        _exit_json({"reconfig": {**reconfig, "path": "noop", "restarted": False},
                    "pid": launcher_args.pid, "note": "nothing changed"}, code=0)
    live_values, live_reason = (
        (None, "the weight key changes (holder reshard)") if reshard
        else (None, "--reconfigure-restart given") if getattr(launcher_args, "reconfigure_restart", False)
        else (None, "the port changes") if new_port != old_port
        else live_reconfig_plan(state, reconfig)
    )
    can_apply_in_process = live_values is not None
    if launcher_args.zygote and not can_apply_in_process:
        zygote_reply = _check_zygote_for_reconfig(launcher_args, env, reconfig)
        if launcher_args.zygote_check:
            _exit_json({"zygote_check": zygote_reply, "reconfig": reconfig, "vllm_args": new_parsed}, code=0)
    elif launcher_args.zygote_check:
        _exit_json({"reconfig": reconfig, "vllm_args": new_parsed}, code=0)
    if new_port != old_port and port_in_use(new_port) and not launcher_args.accept_port_in_use:
        _exit_json({
            "finding": _port_in_use_finding(
                new_port, expected="a free port (or the replaced engine's own port)",
                why="the health check would read the other server's /health as this start's",
                actions=["keep the replaced engine's port", "pick a free port", "--accept-port-in-use: start anyway"],
            ),
            "reconfig": reconfig, "note": "not started; old engine left running",
        })
    if can_apply_in_process:
        report, live_reason = apply_in_process_reconfig(
            launcher_args, state, new_vargv, new_parsed, preflight_report, reconfig, live_values)
        if report is not None:
            return report
        if launcher_args.zygote:
            _check_zygote_for_reconfig(launcher_args, env, reconfig, live_fallback_reason=live_reason)
    reshard_report = None
    if reshard:
        reshard_report = reshard_target(launcher_args, state, old_vargv, new_vargv, old_parsed, new_parsed, env, reconfig)
        if load_format(launcher_args, new_parsed) == "ipc_cache":
            preflight_report["weights_probe"] = weights.probe(
                socket_dir(launcher_args, new_parsed), env.get("CUDA_VISIBLE_DEVICES"),
                new_parsed["args"]["tensor_parallel_size"])
    t_stop = time.perf_counter()
    stop_info = stop_engine(launcher_args.pid, port=old_port, marker=state.get("live_dir"),
                            start_time=state.get("start_time"))
    if not stop_info["stopped"]:
        _exit_json({"findings": [_finding(
            "stop_engine", stop_info, "every process of the old engine gone and its port closed",
            why="a new engine next to a live old one would share its GPUs and port",
            actions=["stop the listed pids yourself, then start the engine"],
        )], "reconfig": reconfig, "note": "not started; old engine may still be running"})
    reconfig_report = {
        **(reconfig or {}), "path": "reshard" if reshard else "restart", "restarted": True,
        "replaced_pid": launcher_args.pid, "restart_reason": live_reason,
        "stop_seconds": round(time.perf_counter() - t_stop, 6), "stop": stop_info,
        **({"reshard": reshard_report} if reshard_report is not None else {}),
    }
    return _launch_process(launcher_args, new_vargv, new_parsed, env, preflight_report,
                           extra_report={"reconfig": reconfig_report})


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
    p = corelab.spawn_logged(cmd, a.log, env=env)
    return {"pid": p.pid, "cmd": cmd, **zygote.wait_ready(a.zygote, p.pid)}


_TRACEBACK = "Traceback (most recent call last)"
_EXCEPTION_IGNORED = "Exception ignored "
FAILED = ("Engine core initialization failed", _TRACEBACK)


def _log_shows_failure(text):
    """True when a vLLM startup log contains a fatal failure marker.

    Ignores non-fatal finalizer tracebacks (`Exception ignored ...` on the immediately
    preceding line, e.g. Python 3.11-3.13 `Exception ignored in:` or Python 3.14+
    `Exception ignored while finalizing` from third-party `zmq.Context.__del__`), while
    still failing immediately on any unhandled `Traceback (most recent call last)` or
    `Engine core initialization failed`.
    """
    if not text:
        return False
    if "Engine core initialization failed" in text:
        return True
    if _TRACEBACK not in text:
        return False
    prev = ""
    for line in text.splitlines():
        if _TRACEBACK in line and _EXCEPTION_IGNORED not in prev:
            return True
        prev = line
    return False


def wait_healthy(port, pid, log, timeout=3600, t0=None, poll=0.25):
    """Seconds from t0 (default: now) until /health answers 200, or None as soon as the server
    exits (a zombie counts: a direct start is our unreaped child) or its log shows a failed
    start (the API process can outlive a dead engine core, so the pid alone is not enough)."""
    t0 = time.time() if t0 is None else t0
    while time.time() - t0 < timeout:
        if zygote._exited(pid) or _log_shows_failure(corelab.read_text(log)):
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
    return {"spawn_forced": SPAWN_FORCED in corelab.read_text(log, missing_ok=False)}


def fork_finding(fc):
    """None when the fork applied, else the finding (checks advise; the server still runs)."""
    if not fc["spawn_forced"]:
        return None
    return _finding("fork", fc, "no spawned interpreter after the API server",
                    why="vLLM fell back to spawn: each process imports vLLM again (~11-13 s)",
                    actions=[f'find what initialized CUDA in the parent: the reason after "{SPAWN_FORCED}" in the log',
                             "accept: the server is correct, only slower"],
                    risk="none for correctness; startup is not faster")


def launcher_parser():
    """Only launcher flags live here; none of them is a `vllm serve` flag (test_vllm_args.py checks this)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("cmd", choices=["start", "reconfigure", "zygote"])
    for flag, kw in (
        ("--pid", dict(type=int, default=0, help="reconfigure: pid of the running engine")),
        ("--cache-root", dict(required=True, help="folder for the launcher's caches")),
        ("--log", dict(required=True, help="the server's log file")),
        ("--vllm", dict(default="vllm", help="start: the vllm executable")),
        ("--weights", dict(action="store_true", help="start: map weights held by weights.py (--load-format ipc_cache, no disk fallback)")),
        ("--weights-backend", dict(choices=["", "none", "ipc_cache", "dynamo_gms"], default="", help="start: pluggable resident weights backend (default: ipc_cache with --weights, else none)")),
        ("--weights-socket-dir", dict(default="", help="daemon --socket-dir (default: vLLM's default folder)")),
        ("--gms-socket-dir", dict(default="", help="dynamo_gms: GMS_SOCKET_DIR holding gms_<uuid>_{weights,kv_cache}.sock")),
        ("--gms-read-only", dict(action="store_true", help="dynamo_gms: force RO lock mode (gms_read_only=true; attach committed GMS weights only)")),
        ("--gms-scratch-kv", dict(action="store_true", help="dynamo_gms: set DYN_GMS_SCRATCH_KV_ENABLED=1 for low-VRAM colocated shadow/standby workers")),
        ("--gms-pythonpath", dict(default="", help="dynamo_gms: optional extra PYTHONPATH entry containing gpu_memory_service")),
        ("--weights-accept-missing", dict(action="store_true", help="start: launch although no daemon answers")),
        ("--weights-accept-reconfig", dict(action="store_true", help="reconfigure: proceed with ipc_cache across a Tier-2 WeightCacheKey change")),
        ("--reconfigure-restart", dict(action="store_true", help="reconfigure: force a warm restart even for live fields")),
        ("--kvplan", dict(nargs="?", const="", default=None, metavar="DIR", help="reuse KV plan from DIR (default <cache-root>/kvplan; pass --kvplan=DIR)")),
        ("--fork", dict(action="store_true", help="fork engine core and workers instead of spawning")),
        ("--zygote", dict(default="", help="start: fork from zygote socket; zygote: listen there")),
        ("--zygote-check", dict(action="store_true", help="start/reconfigure: run preflight only (0 ok, 2 refused)")),
        ("--zygote-accept", dict(default="", help="start: accept zygote findings (env,stale)")),
        ("--zygote-modules", dict(default="", help="zygote: file with modules to pre-import")),
        ("--python", dict(default=sys.executable, help="zygote: the vLLM venv's python")),
        ("--accept-port-in-use", dict(action="store_true", help="start: launch although the port already has a listener")),
        ("--timeout", dict(type=float, default=3600, help="start: /health wait, seconds")),
    ):
        p.add_argument(flag, **kw)
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
