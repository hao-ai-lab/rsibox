"""Colocated weight lease, NCCL seeder and accept-gated standby pool for `vllm serve` engines.

Real processes, no simulation:

- Lease (CUDA IPC, no copy). A weight holder is vLLM's weight cache daemon (`vllm preload`,
  started by weights.py), one per GPU. Every engine on that GPU starts with `--load-format
  ipc_cache` (engine.py start --weights) and maps the holder's tensors through CUDA IPC: the
  engine allocates 0 bytes for weights, and K engines on one GPU share one copy.
- Seeder (NCCL, a copy). A holder on another GPU is started without reading the checkpoint
  (`vllm preload --load-format dummy`, random init on the GPU) and then filled from a holder that
  has the weights: two ranks map the two holders through CUDA IPC and `torch.distributed`
  broadcasts every tensor from source to destination in place (`copy`). The same copy restores
  a holder whose weights went wrong (a rejected patch, a corruption), so a restore is a full
  byte copy from a known-good holder, never an in-process snapshot.
- Standby pool (`Pool`). An active and a standby engine, each on its own holder (two GPUs), behind
  a router (one port). The router forwards to the active engine; on a fault (the engine process
  dies, /health stops answering, a sticky CUDA error, or a weight digest that disagrees with the
  ledger) it switches to the standby, which then serves. The failed side is repaired in the
  background: a dead engine is restarted leased (holder still resident), a wrong holder is
  re-copied over NCCL from the serving one.
- Accept gate. `evaluate(patch)` applies a weight patch to the active engine in place, runs the
  gate (greedy tokens and logprobs on reference prompts against the ledger's reference) and
  leaves it pending. Only `accept()` records it in the ledger and replays it onto the standby. A
  rejected patch fails over to the standby (which never saw it) and re-copies the patched holder
  from the standby's. A pending patch that is not accepted before the next `evaluate` is rolled
  back the same way. Invariant: every holder that can serve holds exactly the ledger.

Integrity: `digest` is a per-tensor (sum, sum of squares) over the raw 64-bit words, computed on
the GPU inside the engine, chunked, outside any lock (no host copy). Two holders agree when every
tensor's digest agrees. A permutation inside one tensor is not detected (sums commute).

Patches here are weight patches (scale or add noise to parameters matched by a regex), applied in
place, so CUDA graphs captured on the mapped weights stay valid. Code patches go through the
transaction kernel (adapter/), not here.

Usage (under the vLLM venv's python, one GPU per holder):
  python3 weight_scaling_colocate.py copy --model M --socket-dir D --src-gpu 0 --dst-gpu 1
  python3 weight_scaling_colocate.py demo --model M --gpus 4,5,6 --work-dir W [--max-model-len L]
  python3 engine.py pool ...      # the same entry points through the lab launcher

Each prints one JSON line. LeaseWorkerExtension is the engines' --worker-extension-cls (it runs in
every worker; VLLM_SERVER_DEV_MODE=1 exposes it through POST /collective_rpc).
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import http.server
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_CLS = "weight_scaling_colocate.LeaseWorkerExtension"
CHUNK_WORDS = 1 << 26          # 512 MiB of 64-bit words per digest step


# ---------------------------------------------------------------------------------------------
# Patches and the ledger (plain data: no torch)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    """A weight patch: `op` with `value` on every parameter whose name matches `target`
    (re.search). "scale" multiplies; "noise" adds value * std(param) * N(0, 1) with `seed`;
    "mlp_rescale" multiplies the up half of each matched gate_up_proj by value and divides each
    matched down_proj by it: with value a power of two the MLP's output is bit-identical (power-of-
    two scaling commutes with rounding), so the bytes change and the outputs must not."""
    id: str
    target: str
    op: str = "scale"
    value: float = 1.0
    seed: int = 0

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


class Ledger:
    """Accepted patches, in order. Append-only; entries are frozen, so no caller can change
    what was recorded (the old model kept mutable dicts and a shallow copy for rollback)."""

    def __init__(self):
        self._entries: tuple[Patch, ...] = ()

    def append(self, patch: Patch):
        if any(p.id == patch.id for p in self._entries):
            raise ValueError(f"patch {patch.id!r} already in the ledger")
        self._entries = self._entries + (patch,)

    @property
    def entries(self) -> tuple[Patch, ...]:
        return self._entries

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self._entries]


def digests_agree(a: dict, b: dict) -> dict:
    """{"ok", "differ": names whose digest differs, "missing": names only on one side}."""
    differ = sorted(n for n in a.keys() & b.keys() if a[n] != b[n])
    missing = sorted(a.keys() ^ b.keys())
    return {"ok": not differ and not missing, "differ": differ[:20], "n_differ": len(differ),
            "missing": missing[:20]}


def gate_compare(ref: list, got: list, tol: float) -> dict:
    """Greedy tokens must be identical and every chosen token's logprob within `tol`."""
    same_tokens = all(r["tokens"] == g["tokens"] for r, g in zip(ref, got)) and len(ref) == len(got)
    dlp = max((abs(x - y) for r, g in zip(ref, got)
               for x, y in zip(r["logprobs"], g["logprobs"])), default=0.0)
    return {"pass": bool(same_tokens and dlp <= tol), "same_tokens": same_tokens,
            "max_abs_dlogprob": dlp, "tol": tol}


# ---------------------------------------------------------------------------------------------
# GPU side: digest, patch, fault (torch imported lazily: this module is also imported on CPU)
# ---------------------------------------------------------------------------------------------

def tensor_digest(t):
    """(sum, sum of squares) of the tensor's bytes read as int64 words (wrapping, so exact and
    order independent), chunked to bound the temporary memory. Stays on the GPU."""
    import torch
    flat = t.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    b = flat.view(torch.uint8)
    n8 = b.numel() // 8
    s1 = torch.zeros((), dtype=torch.int64, device=t.device)
    s2 = torch.zeros((), dtype=torch.int64, device=t.device)
    if n8:
        w = b[: n8 * 8].view(torch.int64)
        for i in range(0, n8, CHUNK_WORDS):
            c = w[i:i + CHUNK_WORDS]
            s1 += c.sum()
            s2 += (c * c).sum()
    if b.numel() % 8:
        tail = b[n8 * 8:].to(torch.int64)
        s1 += tail.sum()
        s2 += (tail * tail).sum()
    return s1, s2


def model_digest(named):
    """{name: "s1:s2"} for (name, tensor) pairs, plus seconds (CUDA synchronized)."""
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pend = [(n, tensor_digest(t)) for n, t in named]
    torch.cuda.synchronize()
    out = {n: f"{int(a)}:{int(b)}" for n, (a, b) in pend}
    return out, time.perf_counter() - t0


def apply_patch(named: dict, patch: Patch) -> list[str]:
    """Apply `patch` in place to the matching tensors of `named`; returns the names touched."""
    import torch
    rx = re.compile(patch.target)
    hit = sorted(n for n in named if rx.search(n))
    g = torch.Generator(device="cuda").manual_seed(patch.seed)
    with torch.no_grad():
        for n in hit:
            t = named[n]
            if patch.op == "scale":
                t.mul_(patch.value)
            elif patch.op == "mlp_rescale":
                if "gate_up_proj" in n:
                    t[t.shape[0] // 2:].mul_(patch.value)      # [gate; up] along dim 0
                elif "down_proj" in n:
                    t.mul_(1.0 / patch.value)
            elif patch.op == "noise":
                std = t.float().std().item()
                noise = torch.randn(t.shape, generator=g, device=t.device, dtype=torch.float32)
                t.add_((noise * (patch.value * std)).to(t.dtype))
            else:
                raise ValueError(f"unknown patch op {patch.op!r}")
    torch.cuda.synchronize()
    return hit


class LeaseWorkerExtension:
    """vLLM --worker-extension-cls: methods run in every worker via collective_rpc. Arguments
    arrive as JSON strings (the dev endpoint passes strings only); every method returns a dict
    and catches its own errors except `lease_fault("cuda")`, whose job is to break the context."""

    def _named(self):
        model = self.model_runner.model
        named = {}
        for n, p in model.named_parameters():
            if p.is_cuda:
                named[n] = p.data
        return named

    def lease_digest(self, _arg: str = "{}"):
        try:
            d, s = model_digest(sorted(self._named().items()))
            return {"ok": True, "digest": d, "seconds": s, "pid": os.getpid()}
        except Exception as e:  # noqa: BLE001 (a reply, never a raise across the RPC)
            return {"ok": False, "error": repr(e), "pid": os.getpid()}

    def lease_apply(self, patch_json: str):
        try:
            t0 = time.perf_counter()
            hit = apply_patch(self._named(), Patch(**json.loads(patch_json)))
            return {"ok": True, "touched": len(hit), "seconds": time.perf_counter() - t0}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e)}

    def lease_info(self, _arg: str = "{}"):
        import torch
        named = self._named()
        ptrs = {t.untyped_storage().data_ptr() for t in named.values()}
        return {"ok": True, "pid": os.getpid(), "n_params": len(named),
                "param_bytes": sum(t.numel() * t.element_size() for t in named.values()),
                "distinct_storages": len(ptrs),
                "torch_allocated_bytes": torch.cuda.memory_allocated()}

    def lease_fault(self, kind: str):
        """Fault injection for tests: "corrupt" overwrites one weight in place (the holder's
        memory, shared through IPC); "cuda" triggers a device-side assert (sticky: the
        context is unusable afterwards)."""
        import torch
        if kind == "corrupt":
            named = self._named()
            mlp = sorted(n for n in named if "mlp" in n) or sorted(named)
            name = mlp[len(mlp) // 2]
            with torch.no_grad():
                named[name].view(torch.uint8).random_(0, 256)    # any dtype, fp8 included
            torch.cuda.synchronize()
            return {"ok": True, "corrupted": name}
        if kind == "cuda":
            x = torch.zeros(4, device="cuda")
            idx = torch.tensor([1 << 40], device="cuda")
            y = x[idx]                       # device-side assert: index out of bounds
            torch.cuda.synchronize()
            return {"ok": True, "unexpected": float(y.sum())}
        return {"ok": False, "error": f"unknown fault {kind!r}"}


# ---------------------------------------------------------------------------------------------
# NCCL seeding: copy one holder into another, both mapped through CUDA IPC
# ---------------------------------------------------------------------------------------------

def _holder_tensors(model, socket_dir, max_model_len=None):
    """Map this GPU's holder: ({name: cuda tensor}, key fields). Uses vLLM's own key and
    protocol, so the holder answers exactly as it answers an engine."""
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.weight_cache import protocol as P
    kw = {"model": model}
    if max_model_len:
        kw["max_model_len"] = max_model_len
    cfg = EngineArgs(**kw).create_engine_config()
    key = P.WeightCacheKey.from_model_config(
        cfg.model_config, tp_size=1, tp_rank=0, cache_config=cfg.cache_config,
        enable_expert_parallel=cfg.parallel_config.enable_expert_parallel,
        expert_placement_strategy=cfg.parallel_config.expert_placement_strategy,
        kernel_config=cfg.kernel_config)
    path = P.get_socket_path(P.get_current_device_uuid(), socket_dir)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    try:
        P.send_msg(s, {"cmd": "get_state", "cache_key": key})
        reply = P.recv_msg(s)
    finally:
        s.close()
    if reply.get("status") != "ok":
        raise RuntimeError(f"holder at {path} answered {reply}")
    dev = torch.cuda.current_device()
    return {n: e.rebuild(dev) for n, e in reply["entries"].items() if e.ipc_args is not None}


def copy_rank(a):
    """One side of `copy`: rank 0 maps the source holder, rank 1 the destination; every tensor
    is broadcast 0 -> 1 in place. Prints one JSON line (rank 1 only reports the copy)."""
    import torch
    import torch.distributed as dist
    t_start = time.perf_counter()
    torch.cuda.set_device(0)
    named = _holder_tensors(a.model, a.socket_dir, a.max_model_len)
    t_mapped = time.perf_counter()
    dist.init_process_group("nccl", init_method=a.init, rank=a.rank, world_size=2,
                            device_id=torch.device("cuda", 0))
    names = sorted(named)
    manifest = [(n, tuple(named[n].shape), str(named[n].dtype)) for n in names]
    mine = [manifest]
    theirs = [None]
    dist.broadcast_object_list(theirs if a.rank == 1 else mine, src=0)
    if a.rank == 1 and theirs[0] != manifest:
        raise RuntimeError("source and destination holders hold different tensors: "
                           f"{set(map(tuple, theirs[0])) ^ set(manifest)}")
    nbytes = sum(named[n].numel() * named[n].element_size() for n in names)
    before = [None, None]
    dist.all_gather_object(before, model_digest([(n, named[n]) for n in names])[0])
    agree_before = digests_agree(before[0], before[1])
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for n in names:
        t = named[n]
        if t.is_contiguous():
            dist.broadcast(t.view(-1).view(torch.uint8), src=0)
        else:
            buf = t.contiguous()
            dist.broadcast(buf.view(-1).view(torch.uint8), src=0)
            if a.rank == 1:
                t.copy_(buf)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dig, dig_s = model_digest([(n, named[n]) for n in names])
    out = [dig]
    gathered = [None, None]
    dist.all_gather_object(gathered, out[0])
    agree = digests_agree(gathered[0], gathered[1])
    dist.destroy_process_group()
    print(json.dumps({"rank": a.rank, "tensors": len(names), "bytes": nbytes,
                      "map_s": t_mapped - t_start, "nccl_s": t1 - t0,
                      "gbps": nbytes / max(t1 - t0, 1e-9) / 1e9, "digest_s": dig_s,
                      "digests_agree": agree["ok"], "n_differ": agree["n_differ"],
                      "n_differ_before": agree_before["n_differ"]}))
    return 0 if agree["ok"] else 1


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def copy(model, socket_dir, src_gpu, dst_gpu, max_model_len=None, python=sys.executable,
         log_dir=None):
    """Copy the holder on src_gpu into the holder on dst_gpu over NCCL. Returns the
    destination rank's report plus wall seconds (process start, vLLM import, IPC map, copy).
    A rank that fails stops the other one, which would wait for it in NCCL init."""
    t0 = time.monotonic()
    init = f"tcp://127.0.0.1:{_free_port()}"
    procs, files = [], []
    with contextlib.ExitStack() as stack:
        for rank, gpu in ((0, src_gpu), (1, dst_gpu)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [python, str(Path(__file__).resolve()), "copy-rank", "--rank", str(rank),
                   "--init", init, "--model", model, "--socket-dir", socket_dir]
            if max_model_len:
                cmd += ["--max-model-len", str(max_model_len)]
            err = stack.enter_context(open(os.path.join(
                log_dir, f"copy_{src_gpu}_{dst_gpu}_r{rank}.log"), "w")) \
                if log_dir else subprocess.DEVNULL
            out = stack.enter_context(tempfile.TemporaryFile("w+"))   # no pipe to fill
            files.append(out)
            procs.append(subprocess.Popen(cmd, env=env, stdout=out, stderr=err, text=True))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            codes = [proc.poll() for proc in procs]
            if None not in codes or any(code not in (None, 0) for code in codes):
                break
            time.sleep(0.1)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        outs = []
        for out in files:
            out.seek(0)
            outs.append(out.read())
    rcs = [p.returncode for p in procs]
    rep = {}
    for line in outs[1].splitlines()[::-1]:
        if line.startswith("{"):
            rep = json.loads(line)
            break
    rep.update({"wall_s": time.monotonic() - t0, "returncodes": rcs, "src_gpu": src_gpu,
                "dst_gpu": dst_gpu, "ok": rcs == [0, 0] and rep.get("digests_agree", False)})
    return rep


# ---------------------------------------------------------------------------------------------
# Holders and engines: thin wrappers over weights.py and engine.py (the lab path)
# ---------------------------------------------------------------------------------------------

def vllm_bin(python):
    return str(Path(python).with_name("vllm"))


@dataclass
class Holder:
    gpu: int
    socket_dir: str
    log: str
    proc: subprocess.Popen | None = None
    seeded_from: int | None = None
    report: dict = field(default_factory=dict)

    def start(self, model, python, max_model_len=None, dummy=False):
        import weights
        finding = weights.long_socket_path_finding(self.socket_dir)
        if finding:
            raise RuntimeError(f"holder on GPU {self.gpu}: socket path too long: "
                               f"{finding['observed']}")
        extra = ["--load-format", "dummy"] if dummy else []
        cmd = weights.daemon_cmd(model, self.socket_dir, 1, max_model_len, vllm_bin(python), extra)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu))
        t0 = time.monotonic()
        with open(self.log, "w") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True)
        r = weights.wait_ready(self.log, self.proc.pid, timeout=1800)
        self.report = {"gpu": self.gpu, "dummy": dummy, "ready": r["ready"],
                       "ready_s": time.monotonic() - t0, "pid": self.proc.pid}
        if not r["ready"]:
            weights.stop(self.proc)
            raise RuntimeError(f"holder on GPU {self.gpu} not ready: {r}; see {self.log}")
        return self.report

    def stop(self):
        import weights
        if self.proc is not None:
            weights.stop(self.proc)
            self.proc = None


LOG_PHASES = {
    # vLLM's own startup lines (each measured by vLLM itself, in the engine process)
    "weight_load_s": re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    "kv_profile_warmup_s": re.compile(r"init engine \(profile, create kv cache, warmup model\) "
                                      r"took ([\d.]+) s"),
    "graph_capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),  # whole s
    "torch_compile_s": re.compile(r"torch\.compile took ([\d.]+) s in total"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
}


def log_phases(log):
    text = open(log, errors="replace").read()
    out = {}
    for k, rx in LOG_PHASES.items():
        m = rx.findall(text)
        if not m:
            continue
        last = m[-1]
        if k == "weight_load_s":
            out["weight_alloc_gib"], out[k] = float(last[0]), float(last[1])
        else:
            out[k] = float(last)
    return out


@dataclass
class Engine:
    name: str
    gpu: int
    port: int
    work: str
    holder: Holder | None          # None: load from disk (cold start)
    pid: int | None = None
    start_report: dict = field(default_factory=dict)

    @property
    def log(self):
        return os.path.join(self.work, f"{self.name}.log")

    def start(self, model, python, gmu, max_model_len, extra=()):
        """engine.py start (the lab launcher), leased from the holder when there is one."""
        launcher = [python, str(HERE / "engine.py"), "start", "--cache-root",
                    os.path.join(self.work, "cache"), "--log", self.log, "--vllm",
                    vllm_bin(python)]
        if self.holder is not None:
            launcher += ["--weights", "--weights-socket-dir", self.holder.socket_dir]
        launcher += [model, "--port", str(self.port), "--gpu-memory-utilization", str(gmu),
                     "--max-model-len", str(max_model_len), "--worker-extension-cls", EXT_CLS,
                     *extra]
        pp = os.environ.get("PYTHONPATH")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu), VLLM_SERVER_DEV_MODE="1",
                   PYTHONPATH=os.pathsep.join([str(HERE)] + ([pp] if pp else [])))
        t0 = time.monotonic()
        r = subprocess.run(launcher, env=env, capture_output=True, text=True, timeout=3600)
        t_healthy = time.monotonic() - t0
        line = next((x for x in r.stdout.splitlines()[::-1] if x.startswith("{")), "{}")
        rep = json.loads(line)
        if r.returncode != 0 or rep.get("healthy_after_s") is None:
            raise RuntimeError(f"engine {self.name} did not start: rc={r.returncode} "
                               f"{line[:2000]} stderr={r.stderr[-2000:]}; log {self.log}")
        self.pid = rep["pid"]
        ttft = first_token_s(self.port, model)
        phases = log_phases(self.log)
        known = sum(phases.get(k, 0.0) for k in ("weight_load_s", "kv_profile_warmup_s"))
        if "kv_profile_warmup_s" in phases:
            # vLLM's "init engine" span holds compile + profile + KV alloc + warmup + capture
            phases["kv_profile_alloc_warmup_excl_graph_s"] = (
                phases["kv_profile_warmup_s"] - phases.get("graph_capture_s", 0.0))
        self.start_report = {
            "engine": self.name, "gpu": self.gpu, "leased": self.holder is not None,
            "healthy_after_s": rep["healthy_after_s"], "launcher_wall_s": t_healthy,
            **phases,
            # what vLLM does not time itself: process spawn, imports, API server, distributed
            # init, model build on meta (leased) and engine handshakes
            "spawn_import_init_s_residual": rep["healthy_after_s"] - known,
            "first_token_s": ttft,
            "weights_probe": rep.get("weights_probe"),
        }
        return self.start_report

    def stop(self):
        import engine
        if self.pid:
            engine.stop_engine(self.pid, self.port)
            self.pid = None

    def kill(self):
        """SIGKILL the engine's whole session (API server, engine core, workers)."""
        if self.pid:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # -- calls into the running engine --
    def rpc(self, method, arg="{}", timeout=300):
        body = json.dumps({"method": method, "args": [arg], "timeout": timeout})
        return post_json(self.port, "/collective_rpc", body, timeout + 10)["results"][0]

    def reset_prefix_cache(self):
        return post_json(self.port, "/reset_prefix_cache?reset_running_requests=true", "{}", 60)

    def healthy(self, timeout=0.5):
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
            c.request("GET", "/health")
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except OSError:
            return False


def post_json(port, path, body, timeout):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"POST {path} on :{port}: {r.status} {data[:500]!r}")
    return json.loads(data) if data else {}


PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "In 1905, Albert Einstein published", "The three primary colors are"]


def reference(port, model, max_tokens=16):
    """Greedy completions with the chosen token's logprob, one request per prompt."""
    out = []
    for p in PROMPTS:
        body = json.dumps({"model": model, "prompt": p, "max_tokens": max_tokens,
                           "temperature": 0, "logprobs": 1, "seed": 0})
        r = post_json(port, "/v1/completions", body, 120)["choices"][0]
        out.append({"text": r["text"], "tokens": r["logprobs"]["tokens"],
                    "logprobs": [float(x) for x in r["logprobs"]["token_logprobs"]]})
    return out


def stream_first_token(port, model, prompt="Hello", max_tokens=4, timeout=30):
    """(seconds to the first streamed token, served_by header) or raises."""
    t0 = time.monotonic()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True})
    c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"status {r.status}")
    served = r.getheader("X-Served-By")
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            raise RuntimeError("stream ended before a token")
        buf += chunk
        if b'"text"' in buf:
            t1 = time.monotonic()
            r.read()
            c.close()
            return t1 - t0, served, t1


def first_token_s(port, model):
    return stream_first_token(port, model)[0]


# ---------------------------------------------------------------------------------------------
# Router: one port, forwards to the active engine, switches on failure
# ---------------------------------------------------------------------------------------------

class Router:
    def __init__(self, pool, port):
        self.pool, self.port = pool, port
        self.open = threading.Event()
        self.open.set()
        self.cv = threading.Condition()
        self.inflight = 0
        router = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _forward(self, method):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                router.open.wait()                 # held while the pool drains (quiesce)
                with router.cv:
                    router.inflight += 1
                try:
                    self._forward_body(method, body)
                finally:
                    with router.cv:
                        router.inflight -= 1
                        router.cv.notify_all()

            def _forward_body(self, method, body):
                for attempt in range(3):
                    eng = router.pool.active
                    try:
                        c = http.client.HTTPConnection("127.0.0.1", eng.port, timeout=120)
                        c.request(method, self.path, body=body or None,
                                  headers={"Content-Type": "application/json"})
                        r = c.getresponse()
                    except (OSError, http.client.HTTPException):
                        router.pool.on_backend_error(eng, "connect")
                        continue
                    if r.status >= 500 and attempt < 2:
                        r.read()
                        router.pool.on_backend_error(eng, f"http {r.status}")
                        continue
                    self.send_response(r.status)
                    for k, v in r.getheaders():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(k, v)
                    self.send_header("X-Served-By", eng.name)
                    self.end_headers()
                    try:
                        while True:
                            chunk = r.read1(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (OSError, http.client.HTTPException):
                        # the engine died mid-stream: this client sees a cut stream (its
                        # tokens so far were served); later requests go to the standby
                        router.pool.on_backend_error(eng, "stream broke")
                    c.close()
                    return
                self.send_response(503)
                self.end_headers()

            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @contextlib.contextmanager
    def drained(self, timeout=60):
        """Hold new requests and wait until none is in flight: a check run inside sees the
        engine alone (batched neighbours change logprobs by up to ~0.1 on Qwen3-8B)."""
        self.open.clear()
        t0 = time.monotonic()
        with self.cv:
            self.cv.wait_for(lambda: self.inflight == 0, timeout)
        try:
            yield time.monotonic() - t0
        finally:
            self.open.set()

    def close(self):
        self.open.set()
        self.server.shutdown()


# ---------------------------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------------------------

class Pool:
    """Active and standby engines on two holders (two GPUs), a router, a ledger and the gate.

    Invariants (each a bug of the old model, regression-tested in test_weight_scaling_colocate):
    - (a) Nothing reaches the ledger or the standby without accept(); a pending patch that is not
      accepted is rolled back before the next evaluate (the active holder is re-copied from the
      standby's), so active and standby never diverge silently.
    - (b) A rollback is a byte copy from a holder that holds the ledger (NCCL), checked by digest;
      there is no in-process snapshot to be shallow.
    - (c) Failover goes to the other holder, never to an engine on the faulted holder, and the
      faulted holder is re-copied before its engine serves again.
    - (d) An engine that failed over away, holds an unaccepted patch it no longer serves, is
      being re-copied, or got an accept replay that did not complete, is tainted until
      restore_holder, respawn or accept's digest shows it matches the ledger: failover never
      goes to a tainted engine, restore_holder never copies from one (nor from the engine with
      the pending patch), and evaluate refuses while any engine is tainted. The pending patch
      remembers its engine, so rollback repairs that engine and accept refuses once that
      engine has stopped serving; accept holds the lock across its replay.
    """

    def __init__(self, model, python, work, active: Engine, standby: Engine, router_port,
                 tol=0.05, max_model_len=None):
        self.model, self.python, self.work = model, python, work
        self.max_model_len = max_model_len   # the holders' --max-model-len (their cache key)
        self.active, self.standby = active, standby
        self.ledger = Ledger()
        self.pending: Patch | None = None
        self.pending_engine: Engine | None = None
        self.tainted: set[str] = set()       # engine names whose holder may not hold the ledger
        self.tol = tol
        self.lock = threading.RLock()
        self.events = []
        self.ref = reference(active.port, model)
        self.ref_digest = self.digest(active)
        self.router = Router(self, router_port)
        self._stop = threading.Event()
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()

    # -- state checks --
    def digest(self, eng):
        r = eng.rpc("lease_digest")
        if not r.get("ok"):
            raise RuntimeError(f"digest on {eng.name}: {r}")
        self.events.append({"t": time.monotonic(), "ev": "digest", "engine": eng.name,
                            "seconds": r["seconds"]})
        return r["digest"]

    def check_integrity(self, eng=None):
        """Digest of `eng` (default: active) against the ledger's digest."""
        eng = eng or self.active
        d = self.digest(eng)
        return digests_agree(self.ref_digest, d)

    # -- failover --
    def failover(self, reason, failed=None):
        """Switch the router to the standby. Returns the switch record (monotonic times)."""
        with self.lock:
            failed = failed or self.active
            if failed is not self.active:
                return None                  # already switched
            t = time.monotonic()
            if self.standby.name in self.tainted:
                last = self.events[-1] if self.events else {}
                if last.get("ev") != "failover_refused":     # the watchdog retries every 50 ms
                    self.events.append({"t": t, "ev": "failover_refused", "reason": reason,
                                        "from": failed.name, "tainted": self.standby.name})
                return None
            self.active, self.standby = self.standby, self.active
            self.tainted.add(failed.name)
            rec = {"t": t, "ev": "failover", "reason": reason, "from": failed.name,
                   "to": self.active.name}
            self.events.append(rec)
            return rec

    def on_backend_error(self, eng, why):
        """A refused connection or a 5xx from the active engine. In this pool every request is
        well formed, so a 5xx means the engine broke (a sticky CUDA error answers 500 before
        /health notices)."""
        if eng is self.active:
            self.failover(f"router: {why}", eng)

    def _watchdog(self):
        while not self._stop.is_set():
            eng = self.active
            if eng.pid and not eng.healthy(timeout=0.5) and not eng.healthy(timeout=0.5):
                self.failover("health", eng)
            time.sleep(0.05)

    # -- repairs (the side that does not serve) --
    def restore_holder(self, broken: Engine, good: Engine):
        """Re-copy broken's holder from good's over NCCL, reset broken's prefix cache, and check
        it now matches the ledger digest. broken stays tainted until it does."""
        with self.lock:
            if broken is good or good.name in self.tainted or good is self.pending_engine:
                raise RuntimeError(f"restore {broken.name} from {good.name}: the source is "
                                   "the same engine, holds an unaccepted patch, or does not "
                                   "hold the ledger")
            self.tainted.add(broken.name)   # the copy overwrites it, even if it then fails
        rep = copy(self.model, broken.holder.socket_dir, good.holder.gpu, broken.holder.gpu,
                   max_model_len=self.max_model_len, log_dir=self.work, python=self.python)
        if not rep["ok"]:
            raise RuntimeError(f"restore copy failed: {rep}")
        broken.reset_prefix_cache()
        agree = self.check_integrity(broken)
        if agree["ok"]:
            with self.lock:
                self.tainted.discard(broken.name)
        rec = {"t": time.monotonic(), "ev": "restore", "engine": broken.name,
               "copy": rep, "digest_ok": agree["ok"]}
        self.events.append(rec)
        return rec

    def respawn(self, eng: Engine, gmu, max_model_len):
        """Restart a dead engine leased from its (still resident) holder, then replay nothing:
        the holder already holds the ledger; check it."""
        eng.kill()
        time.sleep(1.0)
        eng.pid = None
        rep = eng.start(self.model, self.python, gmu, max_model_len)
        agree = self.check_integrity(eng)
        if agree["ok"]:
            with self.lock:
                self.tainted.discard(eng.name)
        rec = {"t": time.monotonic(), "ev": "respawn", "engine": eng.name, "start": rep,
               "digest_ok": agree["ok"]}
        self.events.append(rec)
        return rec

    # -- the accept gate --
    def evaluate(self, patch: Patch):
        """Apply `patch` to the active engine, gate it. Pass: pending (accept() commits it).
        Fail: the standby takes over and the patched holder is restored."""
        out = {"patch": patch.id}
        if self.pending is not None:
            out["rolled_back_pending"] = self.rollback("superseded by " + patch.id)
        with self.lock:
            if self.tainted:
                # a rejected patch could not fail over, nor a pending one be rolled back
                raise RuntimeError(f"evaluate {patch.id}: {sorted(self.tainted)} not repaired "
                                   "(restore_holder or respawn first)")
        eng = self.active
        with self.router.drained() as drain_s:
            r = eng.rpc("lease_apply", patch.to_json())
            if not r.get("ok"):
                raise RuntimeError(f"apply {patch.id}: {r}")
            eng.reset_prefix_cache()
            got = reference(eng.port, self.model)
            gate = gate_compare(self.ref, got, self.tol)
        out["drain_s"] = drain_s
        out.update({"touched": r["touched"], "apply_s": r["seconds"], "gate": gate})
        if gate["pass"]:
            with self.lock:
                self.pending, self.pending_engine = patch, eng
            out["state"] = "pending"
            return out
        out["state"] = "rejected"
        out["failover"] = self.failover(f"patch {patch.id} rejected", eng)
        if out["failover"] is None and self.active is eng:
            raise RuntimeError(f"patch {patch.id} rejected and the standby "
                               f"{self.standby.name} does not hold the ledger: {eng.name} "
                               "keeps serving the rejected patch")
        out["restore"] = self.restore_holder(eng, self.active)
        return out

    def rollback(self, why):
        """Undo the pending patch: the standby (ledger state) takes over if the patched engine
        still serves, and the patched holder is re-copied from the serving one."""
        with self.lock:
            patch, eng = self.pending, self.pending_engine
            if patch is None:
                raise RuntimeError("nothing pending to roll back")
            failover_record = None
            if eng is self.active:
                failover_record = self.failover(f"rollback {patch.id}: {why}", eng)
                if failover_record is None:
                    raise RuntimeError(f"rollback {patch.id}: the standby "
                                       f"{self.standby.name} does not hold the ledger")
            self.tainted.add(eng.name)
            self.pending = self.pending_engine = None
        rs = self.restore_holder(eng, self.active)
        return {"patch": patch.id, "failover": failover_record, "restore": rs}

    def accept(self):
        """Commit the pending patch: ledger, then replay onto the standby, then the new
        reference (tokens/logprobs and digest) is the ledger's. The new ledger digest is the
        active engine's, taken before the replay, so the standby is verified against it even if
        the active engine dies after the replay; the reference prompts then come from the
        verified standby."""
        # The lock is held across the replay, so no failover can swap the engines in between
        # (a failover waits for it); a standby whose replay did not complete is tainted.
        with self.lock:
            patch = self.pending
            if patch is None:
                raise RuntimeError("nothing pending to accept")
            if self.pending_engine is not self.active:
                raise RuntimeError(f"accept {patch.id}: its engine "
                                   f"{self.pending_engine.name} no longer serves (failed over); "
                                   "rollback() instead")
            active, standby = self.active, self.standby
            patched_digest = self.digest(active)          # before anything changes
            self.tainted.add(standby.name)
            replay = standby.rpc("lease_apply", patch.to_json())
            if not replay.get("ok"):
                raise RuntimeError(f"replay {patch.id} on standby {standby.name}: {replay}")
            standby.reset_prefix_cache()
            self.ledger.append(patch)
            self.pending = self.pending_engine = None
            self.ref_digest = patched_digest
        standby_agree = digests_agree(patched_digest, self.digest(standby))
        if standby_agree["ok"]:
            with self.lock:
                self.tainted.discard(standby.name)
        with self.router.drained():
            try:
                self.ref = reference(active.port, self.model)
            except Exception:  # noqa: BLE001 - the active engine died after the replay
                if not standby_agree["ok"]:
                    raise
                self.ref = reference(standby.port, self.model)
        return {"accepted": patch.id, "ledger": self.ledger.ids,
                "standby_digest_ok": standby_agree["ok"]}

    def close(self):
        self._stop.set()
        self.router.close()


# ---------------------------------------------------------------------------------------------
# Load generator for failover timing
# ---------------------------------------------------------------------------------------------

class Load:
    """Back-to-back streaming requests through the router; records every first token."""

    def __init__(self, port, model):
        self.port, self.model = port, model
        self.samples, self.errors = [], []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            t_send = time.monotonic()
            try:
                ttft, served, t_tok = stream_first_token(self.port, self.model, timeout=60)
                self.samples.append({"t_send": t_send, "t_first_token": t_tok,
                                     "served_by": served})
            except Exception as e:  # noqa: BLE001
                self.errors.append({"t": time.monotonic(), "error": repr(e)[:200]})
                time.sleep(0.01)

    def first_after(self, t_fault, served_by, timeout=120):
        """First token from `served_by` after t_fault (waits for one)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for s in list(self.samples):
                if s["t_first_token"] > t_fault and s["served_by"] == served_by:
                    return s
            time.sleep(0.01)
        return None

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=60)


def measure_fault(pool, load, kind):
    """Inject `kind` into the active engine; return fault -> first served token on the
    standby, how it was detected, and the post-failover correctness check."""
    victim, target = pool.active, pool.standby
    t_fault = time.monotonic()
    if kind == "kill":
        victim.kill()
    elif kind == "cuda":
        try:
            r = victim.rpc("lease_fault", "cuda", timeout=10)
            err = None if r.get("ok") is False else f"no error: {r}"
        except Exception as e:  # noqa: BLE001 (the engine is expected to break)
            err = repr(e)[:300]
        # the RPC that ran the faulting kernel reports the error: the coordinator's signal
        pool.failover(f"sticky CUDA error from rpc: {err}", victim)
    elif kind == "corrupt":
        victim.rpc("lease_fault", "corrupt")
        integ = pool.check_integrity(victim)
        if not integ["ok"]:
            pool.failover(f"digest mismatch ({integ['n_differ']} tensors)", victim)
    s = load.first_after(t_fault, target.name)
    fo = next((e for e in pool.events[::-1] if e["ev"] == "failover" and e["t"] >= t_fault), None)
    return {"fault": kind, "from": victim.name, "to": target.name,
            "fault_to_switch_s": (fo["t"] - t_fault) if fo else None,
            "fault_to_first_token_s": (s["t_first_token"] - t_fault) if s else None,
            "detected_by": fo["reason"] if fo else None,
            **correctness(pool)}


def correctness(pool):
    """The reference prompts through the router: under live load (tokens must match; logprobs
    move with batching) and drained (must match the ledger reference exactly)."""
    loaded = gate_compare(pool.ref, reference(pool.router.port, pool.model), 0.0)
    with pool.router.drained():
        alone = gate_compare(pool.ref, reference(pool.active.port, pool.model), 0.0)
    return {"correct_under_load": loaded, "correct_drained": alone}


# ---------------------------------------------------------------------------------------------
# The end-to-end demo (what the GPU evidence runs)
# ---------------------------------------------------------------------------------------------

def demo(a):
    model, python, work = a.model, a.python, os.path.abspath(a.work_dir)
    os.makedirs(work, exist_ok=True)
    g0, g1, g2 = (int(x) for x in a.gpus.split(","))
    sock = os.path.join(work, "sock")
    os.makedirs(sock, mode=0o700, exist_ok=True)
    res = {"model": model, "gpus": [g0, g1, g2], "max_model_len": a.max_model_len}
    gmu, mml = a.gmu, a.max_model_len
    eng_list, holders = [], []

    def save():
        with open(os.path.join(work, "result.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)

    try:
        # 1. cold start from disk, GPU g2 (nothing else on it). The first one also fills the
        # compile cache; the second (same caches the leased starts get) is the baseline.
        for tag in ("cold_first", "cold_start"):
            cold = Engine(tag, g2, a.port_base + 0, work, None)
            eng_list.append(cold)
            res[tag] = cold.start(model, python, gmu, mml)
            res["cold_ref"] = reference(cold.port, model)
            cold.stop()
            save()
        # 2. holder H0 from disk on g0; H1 on g1 without the checkpoint, seeded over NCCL
        h0 = Holder(g0, sock, os.path.join(work, "holder0.log"))
        h1 = Holder(g1, sock, os.path.join(work, "holder1.log"), seeded_from=g0)
        holders += [h0, h1]
        res["holder0"] = h0.start(model, python, mml)
        res["holder1_dummy"] = h1.start(model, python, mml, dummy=True)
        res["seed_h0_to_h1"] = copy(model, sock, g0, g1, mml, python, work)
        save()
        # 3. leased engines: active on H0, standby on H1, and a second lessee of H0
        act = Engine("A", g0, a.port_base + 1, work, h0)
        sby = Engine("S", g1, a.port_base + 2, work, h1)
        eng_list += [act, sby]
        res["leased_start_active"] = act.start(model, python, gmu, mml)
        res["leased_start_standby"] = sby.start(model, python, gmu, mml)
        res["lease_info"] = {"A": act.rpc("lease_info"), "S": sby.rpc("lease_info")}
        colo = Engine("A2", g0, a.port_base + 3, work, h0)
        eng_list.append(colo)
        res["leased_start_colocated"] = colo.start(model, python, gmu, mml)
        res["colocated_info"] = colo.rpc("lease_info")
        res["colocated_same_outputs"] = gate_compare(reference(act.port, model),
                                                     reference(colo.port, model), 0.0)
        colo.stop()
        save()
        # 4. correctness of the lease: leased (H0, H1) == cold disk load
        pool = Pool(model, python, work, act, sby, a.port_base + 9, tol=a.tol,
                    max_model_len=mml)
        res["lease_vs_cold"] = {"A": gate_compare(res["cold_ref"], pool.ref, 0.0),
                                "S": gate_compare(res["cold_ref"], reference(sby.port, model),
                                                  0.0),
                                "digest_H0_vs_H1": digests_agree(pool.ref_digest,
                                                                 pool.digest(sby))["ok"]}
        load = Load(pool.router.port, model)
        time.sleep(3)
        res["scenarios"] = sc = []
        # 5a. process crash of the active engine
        r = measure_fault(pool, load, "kill")
        r["repair"] = pool.respawn(r_eng(pool, r["from"]), gmu, mml)
        sc.append(r); save()
        # 5b. weight corruption in the active holder (now S on H1): failover to H0, re-copy H1
        r = measure_fault(pool, load, "corrupt")
        r["repair"] = pool.restore_holder(r_eng(pool, r["from"]), pool.active)
        sc.append(r); save()
        if not a.patches:
            sc.append(measure_fault(pool, load, "cuda"))
            sc[-1]["repair"] = pool.respawn(r_eng(pool, sc[-1]["from"]), gmu, mml)
            raise _Done
        # 5c. a rejected patch: standby takes over, patched holder restored
        t_f = time.monotonic()
        victim = pool.active
        ev = pool.evaluate(Patch("bad-zero-mlp", r"layers\.(1[0-9])\.mlp\.down_proj", "scale", 0.0))
        s = load.first_after(t_f, pool.active.name)
        ev["fault_to_first_token_s"] = (s["t_first_token"] - t_f) if s else None
        ev.update(correctness(pool))
        ev["from"] = victim.name
        sc.append({"fault": "patch_rejected", **ev}); save()
        # 5d. bug (a): a passing patch left unaccepted, then another evaluate
        e1 = pool.evaluate(GOOD[0])
        res["accept_gate"] = {"first": e1}
        save()
        e2 = pool.evaluate(GOOD[1])
        res["accept_gate"]["second"] = e2
        save()
        acc = pool.accept() if pool.pending else {"accepted": None, "why": "second not pending"}
        res["accept_gate"].update({"accept": acc,
                              "digest_after": {"active": pool.check_integrity()["ok"],
                                               "standby": pool.check_integrity(pool.standby)["ok"]},
                              "ledger": pool.ledger.ids})
        save()
        # 5e. sticky CUDA error in the active engine
        r = measure_fault(pool, load, "cuda")
        r["repair"] = pool.respawn(r_eng(pool, r["from"]), gmu, mml)
        sc.append(r)
        raise _Done
    except _Done:
        load.stop()
        res["load"] = {"requests": len(load.samples), "errors": len(load.errors),
                       "errors_sample": load.errors[:5]}
        res["events"] = [{k: v for k, v in e.items() if k != "copy"} for e in pool.events]
        pool.close()
        save()
    finally:
        for e in eng_list:
            try:
                e.stop()
            except Exception:  # noqa: BLE001
                e.kill()
        for h in holders:
            h.stop()
        save()
    print(json.dumps({"ok": True, "result": os.path.join(work, "result.json")}))
    return 0


# Benign weight patches: exact reparameterizations (bytes change, outputs must not). Measured:
# even 1e-5 * std noise on one down_proj flips a greedy token of Qwen3-8B (gate_noise_probe.py),
# so noise is no benign patch for a 16-token greedy gate.
GOOD = (Patch("good-1", r"layers\.3\.mlp\.(gate_up|down)_proj", "mlp_rescale", 2.0),
        Patch("good-2", r"layers\.5\.mlp\.(gate_up|down)_proj", "mlp_rescale", 4.0))


class _Done(Exception):
    pass


def r_eng(pool, name):
    return pool.active if pool.active.name == name else pool.standby


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy-rank")
    c.add_argument("--rank", type=int, required=True)
    c.add_argument("--init", required=True)
    demo_parser = sub.add_parser("demo")
    for subparser in (c, sub.add_parser("copy"), demo_parser):
        subparser.add_argument("--model", required=True)
        # copy and copy-rank: unset means the holders were started without --max-model-len
        subparser.add_argument("--max-model-len", type=int,
                               default=4096 if subparser is demo_parser else None)
        subparser.add_argument("--python", default=sys.executable)
    cp = sub._name_parser_map["copy"]
    cp.add_argument("--socket-dir", required=True)
    cp.add_argument("--src-gpu", type=int, required=True)
    cp.add_argument("--dst-gpu", type=int, required=True)
    c.add_argument("--socket-dir", required=True)
    d = sub._name_parser_map["demo"]
    d.add_argument("--gpus", required=True, help="three GPUs: holder0/active, holder1/standby, cold")
    d.add_argument("--work-dir", required=True)
    d.add_argument("--gmu", type=float, default=0.35)
    d.add_argument("--port-base", type=int, default=8751)
    d.add_argument("--tol", type=float, default=0.05)
    d.add_argument("--no-patches", dest="patches", action="store_false",
                   help="skip the weight-patch scenarios (their targets are dense-model names)")
    return p


def main(argv=None):
    sys.path.insert(0, str(HERE))
    a = parser().parse_args(argv)
    if a.cmd == "copy-rank":
        return copy_rank(a)
    if a.cmd == "copy":
        r = copy(a.model, a.socket_dir, a.src_gpu, a.dst_gpu, a.max_model_len, a.python)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    return demo(a)


if __name__ == "__main__":
    sys.exit(main())
