"""S7: Per-GPU Region-Granular Kernel Exclusivity Service (`KernelService`, pure stdlib).

Allows K concurrent workers sharing a GPU to compile, generate inputs, and verify off-region
with warm CUDA contexts, while serializing timed critical regions with sibling GPU drain,
delegated `cgroup2` or verified `(pgid, sid, start_ticks)` `SIGSTOP`/`SIGCONT` quiesce,
NUMA-local CPU pinning, and L2 cache flush.
"""
from __future__ import annotations
import contextlib, inspect, math, os, signal, threading, time
from dataclasses import dataclass
from typing import Any, Callable, Iterator
try: import fcntl
except ImportError: fcntl = None  # pragma: no cover
from ..ipc import atomic_write_json, read_json
from ..proc import descendants, pid_alive

class KernelServiceError(RuntimeError): """Base error raised by `KernelService`."""
class KernelClosedError(KernelServiceError): """Raised fail-closed when operating on a closed `KernelService`."""
class SlotNotFoundError(KernelServiceError): """Raised when referencing an unregistered `(gpu_id, slot_id)`."""
class GateTimeoutError(TimeoutError, KernelServiceError): """Raised when acquiring or draining a `RegionGate` exceeds `timeout_s`."""

@dataclass(frozen=True)
class SlotRegistration:
    """Immutable snapshot of a worker slot registered with `KernelService`."""
    gpu_id: int; slot_id: str; pid: int; pgid: int | None; sid: int | None; start_ticks: int | None; cgroup_path: str | None; alive: bool; frozen: bool; last_epoch: int
@dataclass(frozen=True)
class RegionTicket:
    """Receipt returned on entry to `KernelService.critical_region()`."""
    gpu_id: int; slot_id: str; epoch: int; acquired_at: float; drained_slots: tuple[str, ...]; frozen_slots: tuple[str, ...]; freeze_mode: str; pinned_cpus: tuple[int, ...]
@dataclass
class _SlotEntry:
    gpu_id: int; slot_id: str; pid: int; pgid: int | None; sid: int | None; start_ticks: int | None; cgroup_path: str | None
    drain_fn: Callable[..., Any] | None; freeze_fn: Callable[[], Any] | None; thaw_fn: Callable[[], Any] | None
    frozen: bool = False; freeze_method: str | None = None; last_epoch: int = 0
    def snapshot(self, *, proc_root: str) -> SlotRegistration:
        return SlotRegistration(self.gpu_id, self.slot_id, self.pid, self.pgid, self.sid, self.start_ticks, self.cgroup_path, pid_alive(self.pid, proc_root=proc_root), self.frozen, self.last_epoch)

def _req_int(v: Any, name: str, min_val: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < min_val: raise ValueError(f"{name} must be an int >= {min_val}, got {v!r}")
    return v
def _req_slot_id(v: Any, name: str = "slot_id") -> str:
    if not isinstance(v, str) or not v.strip() or any(bad in v for bad in ("/", "\\", "..", "\x00")):
        raise ValueError(f"{name} must be a safe non-empty identifier, got {v!r}")
    return v
def _req_timeout(v: float | None) -> float | None:
    if v is None: return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v <= 0:
        raise ValueError(f"timeout_s must be finite > 0 or None, got {v!r}")
    return float(v)
def _read_proc_identity(pid: int, *, proc_root: str = "/proc") -> tuple[int | None, int | None, int | None] | None:
    if pid <= 0: return None
    if proc_root and os.path.isdir(proc_root):
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8") as fh: f = fh.read().rsplit(")", 1)[1].split()
            return None if f[0] == "Z" else (int(f[2]), int(f[3]), int(f[19]))
        except (OSError, IndexError, ValueError): return None
    if not pid_alive(pid, proc_root=proc_root): return None
    pgid = sid = None
    with contextlib.suppress(OSError): pgid = os.getpgid(pid)
    if hasattr(os, "getsid"):
        with contextlib.suppress(OSError): sid = os.getsid(pid)
    return pgid, sid, None
def _identity_matches(entry: _SlotEntry, *, proc_root: str = "/proc") -> bool:
    ident = _read_proc_identity(entry.pid, proc_root=proc_root)
    return ident is not None and not any(exp is not None and cur is not None and cur != exp for cur, exp in zip(ident, (entry.pgid, entry.sid, entry.start_ticks)))
def _read_proc_cgroup(pid: int, *, proc_root: str = "/proc", cgroup_root: str | None = "/sys/fs/cgroup") -> str | None:
    if not (proc_root and cgroup_root and os.path.isdir(cgroup_root)): return None
    with contextlib.suppress(OSError):
        with open(os.path.join(proc_root, str(pid), "cgroup"), encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    cand = os.path.join(cgroup_root, parts[2].lstrip("/")) if parts[2].lstrip("/") else cgroup_root
                    return cand if os.path.isdir(cand) else None
    return None
def _set_cgroup_freeze(cgroup_path: str, freeze: bool, *, timeout_s: float = 1.0) -> bool:
    freeze_file, events_file, target = os.path.join(cgroup_path, "cgroup.freeze"), os.path.join(cgroup_path, "cgroup.events"), ("1" if freeze else "0")
    if not os.path.isfile(freeze_file): return False
    try:
        with open(freeze_file, "w", encoding="utf-8") as fh: fh.write(target)
    except OSError: return False
    if not os.path.isfile(events_file): return True
    want, deadline, ok = f"frozen {target}", time.monotonic() + max(0.05, timeout_s), False
    try:
        while time.monotonic() < deadline:
            with open(events_file, encoding="utf-8") as fh:
                if want in fh.read(): ok = True; return True
            time.sleep(0.005)
    except OSError: pass
    finally:
        if freeze and not ok:
            with contextlib.suppress(OSError):
                with open(freeze_file, "w", encoding="utf-8") as fh: fh.write("0")
    return False
def select_numa_cpu_slice(
    gpu_id: int, *, node_root: str = "/sys/devices/system/node",
    allowed_cpus: set[int] | frozenset[int] | None = None, gpus_per_node: int | None = None, cores_per_gpu: int = 1,
) -> tuple[int, ...]:
    """Partition NUMA-local CPUs into per-GPU slices for timed-loop thread pinning."""
    _req_int(gpu_id, "gpu_id"); _req_int(cores_per_gpu, "cores_per_gpu", 1)
    if gpus_per_node is not None: _req_int(gpus_per_node, "gpus_per_node", 1)
    allowed = allowed_cpus
    if allowed is None and hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError): allowed = frozenset(os.sched_getaffinity(0))
    nodes: list[tuple[int, ...]] = []
    if node_root and os.path.isdir(node_root):
        for _, cpulist in sorted((int(n[4:]), os.path.join(node_root, n, "cpulist")) for n in os.listdir(node_root) if n.startswith("node") and n[4:].isdigit()):
            with contextlib.suppress(OSError, ValueError):
                with open(cpulist, encoding="utf-8") as fh:
                    cpus = [c for tok in fh.read().strip().split(",") if tok.strip() for c in (range(int(tok.split("-", 1)[0]), int(tok.split("-", 1)[1]) + 1) if "-" in tok else (int(tok),))]
                if parsed := tuple(sorted({c for c in cpus if allowed is None or c in allowed})): nodes.append(parsed)
    if not nodes:
        if not allowed: return ()
        nodes = [tuple(sorted(allowed))]
    per_node = gpus_per_node
    if per_node is None:
        if os.path.isdir("/proc/driver/nvidia/gpus"):
            with contextlib.suppress(OSError):
                if cnt := len(os.listdir("/proc/driver/nvidia/gpus")): per_node = max(1, cnt // len(nodes))
        if per_node is None: per_node = 4 if (len(nodes) > 1 and len(nodes[0]) >= 8) else 1
    node_cpus = nodes[(gpu_id // per_node) % len(nodes)]; count = min(cores_per_gpu, len(node_cpus))
    start = ((gpu_id % per_node) * count) % len(node_cpus)
    return tuple(node_cpus[(start + i) % len(node_cpus)] for i in range(count))
def _read_json_quiet(path: str) -> dict[str, Any] | None:
    try: data = read_json(path); return data if isinstance(data, dict) else None
    except (OSError, ValueError): return None

class RegionGate:
    """Per-GPU mutual-exclusion and sibling submission gate with monotonic epochs."""
    def __init__(self, gpu_id: int, *, state_dir: str | None = None) -> None:
        self.gpu_id, self._state_dir = _req_int(gpu_id, "gpu_id"), state_dir
        self._region_lock, self._state_lock = threading.Lock(), threading.RLock()
        self._cond = threading.Condition(self._state_lock)
        self._epoch: int = 0; self._blocked: bool = False; self._holder_slot_id: str | None = None
        self._holder_thread_id: int | None = None; self._acks: dict[str, int] = {}
        self._gate_path = os.path.join(state_dir, f"gpu_{gpu_id}.gate.json") if state_dir else None
        self._lock_path = os.path.join(state_dir, f"gpu_{gpu_id}.lock") if state_dir else None
    def _read_shared_state(self) -> dict[str, Any] | None: return _read_json_quiet(self._gate_path) if self._gate_path else None
    def _state_val(self, key: str, local: Any) -> Any:
        s = self._read_shared_state(); return s[key] if s is not None and key in s else local
    @property
    def epoch(self) -> int: return int(self._state_val("epoch", self._epoch))
    @property
    def blocked(self) -> bool: return bool(self._state_val("blocked", self._blocked))
    @property
    def holder_slot_id(self) -> str | None: return self._state_val("holder_slot_id", self._holder_slot_id)
    def _write_shared_state(self, holder_pid: int | None, holder_ticks: int | None, frozen_slots: list[dict[str, Any]] | None = None) -> None:
        if self._gate_path is not None:
            with contextlib.suppress(OSError):
                atomic_write_json(self._gate_path, {"gpu_id": self.gpu_id, "epoch": self._epoch, "blocked": self._blocked,
                    "holder_slot_id": self._holder_slot_id, "holder_pid": holder_pid, "holder_start_ticks": holder_ticks, "frozen_slots": frozen_slots or []})
    def is_blocked(self) -> bool:
        """True when a critical region currently holds exclusive access to `gpu_id`."""
        with self._state_lock: return self.blocked
    def wait_if_blocked(self, timeout_s: float | None = None) -> bool:
        """Block off-region GPU submission until the active critical region completes."""
        timeout = _req_timeout(timeout_s); deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self.is_blocked():
                rem = None if deadline is None else deadline - time.monotonic()
                if rem is not None and rem <= 0.0: raise GateTimeoutError(f"timed out waiting for GPU {self.gpu_id} gate to unblock")
                wait_for: float = 0.01 if self._gate_path else (0.5 if rem is None else min(rem, 0.5))
                self._cond.wait(timeout=wait_for if rem is None else min(wait_for, rem))
            return True
    def ack_drained(self, slot_id: str, epoch: int) -> None:
        """Record sibling drain acknowledgement matching `epoch`."""
        sid, ep = _req_slot_id(slot_id), _req_int(epoch, "epoch")
        with self._cond:
            self._acks[sid] = ep
            if self._state_dir: atomic_write_json(os.path.join(self._state_dir, f"gpu_{self.gpu_id}.ack.{sid}.json"), {"slot_id": sid, "epoch": ep})
            self._cond.notify_all()
    def has_ack(self, slot_id: str, epoch: int) -> bool:
        with self._state_lock:
            if self._acks.get(slot_id) == epoch: return True
            if self._state_dir and (data := _read_json_quiet(os.path.join(self._state_dir, f"gpu_{self.gpu_id}.ack.{slot_id}.json"))) and data.get("epoch") == epoch:
                self._acks[slot_id] = epoch; return True
            return False

class KernelService:
    """S7 coordinator for per-GPU region-granular exclusivity across warm concurrent slots."""
    def __init__(self, *, state_dir: str | None = None, default_timeout_s: float = 30.0, pin_cpus: bool = True,
                 cores_per_gpu: int = 1, gpus_per_node: int | None = None, proc_root: str = "/proc",
                 cgroup_root: str | None = "/sys/fs/cgroup", node_root: str = "/sys/devices/system/node") -> None:
        if state_dir is not None:
            if not isinstance(state_dir, str) or not state_dir.strip(): raise ValueError("state_dir must be a non-empty string")
            state_dir = os.path.abspath(state_dir); os.makedirs(state_dir, exist_ok=True)
        self._state_dir, self._default_timeout_s = state_dir, _req_timeout(default_timeout_s) or 30.0
        self._pin_cpus, self._cores_per_gpu = bool(pin_cpus), _req_int(cores_per_gpu, "cores_per_gpu", 1)
        self._gpus_per_node = _req_int(gpus_per_node, "gpus_per_node", 1) if gpus_per_node is not None else None
        self._proc_root, self._cgroup_root, self._node_root = proc_root, (os.path.abspath(cgroup_root) if cgroup_root else None), node_root
        self._lock, self._closed = threading.RLock(), False
        self._gates: dict[int, RegionGate] = {}; self._slots: dict[int, dict[str, _SlotEntry]] = {}
        self._owned_slots: set[tuple[int, str]] = set(); self._region_entries = self._recovered_holders = 0
    def __enter__(self) -> KernelService:
        with self._lock: self._require_open_locked(); return self
    def __exit__(self, *_exc: Any) -> None: self.close()
    def _require_open_locked(self) -> None:
        if self._closed: raise KernelClosedError("KernelService is closed")
    def _gate_for_locked(self, gpu_id: int) -> RegionGate:
        return self._gates.setdefault(gpu_id, RegionGate(gpu_id, state_dir=self._state_dir)) if gpu_id not in self._gates else self._gates[gpu_id]
    def get_gate(self, gpu_id: int) -> RegionGate:
        """Return the `RegionGate` managing exclusivity for `gpu_id`."""
        gid = _req_int(gpu_id, "gpu_id")
        with self._lock: self._require_open_locked(); return self._gate_for_locked(gid)
    def register_slot(self, gpu_id: int, slot_id: str, pid: int | None = None, *, pgid: int | None = None, cgroup_path: str | None = None,
                      drain_fn: Callable[..., Any] | None = None, freeze_fn: Callable[[], Any] | None = None, thaw_fn: Callable[[], Any] | None = None) -> SlotRegistration:
        """Register a persistent worker slot sharing `gpu_id`."""
        gid, sid = _req_int(gpu_id, "gpu_id"), _req_slot_id(slot_id)
        rpid = os.getpid() if pid is None else _req_int(pid, "pid", 1)
        for fn, name in ((drain_fn, "drain_fn"), (freeze_fn, "freeze_fn"), (thaw_fn, "thaw_fn")):
            if fn is not None and not callable(fn): raise TypeError(f"{name} must be callable or None")
        ident = _read_proc_identity(rpid, proc_root=self._proc_root)
        rpgid = pgid if pgid is not None else (ident[0] if ident else None)
        if rpgid is not None: _req_int(rpgid, "pgid", 1)
        entry = _SlotEntry(gid, sid, rpid, rpgid, ident[1] if ident else None, ident[2] if ident else None,
                           os.path.abspath(cgroup_path) if cgroup_path is not None else None, drain_fn, freeze_fn, thaw_fn)
        with self._lock:
            self._require_open_locked(); self._gate_for_locked(gid)
            self._slots.setdefault(gid, {})[sid] = entry; self._owned_slots.add((gid, sid))
            if self._state_dir: atomic_write_json(os.path.join(self._state_dir, f"gpu_{gid}.slot.{sid}.json"), {"gpu_id": gid, "slot_id": sid, "pid": rpid, "pgid": rpgid, "sid": entry.sid, "start_ticks": entry.start_ticks, "cgroup_path": entry.cgroup_path})
            return entry.snapshot(proc_root=self._proc_root)
    def _rm_slot_file(self, gpu_id: int, slot_id: str) -> None:
        if self._state_dir:
            with contextlib.suppress(OSError): os.remove(os.path.join(self._state_dir, f"gpu_{gpu_id}.slot.{slot_id}.json"))
    def unregister_slot(self, gpu_id: int, slot_id: str) -> bool:
        """Idempotently remove `(gpu_id, slot_id)`, thawing it cleanly under the GPU gate lock."""
        gid, sid = _req_int(gpu_id, "gpu_id"), _req_slot_id(slot_id)
        with self._lock: gate = self._gate_for_locked(gid)
        same_thread = gate._holder_thread_id == threading.get_ident()
        locked, fh = (True, None) if same_thread else self._lock_gate(gate, self._default_timeout_s)
        try:
            with self._lock: self._owned_slots.discard((gid, sid)); entry = self._slots.get(gid, {}).pop(sid, None)
            self._rm_slot_file(gid, sid)
            if entry is None: return False
            if entry.frozen and not same_thread: self._thaw_entry(entry)
            return True
        finally:
            if not same_thread and locked: self._unlock_gate(gate, fh)
    def _sync_shared_slots_locked(self, gpu_id: int) -> list[str]:
        pruned: list[str] = []
        if not self._state_dir: return pruned
        gpu_slots, prefix, seen_sids = self._slots.setdefault(gpu_id, {}), f"gpu_{gpu_id}.slot.", set()
        with contextlib.suppress(OSError):
            for fname in os.listdir(self._state_dir):
                if not (fname.startswith(prefix) and fname.endswith(".json")): continue
                fpath = os.path.join(self._state_dir, fname); data = _read_json_quiet(fpath)
                if not data or not isinstance(data.get("slot_id"), str) or not isinstance(data.get("pid"), int): continue
                sid = data["slot_id"]
                if sid in gpu_slots and gpu_slots[sid].pid == data["pid"] and gpu_slots[sid].start_ticks == data.get("start_ticks"):
                    seen_sids.add(sid); continue
                ent = _SlotEntry(gpu_id, sid, data["pid"], data.get("pgid"), data.get("sid"), data.get("start_ticks"), data.get("cgroup_path"), None, None, None)
                if _identity_matches(ent, proc_root=self._proc_root): gpu_slots[sid] = ent; seen_sids.add(sid)
                else:
                    with contextlib.suppress(OSError): os.remove(fpath)
                    pruned.append(sid)
            for sid in list(gpu_slots):
                if sid not in seen_sids and (gpu_id, sid) not in self._owned_slots: gpu_slots.pop(sid, None); pruned.append(sid)
        return pruned
    def list_slots(self, gpu_id: int | None = None) -> list[SlotRegistration]:
        """List registered slot snapshots (optionally filtered to `gpu_id`)."""
        with self._lock:
            gids = {_req_int(gpu_id, "gpu_id")} if gpu_id is not None else set(self._slots)
            if gpu_id is None and self._state_dir:
                with contextlib.suppress(OSError):
                    for fn in os.listdir(self._state_dir):
                        if fn.startswith("gpu_") and ".slot." in fn and fn.endswith(".json") and (p := fn[4:fn.index(".slot.")]).isdigit(): gids.add(int(p))
            for g in sorted(gids): self._sync_shared_slots_locked(g)
            entries = (list(self._slots.get(_req_int(gpu_id, "gpu_id"), {}).values()) if gpu_id is not None else [e for g in self._slots.values() for e in g.values()])
        return [e.snapshot(proc_root=self._proc_root) for e in entries]
    def submission_gate(self, gpu_id: int, slot_id: str | None = None, *, timeout_s: float | None = None) -> bool:
        """Wait if `gpu_id` is currently blocked by an active critical region."""
        gid = _req_int(gpu_id, "gpu_id")
        with self._lock:
            self._require_open_locked(); gate = self._gate_for_locked(gid)
            if slot_id is not None and gate.holder_slot_id == slot_id: return True
        return gate.wait_if_blocked(timeout_s=self._default_timeout_s if timeout_s is None else timeout_s)
    def _stale_shared_holder(self, gate: RegionGate, gpu_slots: dict[str, _SlotEntry] | None = None) -> bool:
        s = gate._read_shared_state()
        if s is not None:
            if not s.get("blocked") or s.get("holder_pid") is None: return False
            ident, hticks = _read_proc_identity(int(s["holder_pid"]), proc_root=self._proc_root), s.get("holder_start_ticks")
            return ident is None or (hticks is not None and ident[2] is not None and ident[2] != hticks)
        return gate._blocked and (gate._holder_slot_id is None or gpu_slots is None or gate._holder_slot_id not in gpu_slots or not _identity_matches(gpu_slots[gate._holder_slot_id], proc_root=self._proc_root))
    def _recover_gate_locked(self, gid: int, gate: RegionGate, slot_id: str | None = None) -> tuple[list[str], list[str], bool]:
        thawed: list[str] = []
        with gate._cond:
            removed = [s for s in self._sync_shared_slots_locked(gid) if slot_id is None or s == slot_id]
            gpu_slots, shared = self._slots.get(gid, {}), gate._read_shared_state()
            hs = shared.get("holder_slot_id") if shared is not None else gate._holder_slot_id
            stale = self._stale_shared_holder(gate, gpu_slots) or (slot_id is not None and hs == slot_id and self._state_dir is None and slot_id in gpu_slots)
            for sid in ([slot_id] if slot_id is not None else [s for s, e in list(gpu_slots.items()) if not _identity_matches(e, proc_root=self._proc_root)]):
                ent = gpu_slots.get(sid)
                if ent is not None and (not _identity_matches(ent, proc_root=self._proc_root) or (sid == hs and stale) or self._state_dir is None):
                    gpu_slots.pop(sid, None); self._rm_slot_file(gid, sid)
                    if sid not in removed: removed.append(sid)
            if stale or (slot_id is not None and hs == slot_id and hs not in gpu_slots and not self._state_dir):
                for ent in gpu_slots.values():
                    if ent.frozen: self._thaw_entry(ent); thawed.append(ent.slot_id)
                for rec in (shared or {}).get("frozen_slots") or ():
                    if isinstance(rec, dict) and rec.get("slot_id") and str(rec["slot_id"]) not in thawed:
                        self._thaw_entry(_SlotEntry(gid, str(rec["slot_id"]), int(rec.get("pid") or 0), rec.get("pgid"), rec.get("sid"), rec.get("start_ticks"), rec.get("cgroup_path"), None, None, None, frozen=True, freeze_method=rec.get("freeze_method"))); thawed.append(str(rec["slot_id"]))
                gate._epoch = max(gate._epoch, int(shared["epoch"]) if shared else 0); gate._blocked, gate._holder_slot_id = False, None; gate._write_shared_state(None, None, []); gate._cond.notify_all()
                self._recovered_holders += 1; stale = True
        return removed, thawed, stale
    def holder_died(self, gpu_id: int, slot_id: str | None = None) -> dict[str, Any]:
        """Idempotently clean dead slots and recover any stuck gate held by a dead holder."""
        gid = _req_int(gpu_id, "gpu_id")
        if slot_id is not None: _req_slot_id(slot_id)
        with self._lock: gate = self._gate_for_locked(gid)
        ok, fh = self._lock_gate(gate, None, blocking=False)
        if not ok: return {"gpu_id": gid, "removed_slots": [], "thawed_slots": [], "recovered_gate": False}
        try:
            with self._lock: removed, thawed, stale = self._recover_gate_locked(gid, gate, slot_id)
            return {"gpu_id": gid, "removed_slots": removed, "thawed_slots": thawed, "recovered_gate": stale}
        finally: self._unlock_gate(gate, fh)
    def _prune_dead_siblings_locked(self, gpu_id: int, active_slot_id: str) -> list[_SlotEntry]:
        self._sync_shared_slots_locked(gpu_id); gpu_slots, alive = self._slots.get(gpu_id, {}), []
        for sid, ent in list(gpu_slots.items()):
            if sid != active_slot_id:
                if _identity_matches(ent, proc_root=self._proc_root): alive.append(ent)
                else: gpu_slots.pop(sid, None); self._rm_slot_file(gpu_id, sid)
        return alive
    def _freeze_entry(self, entry: _SlotEntry, *, holder_pid: int, holder_pgid: int | None, timeout_s: float) -> str:
        if entry.freeze_fn is not None: entry.freeze_fn(); entry.frozen, entry.freeze_method = True, "callback"; return "callback"
        if entry.cgroup_path is not None:
            hcg = _read_proc_cgroup(holder_pid, proc_root=self._proc_root, cgroup_root=self._cgroup_root)
            if not (hcg is not None and os.path.commonpath([entry.cgroup_path, hcg]) == entry.cgroup_path) and _set_cgroup_freeze(entry.cgroup_path, True, timeout_s=min(timeout_s, 1.0)):
                entry.frozen, entry.freeze_method = True, "cgroup2"; return "cgroup2"
        if entry.pid == holder_pid or not _identity_matches(entry, proc_root=self._proc_root): return "none"
        if entry.pgid is not None and holder_pgid is not None and entry.pgid != holder_pgid:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(entry.pgid, signal.SIGSTOP); entry.frozen, entry.freeze_method = True, "sigstop_pg"; return "sigstop_pg"
        signaled = False
        for target in [p for p in descendants(entry.pid, proc_root=self._proc_root) if p != holder_pid] + [entry.pid]:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError): os.kill(target, signal.SIGSTOP); signaled = True
        if signaled: entry.frozen, entry.freeze_method = True, "sigstop_pid"; return "sigstop_pid"
        return "none"
    def _thaw_entry(self, entry: _SlotEntry) -> None:
        method, entry.frozen, entry.freeze_method = entry.freeze_method, False, None
        if method == "callback" and entry.thaw_fn is not None:
            with contextlib.suppress(Exception): entry.thaw_fn()
        elif method == "cgroup2" and entry.cgroup_path is not None: _set_cgroup_freeze(entry.cgroup_path, False, timeout_s=1.0)
        elif _identity_matches(entry, proc_root=self._proc_root):
            for t, fn in ([(entry.pgid, os.killpg)] if (method == "sigstop_pg" and entry.pgid is not None)
                          else ([(p, os.kill) for p in [entry.pid] + descendants(entry.pid, proc_root=self._proc_root)] if method == "sigstop_pid" else [])):
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError): fn(t, signal.SIGCONT)
    def _lock_gate(self, gate: RegionGate, timeout: float | None, *, blocking: bool = True) -> tuple[bool, Any | None]:
        if gate._holder_thread_id == threading.get_ident():
            if not blocking: return False, None
            raise KernelServiceError(f"re-entrant critical_region() on GPU {gate.gpu_id} is forbidden")
        if not (gate._region_lock.acquire(timeout=timeout) if (blocking and timeout is not None) else gate._region_lock.acquire(blocking=blocking)):
            if not blocking: return False, None
            raise GateTimeoutError(f"timed out acquiring lock for GPU {gate.gpu_id}")
        gate._holder_thread_id, fh, deadline = threading.get_ident(), None, (None if timeout is None else time.monotonic() + timeout)
        try:
            if gate._lock_path and fcntl is not None:
                fh = open(gate._lock_path, "a+b")
                while True:
                    try: fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); break
                    except BlockingIOError:
                        if not blocking: self._unlock_gate(gate, fh); return False, None
                        if deadline is not None and time.monotonic() >= deadline: raise GateTimeoutError(f"timed out acquiring IPC lock for GPU {gate.gpu_id}")
                        time.sleep(0.002)
            return True, fh
        except BaseException:
            self._unlock_gate(gate, fh)
            if not blocking: return False, None
            raise
    @staticmethod
    def _unlock_gate(gate: RegionGate, flock_handle: Any | None) -> None:
        try:
            if flock_handle is not None:
                with contextlib.suppress(OSError):
                    if fcntl is not None: fcntl.flock(flock_handle.fileno(), fcntl.LOCK_UN)
                with contextlib.suppress(OSError): flock_handle.close()
        finally: gate._holder_thread_id = None; gate._region_lock.release()
    def _drain_siblings(self, gate: RegionGate, siblings: list[_SlotEntry], epoch: int, deadline: float) -> list[str]:
        drained: list[str] = []
        for sib in siblings:
            if sib.drain_fn is not None:
                has_arg = False
                with contextlib.suppress(TypeError, ValueError):
                    has_arg = any(p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL) for p in inspect.signature(sib.drain_fn).parameters.values())
                reply = sib.drain_fn(epoch) if has_arg else sib.drain_fn()
                if (isinstance(reply, int) and not isinstance(reply, bool) and reply != epoch) or reply is False or (isinstance(reply, dict) and (not reply.get("ok", True) or (reply.get("epoch") is not None and reply.get("epoch") != epoch))):
                    raise KernelServiceError(f"sibling {sib.slot_id!r} drain failed at epoch {epoch}: {reply!r}")
                sib.last_epoch = epoch; drained.append(sib.slot_id)
            elif self._state_dir and (os.path.exists(os.path.join(self._state_dir, f"gpu_{gate.gpu_id}.ack.{sib.slot_id}.json")) or gate.has_ack(sib.slot_id, epoch)):
                while not gate.has_ack(sib.slot_id, epoch):
                    if time.monotonic() >= deadline: raise GateTimeoutError(f"timed out draining sibling {sib.slot_id!r} at epoch {epoch}")
                    time.sleep(0.002)
                sib.last_epoch = epoch; drained.append(sib.slot_id)
        return drained
    @contextlib.contextmanager
    def critical_region(self, gpu_id: int, slot_id: str, *, sync_gpu_fn: Callable[[], Any] | None = None,
                        flush_l2_fn: Callable[[], Any] | None = None, timeout_s: float | None = None) -> Iterator[RegionTicket]:
        """Execute a timed GPU loop under per-GPU region-granular exclusivity."""
        gid, sid = _req_int(gpu_id, "gpu_id"), _req_slot_id(slot_id)
        timeout = self._default_timeout_s if timeout_s is None else _req_timeout(timeout_s); assert timeout is not None
        with self._lock:
            self._require_open_locked(); gate = self._gate_for_locked(gid)
            if sid not in self._slots.get(gid, {}): raise SlotNotFoundError(f"slot {sid!r} is not registered on GPU {gid}")
        _, flock_handle = self._lock_gate(gate, timeout, blocking=True)
        frozen_entries: list[_SlotEntry] = []; orig_affinity: set[int] | None = None; epoch = 0
        try:
            deadline, hpid = time.monotonic() + timeout, os.getpid()
            hident = _read_proc_identity(hpid, proc_root=self._proc_root); hpgid, hticks = (hident[0], hident[2]) if hident else (None, None)
            with self._lock:
                self._require_open_locked()
                if self._stale_shared_holder(gate, self._slots.get(gid)): self._recover_gate_locked(gid, gate)
                holder_entry = self._slots.get(gid, {}).get(sid)
                if holder_entry is None: raise SlotNotFoundError(f"slot {sid!r} is not registered on GPU {gid}")
                with gate._cond:
                    prev = gate._read_shared_state(); epoch = gate._epoch = max(gate._epoch, int(prev["epoch"]) if prev else 0) + 1
                    gate._blocked, gate._holder_slot_id, holder_entry.last_epoch = True, sid, epoch; gate._write_shared_state(hpid, hticks, []); gate._cond.notify_all()
                siblings = self._prune_dead_siblings_locked(gid, sid)
            drained = self._drain_siblings(gate, siblings, epoch, deadline)
            if sync_gpu_fn is not None: sync_gpu_fn()
            methods: set[str] = set()
            for sib in siblings:
                mode = self._freeze_entry(sib, holder_pid=hpid, holder_pgid=hpgid, timeout_s=max(0.05, deadline - time.monotonic()))
                if mode != "none":
                    frozen_entries.append(sib); methods.add(mode)
                    if self._state_dir:
                        gate._write_shared_state(hpid, hticks, [{"slot_id": e.slot_id, "pid": e.pid, "pgid": e.pgid, "sid": e.sid, "start_ticks": e.start_ticks, "cgroup_path": e.cgroup_path, "freeze_method": e.freeze_method} for e in frozen_entries])
            pinned_cpus: tuple[int, ...] = ()
            if self._pin_cpus and hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
                with contextlib.suppress(OSError):
                    orig_affinity = set(os.sched_getaffinity(0))
                    cand = select_numa_cpu_slice(gid, node_root=self._node_root, allowed_cpus=frozenset(orig_affinity), gpus_per_node=self._gpus_per_node, cores_per_gpu=self._cores_per_gpu)
                    if cand: os.sched_setaffinity(0, set(cand)); pinned_cpus = cand
            if flush_l2_fn is not None: flush_l2_fn()
            self._region_entries += 1
            yield RegionTicket(gid, sid, epoch, time.time(), tuple(drained), tuple(e.slot_id for e in frozen_entries), ("none" if not methods else (next(iter(methods)) if len(methods) == 1 else "mixed")), pinned_cpus)
        finally:
            try:
                if sync_gpu_fn is not None:
                    with contextlib.suppress(Exception): sync_gpu_fn()
                if orig_affinity is not None and hasattr(os, "sched_setaffinity"):
                    with contextlib.suppress(OSError): os.sched_setaffinity(0, orig_affinity)
                for sib in reversed(frozen_entries): self._thaw_entry(sib)  # Thaw BEFORE unblocking gate or releasing lock.
                with self._lock: self._prune_dead_siblings_locked(gid, sid)
                if epoch:
                    with gate._cond: gate._blocked, gate._holder_slot_id = False, None; gate._write_shared_state(None, None, []); gate._cond.notify_all()
            finally: self._unlock_gate(gate, flock_handle)
    def stats(self) -> dict[str, Any]:
        """Return lightweight observability counters and per-GPU slot summaries."""
        with self._lock:
            return {"closed": self._closed, "gpus": sorted(self._gates), "slots_per_gpu": {gid: len(slots) for gid, slots in sorted(self._slots.items())},
                    "epochs": {gid: gate._epoch for gid, gate in sorted(self._gates.items())}, "region_entries": self._region_entries, "recovered_holders": self._recovered_holders}
    def teardown(self) -> dict[str, Any]:
        """Idempotently clean own slots and close service without disturbing live external holders."""
        with self._lock:
            if self._closed: return {"already_closed": True, "thawed": 0}
            self._closed = True; owned, gates = list(self._owned_slots), list(self._gates.values()); self._owned_slots.clear()
        my_pid, thawed = os.getpid(), 0
        for gid, sid in owned: self._rm_slot_file(gid, sid)
        for gate in gates:
            ok, fh = self._lock_gate(gate, None, blocking=False)
            if not ok: continue
            try:
                shared = gate._read_shared_state()
                if shared is not None and shared.get("blocked") and shared.get("holder_pid") is not None and int(shared["holder_pid"]) != my_pid and not self._stale_shared_holder(gate): continue
                for ent in list(self._slots.get(gate.gpu_id, {}).values()):
                    if ent.frozen: self._thaw_entry(ent); thawed += 1
                with gate._cond: gate._epoch = max(gate._epoch, int(shared["epoch"]) if shared else 0); gate._blocked, gate._holder_slot_id = False, None; gate._write_shared_state(None, None, []); gate._cond.notify_all()
            finally: self._unlock_gate(gate, fh)
        return {"already_closed": False, "thawed": thawed}
    def close(self) -> None: self.teardown()

__all__ = ["GateTimeoutError", "KernelClosedError", "KernelService", "KernelServiceError",
           "RegionGate", "RegionTicket", "SlotNotFoundError", "SlotRegistration", "select_numa_cpu_slice"]
