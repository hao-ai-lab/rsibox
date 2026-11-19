"""Versioned C++/CUDA side-library hot-reload (`NATIVE`) for vLLM GPU workers.

Compiles edited `.cu`/`.cpp` translation units plus a generated `reg.cu` defining a unique
Torch library namespace (`_Cv<key>`) into a node-local content-addressed shared library, loads
it via `ctypes.CDLL(..., RTLD_NOW | os.RTLD_LOCAL)` without calling `dlclose`, and rebinds the
target dispatcher op(s) on `torch.library.Library(ns, "IMPL")` inside the worker commit gap.

Undo calls `lib._destroy()` via the kernel journal, restoring the original C++ implementation in
sub-millisecond time without recompiling or unloading shared libraries (already-captured graphs
are handled orthogonally by `R1` recapture in the same step gap).
"""
import contextlib
import ctypes
import fcntl
import glob
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import weakref

from kernel import findings as F

_TORCH_TARGET_VERSION = "0x020B000000000000ULL"
_LOADED_LIBS: dict[str, ctypes.CDLL] = {}
_LIBS_LOCK = threading.Lock()
_MANAGERS: dict[int, "weakref.ref[NativeManager]"] = {}


def find_nvcc() -> str:
    """Locate `nvcc` without requiring it on default `PATH`."""
    candidates = [
        os.environ.get("FASTRSI_NVCC"),
        shutil.which("nvcc"),
    ]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    candidates.append("/usr/local/cuda/bin/nvcc")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True))
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("nvcc not found (set FASTRSI_NVCC, CUDA_HOME, or install CUDA toolkit)")


def torch_paths() -> tuple[str, str]:
    """Return `(torch_include_dir, torch_lib_dir)` lazily."""
    import torch

    root = os.path.dirname(os.path.realpath(torch.__file__))
    return os.path.join(root, "include"), os.path.join(root, "lib")


def detect_arch(arch: str | None = "auto") -> list[str]:
    """Return nvcc `-gencode` arguments for `arch` (`'auto'`, `'100'`, `'100a'`, or `'compute_X,sm_Y'`)."""
    spec = os.environ.get("FASTRSI_NATIVE_ARCH") or arch or "auto"
    if spec == "auto":
        import torch

        major, minor = torch.cuda.get_device_capability()
        sm = f"{major}{minor}"
        return ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]
    if "," in spec and "compute_" in spec and "sm_" in spec:
        return ["-gencode", spec]
    clean = spec.removeprefix("sm_").removeprefix("compute_")
    return ["-gencode", f"arch=compute_{clean},code=sm_{clean}"]


def _validate_rel_path(rel: str) -> str:
    """Validate that `rel` is a normalized relative subpath with no `/` or `..` escape."""
    if not isinstance(rel, str) or not rel or "\\" in rel or os.path.isabs(rel):
        raise ValueError(f"invalid relative path in native spec: {rel!r}")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid relative path in native spec: {rel!r}")
    return rel


def _safe_stage_path(root: str, rel: str) -> str:
    """Resolve `rel` under `root`, rejecting any lexical or symlink escape."""
    clean = _validate_rel_path(rel)
    root_real = os.path.realpath(root)
    dst_real = os.path.realpath(os.path.join(root_real, *clean.split("/")))
    if dst_real == root_real or not dst_real.startswith(root_real + os.sep):
        raise ValueError(f"path escapes csrc root: {rel!r}")
    return os.path.join(root, *clean.split("/"))


def spec_key(spec: dict) -> str:
    """Stable content-addressed hash key for a normalized native build spec."""
    files = [_validate_rel_path(rel) for rel in (spec.get("files") or ())]
    sources = {_validate_rel_path(rel): text for rel, text in (spec.get("sources") or {}).items()}
    edits = [[_validate_rel_path(e[0]), *e[1:]] for e in (spec.get("edits") or ())]
    csrc_root = spec.get("csrc_root")
    disk_hashes = {}
    if csrc_root:
        for rel in files:
            if rel in sources:
                continue
            path = _safe_stage_path(csrc_root, rel)
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    disk_hashes[rel] = hashlib.sha256(fh.read()).hexdigest()
    arch = os.environ.get("FASTRSI_NATIVE_ARCH") or spec.get("arch") or "auto"
    payload = {
        "files": files,
        "disk_hashes": disk_hashes,
        "sources": sources,
        "edits": edits,
        "ops": spec.get("ops") or {},
        "headers": list(spec.get("headers") or ["csrc/libtorch_stable/ops.h"]),
        "stable": bool(spec.get("stable", True)),
        "split_compile": int(spec.get("split_compile", 8)),
        "arch": arch,
        "extra_nvcc_flags": list(spec.get("extra_nvcc_flags") or ()),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def generate_registration(key: str, ops: dict, headers=None, stable: bool = True) -> str:
    """Emit `reg.cu` registering `ops` into unique namespace `_Cv<key>`."""
    hdrs = list(headers or ["csrc/libtorch_stable/ops.h"])
    lines = [f'#include "{h}"' for h in hdrs]
    ns = f"_Cv{key}"
    if stable:
        lines.append("#include <torch/csrc/stable/library.h>")
        lines.append(f"STABLE_TORCH_LIBRARY({ns}, ops) {{")
        for op_spec in ops.values():
            lines.append(f'  ops.def("{op_spec["schema"]}");')
        lines.append("}")
        lines.append(f"STABLE_TORCH_LIBRARY_IMPL({ns}, CUDA, ops) {{")
        for op_name, op_spec in ops.items():
            lines.append(f'  ops.impl("{op_name}", TORCH_BOX(&{op_spec["fn"]}));')
        lines.append("}")
    else:
        lines.append("#include <torch/library.h>")
        lines.append(f"TORCH_LIBRARY({ns}, ops) {{")
        for op_spec in ops.values():
            lines.append(f'  ops.def("{op_spec["schema"]}");')
        lines.append("}")
        lines.append(f"TORCH_LIBRARY_IMPL({ns}, CUDA, ops) {{")
        for op_name, op_spec in ops.items():
            lines.append(f'  ops.impl("{op_name}", &{op_spec["fn"]});')
        lines.append("}")
    return "\n".join(lines) + "\n"


def _apply_source_changes(csrc_stage: str, spec: dict) -> None:
    """Apply `spec['sources']` and `spec['edits']` inside staged `csrc/` directory."""
    for rel, content in (spec.get("sources") or {}).items():
        dst = _safe_stage_path(csrc_stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(content)
    for rel, old, new in spec.get("edits") or ():
        dst = _safe_stage_path(csrc_stage, rel)
        with open(dst, "r", encoding="utf-8") as fh:
            text = fh.read()
        if old not in text:
            raise ValueError(f"edit snippet not found in {rel}: {old[:80]!r}")
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text.replace(old, new, 1))


def build_native(spec: dict) -> dict:
    """Compile `spec` on node-local storage with `flock` sharing across worker ranks."""
    csrc_root = os.path.abspath(spec["csrc_root"])
    if not os.path.isdir(csrc_root):
        raise FileNotFoundError(f"csrc_root not found: {csrc_root}")
    files = [_validate_rel_path(rel) for rel in (spec.get("files") or ())]
    ops = dict(spec.get("ops") or {})
    if not files or not ops:
        raise ValueError("native spec requires non-empty 'files' and 'ops'")

    key = spec_key(spec)
    build_root = spec.get("build_root") or os.environ.get(
        "FASTRSI_NATIVE_BUILD_ROOT", f"/tmp/fastrsi-native-{os.getuid()}"
    )
    build_dir = os.path.join(build_root, key)
    os.makedirs(build_dir, exist_ok=True)
    final_so = os.path.join(build_dir, f"libfastrsi_{key}.so")
    lock_path = os.path.join(build_dir, "build.lock")
    use_cache = bool(spec.get("cache", True))

    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if use_cache and os.path.isfile(final_so):
            return {"key": key, "so": final_so, "build_s": 0.0, "cached": True}

        t0 = time.perf_counter()
        nvcc = find_nvcc()
        torch_inc, torch_lib = torch_paths()
        arch_flags = detect_arch(spec.get("arch", "auto"))
        stage_dir = os.path.join(build_dir, f"stage-{os.getpid()}")
        shutil.rmtree(stage_dir, ignore_errors=True)
        try:
            csrc_stage = os.path.join(stage_dir, "csrc")
            shutil.copytree(csrc_root, csrc_stage)
            _apply_source_changes(csrc_stage, spec)
            with open(os.path.join(stage_dir, "reg.cu"), "w", encoding="utf-8") as fh:
                fh.write(generate_registration(key, ops, spec.get("headers"), spec.get("stable", True)))

            flags = [
                "-std=c++20",
                "-O2",
                "-DNDEBUG",
                "-Xcompiler=-fPIC",
                "-Xcompiler=-fvisibility=hidden",
                "-DUSE_CUDA",
                "-DENABLE_FP8",
                f"-DTORCH_TARGET_VERSION={_TORCH_TARGET_VERSION}",
                "-DTORCH_EXTENSION_NAME=_C",
                "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
                *arch_flags,
                f"-I{torch_inc}",
                "-I.",
                "-Icsrc",
                "-Icsrc/libtorch_stable",
                "-w",
            ]
            split = int(spec.get("split_compile", 8))
            if split > 0:
                flags.append(f"-split-compile={split}")
            threads = int(spec.get("threads", 8))
            if threads > 0:
                flags.append(f"--threads={threads}")
            flags.extend(str(f) for f in (spec.get("extra_nvcc_flags") or ()))

            objs = []
            for rel in files:
                _safe_stage_path(csrc_stage, rel)
            for idx, tu in enumerate([f"csrc/{rel}" for rel in files] + ["reg.cu"]):
                obj = f"o{idx}.o"
                objs.append(obj)
                proc = subprocess.run(
                    [nvcc, *flags, "-c", tu, "-o", obj],
                    cwd=stage_dir,
                    capture_output=True,
                    text=True,
                    timeout=int(spec.get("compile_timeout_s", 300)),
                )
                if proc.returncode != 0:
                    msg = (proc.stderr or proc.stdout or "").strip()[-1200:]
                    raise RuntimeError(f"nvcc compile failed for {tu}: {msg}")

            tmp_so = os.path.join(build_dir, f"libfastrsi_{key}.so.tmp.{os.getpid()}")
            with contextlib.suppress(OSError):
                os.unlink(tmp_so)
            link_cmd = [
                nvcc,
                "-shared",
                "-Xlinker=-Bsymbolic",
                "-o",
                tmp_so,
                *objs,
                f"-L{torch_lib}",
                "-ltorch_cpu",
                "-ltorch",
                "-lc10",
                "-lcuda",
                f"-Xlinker=-rpath={torch_lib}",
            ]
            proc = subprocess.run(
                link_cmd,
                cwd=stage_dir,
                capture_output=True,
                text=True,
                timeout=int(spec.get("link_timeout_s", 120)),
            )
            if proc.returncode != 0:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_so)
                msg = (proc.stderr or proc.stdout or "").strip()[-1200:]
                raise RuntimeError(f"nvcc link failed: {msg}")
            os.replace(tmp_so, final_so)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

        return {"key": key, "so": final_so, "build_s": round(time.perf_counter() - t0, 6), "cached": False}


def load_library(so_path: str, key: str) -> dict:
    """Load `so_path` once per process with `RTLD_NOW | RTLD_LOCAL` (never `dlclose`)."""
    with _LIBS_LOCK:
        if key in _LOADED_LIBS:
            return {"key": key, "load_s": 0.0, "already_loaded": True}
        t0 = time.perf_counter()
        mode = getattr(os, "RTLD_NOW", ctypes.DEFAULT_MODE) | getattr(os, "RTLD_LOCAL", 0)
        _LOADED_LIBS[key] = ctypes.CDLL(so_path, mode=mode)
        return {"key": key, "load_s": round(time.perf_counter() - t0, 6), "already_loaded": False}


def _resolve_op(namespace: str, op_name: str):
    import torch

    ns_obj = getattr(torch.ops, namespace)
    return getattr(ns_obj, op_name)


def _witness_key(op_spec: dict, op_name: str, key: str) -> str:
    ns = op_spec.get("namespace", "_C")
    return f"native:{ns}.{op_name}:{key}"


def _resolve_module_attr(spec: str):
    mod_name, sep, attr = spec.partition(":")
    if not sep or not mod_name or not attr:
        raise ValueError(f"rebind target must be 'module.path:attr', got {spec!r}")
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    return mod, attr


def commit_txn(mgr_id: int, txn: str, journal, arm) -> None:
    """Repair-cell entrypoint invoked inside the worker commit gap (`LocalParticipant.commit`)."""
    import torch

    ref = _MANAGERS.get(mgr_id)
    mgr = ref() if ref is not None else None
    if mgr is None:
        raise RuntimeError(f"NativeManager {mgr_id} is no longer alive")
    st = mgr._staged.pop(txn, None)
    if st is None:
        raise RuntimeError(f"no staged native state for {txn}")

    key = st["key"]
    ops = st["ops"]
    ver_ns = f"_Cv{key}"
    bound_libs = []
    try:
        for op_name, op_spec in ops.items():
            ns = op_spec.get("namespace", "_C")
            dispatch_key = op_spec.get("dispatch_key", "CUDA")
            wkey = _witness_key(op_spec, op_name, key)
            hit = arm(wkey)
            new_op = _resolve_op(ver_ns, op_name)

            def _make_wrapper(fn, counter):
                def _native_op_wrapper(*args, **kwargs):
                    counter()
                    return fn(*args, **kwargs)
                return _native_op_wrapper

            wrapper = _make_wrapper(new_op, hit)
            lib = torch.library.Library(ns, "IMPL")
            lib.impl(op_name, wrapper, dispatch_key)
            bound_libs.append(lib)
            journal.record_call(lib._destroy, f"native.unbind {ns}::{op_name} ({key})")

            for target in op_spec.get("rebind") or ():
                obj, attr = _resolve_module_attr(target)
                journal.set_attr(obj, attr, wrapper)
    except Exception:
        for lib in reversed(bound_libs):
            with contextlib.suppress(Exception):
                lib._destroy()
        raise

    mgr._committed[txn] = st
    journal.record_call(lambda t=txn: mgr._committed.pop(t, None), f"native.meta {txn}")


class NativeManager:
    """Per-worker coordinator helper for `plan['native']` 2PC staging, commit, and hashes."""

    def __init__(self):
        self._staged: dict[str, dict] = {}
        self._committed: dict[str, dict] = {}
        _MANAGERS[id(self)] = weakref.ref(self, lambda _r, mid=id(self): _MANAGERS.pop(mid, None))

    def prepare(
        self,
        txn: str,
        spec: dict,
        *,
        graph=None,
        wants_r1: bool = False,
        accept=(),
        participant: str | None = None,
    ) -> dict:
        """Validate `spec`, check graph safety (`R1`), build `.so`, and load versioned ops."""
        ops = dict(spec.get("ops") or {})
        if not ops:
            f = F.finding("patch_invalid", "plan['native'] has no 'ops'", "non-empty ops mapping", subject="native")
            if participant:
                f = {**f, "participant": participant}
            return F.not_ok("native: missing ops", [f], txn=txn, staged={})

        findings = []
        if graph is not None:
            mode = f"cudagraph_mode {graph['cudagraph_mode']}, compile mode {graph['compile_mode']}"
            key_hint = spec_key(spec)
            for op_name, op_spec in ops.items():
                subj = _witness_key(op_spec, op_name, key_hint)
                finding = {
                    **F.finding(
                        "captured_target",
                        (
                            f"the engine runs {mode}; native op {op_spec.get('namespace', '_C')}.{op_name} "
                            + (
                                'the plan asks for R1: the host recaptures the graphs in the commit\'s gap'
                                if wants_r1
                                else 'add "rebuild": ["R1"] to the plan to recapture the graphs in the commit\'s gap, or restart with the edit, or run eager'
                            )
                        ),
                        "a native op re-captured when graphs are active",
                        subject=subj,
                    ),
                    "route": "R1",
                    "today": "T1",
                    **({"advisory": True} if wants_r1 else {}),
                }
                if participant:
                    finding["participant"] = participant
                findings.append(finding)
        decision = F.decide(findings, accept)
        if not decision["proceed"]:
            return F.not_ok(f"vote no: {decision['blocking']}", decision["findings"], txn=txn, staged={})

        try:
            built = build_native(spec)
            loaded = load_library(built["so"], built["key"])
            ver_ns = f"_Cv{built['key']}"
            for op_name in ops:
                _resolve_op(ver_ns, op_name)
        except Exception as exc:
            f = F.finding(
                "patch_invalid",
                f"{type(exc).__name__}: {exc}",
                "native extension builds and registers cleanly",
                subject="native",
            )
            if participant:
                f = {**f, "participant": participant}
            return F.not_ok(f"native build/load failed: {type(exc).__name__}: {exc}", [f], txn=txn, staged={})

        self._staged[txn] = {
            "key": built["key"],
            "so": built["so"],
            "ops": ops,
            "build_s": built["build_s"],
            "load_s": loaded["load_s"],
            "cached": built["cached"],
        }
        return F.ok(
            findings=decision["findings"],
            native={
                "key": built["key"],
                "so": built["so"],
                "build_s": built["build_s"],
                "load_s": loaded["load_s"],
                "cached": built["cached"],
                "ops": sorted(ops),
            },
        )

    def commit_cell(self, txn: str) -> dict:
        """Build the worker-only repair cell invoking `commit_txn` in the step gap."""
        cell = f"from {__name__} import commit_txn\ncommit_txn({id(self)!r}, {txn!r}, journal, arm)"
        return {"cell": cell, "declares": [], "on": ["worker"]}

    def has_staged(self, txn: str) -> bool:
        return txn in self._staged

    def validate(self, txn: str) -> dict:
        """Verify staged `.so` and versioned ops are still ready in the commit gap."""
        st = self._staged.get(txn)
        if st is None:
            return F.not_ok(f"native validate: nothing staged for {txn}")
        if st["key"] not in _LOADED_LIBS:
            return F.not_ok(f"native validate: library {st['key']} is not loaded")
        return F.ok()

    def abort(self, txn: str) -> None:
        self._staged.pop(txn, None)

    def hashes(self, txn: str) -> dict[str, list[str]]:
        st = self._committed.get(txn)
        if not st:
            return {}
        return {_witness_key(op_spec, op_name, st["key"]): [st["key"]] for op_name, op_spec in st["ops"].items()}
