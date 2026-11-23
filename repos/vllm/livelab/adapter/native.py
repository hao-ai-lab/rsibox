"""Versioned C++/CUDA side-library hot-reload (`NATIVE`) for vLLM GPU workers.

Compiles edited `.cu`/`.cpp` translation units plus a generated `reg.cu` defining a unique
Torch library namespace (`_Cv<key>`) into a node-local content-addressed shared library, loads
it via `ctypes.CDLL(..., RTLD_NOW | os.RTLD_LOCAL)` without calling `dlclose`, and rebinds the
target dispatcher op(s) on `torch.library.Library(ns, "IMPL")` inside the worker commit gap.

Undo calls `lib._destroy()` via the kernel journal, restoring the original C++ implementation in
sub-millisecond time without recompiling or unloading shared libraries (already-captured graphs
are handled orthogonally by `R1` recapture in the same step gap).
"""
import concurrent.futures
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
        os.environ.get("RSIBOX_NVCC"),
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
    raise FileNotFoundError("nvcc not found (set RSIBOX_NVCC, CUDA_HOME, or install CUDA toolkit)")


def find_gxx() -> str | None:
    """Locate host `g++` for compiling/linking pure-C++ registration units without `nvcc`."""
    candidates = [
        os.environ.get("RSIBOX_GXX"),
        shutil.which("g++"),
        "/usr/bin/g++",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _cuda_dirs(nvcc: str) -> tuple[str | None, str | None]:
    """Return `(cuda_include_dir, cuda_lib_dir)` adjacent to `nvcc` when present on disk."""
    roots = [
        os.path.dirname(os.path.dirname(os.path.abspath(nvcc))),
        os.path.dirname(os.path.dirname(os.path.realpath(nvcc))),
    ]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        roots.append(os.path.abspath(cuda_home))
    inc_dir = None
    lib_dir = None
    for root in roots:
        if inc_dir is None:
            cand_inc = os.path.join(root, "include")
            if os.path.isfile(os.path.join(cand_inc, "cuda_runtime.h")):
                inc_dir = cand_inc
        if lib_dir is None:
            for sub in ("lib64", "lib"):
                cand_lib = os.path.join(root, sub)
                if os.path.isfile(os.path.join(cand_lib, "libcudart.so")):
                    lib_dir = cand_lib
                    break
    return inc_dir, lib_dir


def torch_paths() -> tuple[str, str]:
    """Return `(torch_include_dir, torch_lib_dir)` lazily."""
    import torch

    root = os.path.dirname(os.path.realpath(torch.__file__))
    return os.path.join(root, "include"), os.path.join(root, "lib")


def detect_arch(arch: str | None = "auto") -> list[str]:
    """Return nvcc `-gencode` arguments for `arch` (`'auto'`, `'100'`, `'100a'`, or `'compute_X,sm_Y'`)."""
    spec = os.environ.get("RSIBOX_NATIVE_ARCH") or arch or "auto"
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


def _safe_stage_path(root: str, rel: str, *, allowed_roots=None) -> str:
    """Resolve `rel` under `root`, rejecting any lexical or symlink escape."""
    clean = _validate_rel_path(rel)
    root_real = os.path.realpath(root)
    dst_real = os.path.realpath(os.path.join(root_real, *clean.split("/")))
    roots = [os.path.realpath(r) for r in (allowed_roots if allowed_roots is not None else (root_real,))]
    if not any(dst_real != r and dst_real.startswith(r + os.sep) for r in roots):
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
    arch = os.environ.get("RSIBOX_NATIVE_ARCH") or spec.get("arch") or "auto"
    payload = {
        "files": files,
        "disk_hashes": disk_hashes,
        "sources": sources,
        "edits": edits,
        "ops": spec.get("ops") or {},
        "headers": list(spec.get("headers") or ["csrc/libtorch_stable/ops.h"]),
        "stable": bool(spec.get("stable", True)),
        "split_compile": int(spec.get("split_compile", 16)),
        "threads": int(spec.get("threads", 16)),
        "arch": arch,
        "extra_nvcc_flags": list(spec.get("extra_nvcc_flags") or ()),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def generate_registration(key: str, ops: dict, headers=None, stable: bool = True) -> str:
    """Emit `reg.cu` registering `ops` into unique namespace `_Cv<key>`."""
    hdrs = list(headers or ["csrc/libtorch_stable/ops.h"])
    ns = f"_Cv{key}"
    inc = "<torch/csrc/stable/library.h>" if stable else "<torch/library.h>"
    prefix = "STABLE_TORCH_LIBRARY" if stable else "TORCH_LIBRARY"
    lines = [f'#include "{h}"' for h in hdrs] + [f"#include {inc}", f"{prefix}({ns}, ops) {{"]
    lines.extend(f'  ops.def("{op_spec["schema"]}");' for op_spec in ops.values())
    lines.extend(["}", f"{prefix}_IMPL({ns}, CUDA, ops) {{"])
    for op_name, op_spec in ops.items():
        fn_ref = f'TORCH_BOX(&{op_spec["fn"]})' if stable else f'&{op_spec["fn"]}'
        lines.append(f'  ops.impl("{op_name}", {fn_ref});')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _stage_csrc_overlay(
    csrc_root: str,
    csrc_stage: str,
    modified_rels,
    *,
    source_rels=(),
) -> None:
    """Stage `csrc_stage` as a symlink overlay over `csrc_root`.

    Creates real directories for every subdirectory under `csrc_root` (plus any new
    directories required by `modified_rels`), symlinks only regular files (never
    directories) to `os.path.realpath(...)` so `#include "../..."` from any subdirectory
    resolves inside `csrc_stage`, materializes regular non-symlink copies for edited
    files (`modified_rels - source_rels`), and leaves `source_rels` paths uncreated
    until `_apply_source_changes()` writes fresh files.
    """
    csrc_real = os.path.realpath(csrc_root)
    source_set = {_validate_rel_path(rel) for rel in (source_rels or ())}
    mod_set = {_validate_rel_path(rel) for rel in (modified_rels or ())} | source_set
    copy_set = mod_set - source_set

    needed_dirs: set[str] = {""}
    for rel in mod_set:
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            needed_dirs.add("/".join(parts[:depth]))

    def _populate(rel_dir: str, src_dir: str, ancestors: frozenset[tuple[int, int]]) -> None:
        dst_dir = os.path.join(csrc_stage, *rel_dir.split("/")) if rel_dir else csrc_stage
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.isdir(src_dir):
            return
        with os.scandir(src_dir) as it:
            entries = sorted(it, key=lambda e: e.name)
        for entry in entries:
            entry_rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            dst_entry = os.path.join(dst_dir, entry.name)
            is_dir = entry.is_dir(follow_symlinks=False) or (entry.is_symlink() and os.path.isdir(entry.path))
            if entry_rel in needed_dirs:
                _safe_stage_path(csrc_real, entry_rel)
                if not is_dir:
                    raise ValueError(f"cannot create overlay directory over non-directory: {entry_rel!r}")
            if is_dir:
                if entry_rel in copy_set or entry_rel in source_set:
                    raise ValueError(f"cannot edit directory path in native spec: {entry_rel!r}")
                real_sub = os.path.realpath(entry.path)
                st = os.stat(real_sub)
                dir_id = (st.st_dev, st.st_ino)
                if dir_id in ancestors:
                    continue
                _populate(entry_rel, real_sub, ancestors | {dir_id})
            elif entry_rel in source_set:
                continue
            elif entry_rel in copy_set:
                src_safe = _safe_stage_path(csrc_real, entry_rel)
                shutil.copy2(os.path.realpath(src_safe), dst_entry)
                if not os.access(dst_entry, os.W_OK):
                    os.chmod(dst_entry, os.stat(dst_entry).st_mode | 0o200)
            else:
                os.symlink(os.path.realpath(entry.path), dst_entry)

    root_st = os.stat(csrc_real)
    _populate("", csrc_real, frozenset({(root_st.st_dev, root_st.st_ino)}))
    for rel_dir in sorted(needed_dirs):
        if rel_dir:
            os.makedirs(os.path.join(csrc_stage, *rel_dir.split("/")), exist_ok=True)


def _apply_source_changes(csrc_stage: str, spec: dict) -> None:
    """Apply `spec['sources']` and `spec['edits']` inside staged `csrc/` directory."""
    for rel, content in (spec.get("sources") or {}).items():
        dst = _safe_stage_path(csrc_stage, rel)
        if os.path.islink(dst):
            raise ValueError(f"refusing to write through symlink: {rel!r}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(content)
    for rel, old, new in spec.get("edits") or ():
        dst = _safe_stage_path(csrc_stage, rel)
        if os.path.islink(dst):
            raise ValueError(f"refusing to edit through symlink: {rel!r}")
        with open(dst, "r", encoding="utf-8") as fh:
            text = fh.read()
        if old not in text:
            raise ValueError(f"edit snippet not found in {rel}: {old[:80]!r}")
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text.replace(old, new, 1))


def _tu_threads(requested_threads: int, num_files: int) -> int:
    """Scale `--threads` across concurrent nvcc translation-unit compiles."""
    req = int(requested_threads)
    if req <= 0:
        return 0
    count = int(num_files)
    if count <= 1:
        return req
    return max(1, req // count)


def _is_portable_include_dir(inc: str, stage_real: str | None) -> bool:
    if not inc or not os.path.isabs(inc):
        return False
    if stage_real is not None:
        inc_real = os.path.realpath(inc)
        if inc_real == stage_real or inc_real.startswith(stage_real + os.sep):
            return False
    return True


def _filter_pch_portable_flags(gxx_flags, stage_dir: str | None = None) -> list[str]:
    """Keep compiler flags portable across stage directories for shared PCH builds."""
    stage_real = os.path.realpath(stage_dir) if stage_dir else None
    flags = [str(f) for f in gxx_flags]
    out: list[str] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag in ("-I", "-isystem"):
            if i + 1 < len(flags):
                inc = flags[i + 1]
                if _is_portable_include_dir(inc, stage_real):
                    out.extend((flag, inc))
                i += 2
            else:
                i += 1
            continue
        if flag.startswith("-I"):
            if _is_portable_include_dir(flag[2:], stage_real):
                out.append(flag)
            i += 1
            continue
        if flag.startswith("-isystem"):
            if _is_portable_include_dir(flag[len("-isystem"):], stage_real):
                out.append(flag)
            i += 1
            continue
        out.append(flag)
        i += 1
    return out


def _include_dirs_from_flags(flags: list[str]) -> list[str]:
    dirs: list[str] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag in ("-I", "-isystem"):
            if i + 1 < len(flags):
                dirs.append(flags[i + 1])
                i += 2
            else:
                i += 1
        elif flag.startswith("-I"):
            dirs.append(flag[2:])
            i += 1
        elif flag.startswith("-isystem"):
            dirs.append(flag[len("-isystem"):])
            i += 1
        else:
            i += 1
    return dirs


def _ensure_reg_pch(
    build_root: str,
    gxx: str,
    gxx_flags: list[str],
    stable: bool,
    enabled: bool = True,
    *,
    stage_dir: str | None = None,
) -> tuple[str | None, float, bool]:
    """Build or reuse a precompiled registration header (`reg.hpp.gch`) under `flock`.

    Never raises: returns `(None, elapsed_s, False)` on disabled or failed PCH builds so
    `build_native` falls back seamlessly to plain header parsing.
    """
    env_flag = os.environ.get("RSIBOX_NATIVE_REG_PCH", "1").strip().lower()
    if not enabled or not gxx or env_flag in ("0", "false", "no", "off"):
        return None, 0.0, False

    t0 = time.perf_counter()
    pch_flags = _filter_pch_portable_flags(gxx_flags, stage_dir=stage_dir)
    header_rel = "torch/csrc/stable/library.h" if stable else "torch/library.h"
    header_src = f"#include <{header_rel}>\n"
    real_gxx = os.path.realpath(gxx)
    try:
        st = os.stat(real_gxx)
        gxx_stat = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        gxx_stat = None
    header_stat = None
    for inc_dir in _include_dirs_from_flags(pch_flags):
        cand = os.path.join(inc_dir, *header_rel.split("/"))
        if os.path.isfile(cand):
            try:
                real_hdr = os.path.realpath(cand)
                hst = os.stat(real_hdr)
                header_stat = {"path": real_hdr, "size": hst.st_size, "mtime_ns": hst.st_mtime_ns}
                break
            except OSError:
                pass
    torch_mod = sys.modules.get("torch")
    torch_version = getattr(torch_mod, "__version__", None) if torch_mod is not None else None
    digest_payload = json.dumps(
        {
            "gxx": real_gxx,
            "gxx_stat": gxx_stat,
            "flags": pch_flags,
            "header": header_src,
            "header_stat": header_stat,
            "torch_version": torch_version,
        },
        sort_keys=True,
    ).encode("utf-8")
    pch_key = hashlib.sha256(digest_payload).hexdigest()[:16]
    pch_dir = os.path.join(build_root, "_pch", f"reg_{pch_key}")
    hpp_path = os.path.join(pch_dir, "reg.hpp")
    gch_path = os.path.join(pch_dir, "reg.hpp.gch")
    lock_path = os.path.join(pch_dir, "reg.lock")
    tmp_gch = os.path.join(pch_dir, f"reg.hpp.gch.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        os.makedirs(pch_dir, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(gch_path) and os.path.isfile(hpp_path):
                return hpp_path, round(time.perf_counter() - t0, 6), True
            try:
                with open(hpp_path, "w", encoding="utf-8") as fh:
                    fh.write(header_src)
                with contextlib.suppress(OSError):
                    os.unlink(tmp_gch)
                proc = subprocess.run(
                    [gxx, *pch_flags, "-x", "c++-header", hpp_path, "-o", tmp_gch],
                    cwd=pch_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode != 0 or not os.path.isfile(tmp_gch):
                    return None, round(time.perf_counter() - t0, 6), False
                os.replace(tmp_gch, gch_path)
                return hpp_path, round(time.perf_counter() - t0, 6), False
            except (OSError, subprocess.SubprocessError):
                return None, round(time.perf_counter() - t0, 6), False
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_gch)
    except OSError:
        return None, round(time.perf_counter() - t0, 6), False


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
        "RSIBOX_NATIVE_BUILD_ROOT", f"/tmp/rsibox-native-{os.getuid()}"
    )
    build_dir = os.path.join(build_root, key)
    os.makedirs(build_dir, exist_ok=True)
    final_so = os.path.join(build_dir, f"librsibox_{key}.so")
    lock_path = os.path.join(build_dir, "build.lock")
    use_cache = bool(spec.get("cache", True))

    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if use_cache and os.path.isfile(final_so):
            return {
                "key": key,
                "so": final_so,
                "build_s": 0.0,
                "cached": True,
                "timings": {
                    "stage_s": 0.0,
                    "pch_s": 0.0,
                    "pch_cached": False,
                    "compile_s": 0.0,
                    "link_s": 0.0,
                    "tu_s": {},
                },
            }

        t0 = time.perf_counter()
        nvcc = find_nvcc()
        gxx = find_gxx()
        cuda_inc, cuda_lib = _cuda_dirs(nvcc)
        torch_inc, torch_lib = torch_paths()
        arch_flags = detect_arch(spec.get("arch", "auto"))
        stage_dir = os.path.join(build_dir, f"stage-{os.getpid()}")
        shutil.rmtree(stage_dir, ignore_errors=True)
        try:
            csrc_stage = os.path.join(stage_dir, "csrc")
            source_rels = {_validate_rel_path(rel) for rel in (spec.get("sources") or ())}
            edit_rels = {_validate_rel_path(e[0]) for e in (spec.get("edits") or ())}
            t_stage = time.perf_counter()
            _stage_csrc_overlay(csrc_root, csrc_stage, source_rels | edit_rels, source_rels=source_rels)
            stage_s = round(time.perf_counter() - t_stage, 6)
            _apply_source_changes(csrc_stage, spec)
            reg_name = "reg.cpp" if gxx else "reg.cu"
            stable = bool(spec.get("stable", True))
            with open(os.path.join(stage_dir, reg_name), "w", encoding="utf-8") as fh:
                fh.write(generate_registration(key, ops, spec.get("headers"), stable))

            common_flags = [
                "-std=c++20", "-DNDEBUG", "-DUSE_CUDA", "-DENABLE_FP8",
                f"-DTORCH_TARGET_VERSION={_TORCH_TARGET_VERSION}", "-DTORCH_EXTENSION_NAME=_C",
                f"-I{torch_inc}", "-I.", "-Icsrc", "-Icsrc/libtorch_stable", "-w",
            ]
            flags = [
                *common_flags, "-O2", "-Xcompiler=-fPIC", "-Xcompiler=-fvisibility=hidden",
                "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1", *arch_flags,
            ]
            if (split := int(spec.get("split_compile", 16))) > 0:
                flags.append(f"-split-compile={split}")
            if (threads := _tu_threads(int(spec.get("threads", 16)), len(files))) > 0:
                flags.append(f"--threads={threads}")
            flags.extend(str(f) for f in (spec.get("extra_nvcc_flags") or ()))

            compile_timeout = int(spec.get("compile_timeout_s", 300))
            jobs, objs = [], []
            for idx, rel in enumerate(files):
                _safe_stage_path(csrc_stage, rel, allowed_roots=(csrc_stage, csrc_root))
                tu, obj = f"csrc/{rel}", f"o{idx}.o"
                objs.append(obj)
                jobs.append((tu, [nvcc, *flags, "-c", tu, "-o", obj]))

            reg_obj = "reg.o"
            objs.append(reg_obj)
            if gxx:
                gxx_flags = [*common_flags, "-O1", "-fPIC", "-fvisibility=hidden",
                             *(["-I" + cuda_inc] if cuda_inc else [])]
                pch_hpp, pch_s, pch_cached = _ensure_reg_pch(
                    build_root, gxx, gxx_flags, stable, enabled=bool(spec.get("pch", True)), stage_dir=stage_dir
                )
                pch_include = ["-include", pch_hpp] if pch_hpp else []
                jobs.append((reg_name, [gxx, *gxx_flags, *pch_include, "-c", reg_name, "-o", reg_obj]))
            else:
                pch_s, pch_cached = 0.0, False
                jobs.append((reg_name, [nvcc, *flags, "-c", reg_name, "-o", reg_obj]))

            def _run_tool(cmd: list[str], label: str, timeout_s: int) -> float:
                t_cmd = time.perf_counter()
                proc = subprocess.run(cmd, cwd=stage_dir, capture_output=True, text=True, timeout=timeout_s)
                if proc.returncode != 0:
                    msg = (proc.stderr or proc.stdout or "").strip()[-1200:]
                    raise RuntimeError(f"{os.path.basename(cmd[0])} {label} failed: {msg}")
                return round(time.perf_counter() - t_cmd, 6)

            tu_s = {}
            t_compile = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                future_to_tu = {
                    pool.submit(_run_tool, cmd, f"compile ({tu_name})", compile_timeout): tu_name
                    for tu_name, cmd in jobs
                }
                for fut in concurrent.futures.as_completed(future_to_tu):
                    tu_s[future_to_tu[fut]] = fut.result()
            compile_s = round(time.perf_counter() - t_compile, 6)

            tmp_so = os.path.join(build_dir, f"librsibox_{key}.so.tmp.{os.getpid()}")
            with contextlib.suppress(OSError):
                os.unlink(tmp_so)
            use_gxx_link = bool(gxx and cuda_lib)
            lp = "-Wl," if use_gxx_link else "-Xlinker="
            link_cmd = [
                gxx if use_gxx_link else nvcc, "-shared", f"{lp}-Bsymbolic", "-o", tmp_so, *objs,
                f"-L{torch_lib}", "-ltorch_cpu", "-ltorch", "-lc10",
                *([f"-L{cuda_lib}", "-lcudart"] if use_gxx_link else []),
                "-lcuda", f"{lp}-rpath={torch_lib}",
            ]
            try:
                link_s = _run_tool(link_cmd, "link", int(spec.get("link_timeout_s", 120)))
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_so)
                raise
            os.replace(tmp_so, final_so)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

        return {
            "key": key,
            "so": final_so,
            "build_s": round(time.perf_counter() - t0, 6),
            "cached": False,
            "timings": {
                "stage_s": stage_s,
                "pch_s": pch_s,
                "pch_cached": pch_cached,
                "compile_s": compile_s,
                "link_s": link_s,
                "tu_s": tu_s,
            },
        }


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
        self, txn: str, spec: dict, *, graph=None, wants_r1: bool = False,
        accept=(), participant: str | None = None,
    ) -> dict:
        """Validate `spec`, check graph safety (`R1`), build `.so`, and load versioned ops."""
        def _tag(finding: dict, **extra) -> dict:
            return {**finding, **extra, **({"participant": participant} if participant else {})}

        ops = dict(spec.get("ops") or {})
        if not ops:
            f = _tag(F.finding("patch_invalid", "plan['native'] has no 'ops'", "non-empty ops mapping", subject="native"))
            return F.not_ok("native: missing ops", [f], txn=txn, staged={})

        findings = []
        if graph is not None:
            mode = f"cudagraph_mode {graph['cudagraph_mode']}, compile mode {graph['compile_mode']}"
            key_hint = spec_key(spec)
            r1_note = (
                "the plan asks for R1: the host recaptures the graphs in the commit's gap"
                if wants_r1
                else 'add "rebuild": ["R1"] to the plan to recapture the graphs in the commit\'s gap, or restart with the edit, or run eager'
            )
            for op_name, op_spec in ops.items():
                findings.append(_tag(
                    F.finding("captured_target",
                              f"the engine runs {mode}; native op {op_spec.get('namespace', '_C')}.{op_name} {r1_note}",
                              "a native op re-captured when graphs are active",
                              subject=_witness_key(op_spec, op_name, key_hint)),
                    route="R1", today="T1", **({"advisory": True} if wants_r1 else {}),
                ))
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
            f = _tag(F.finding("patch_invalid", f"{type(exc).__name__}: {exc}",
                               "native extension builds and registers cleanly", subject="native"))
            return F.not_ok(f"native build/load failed: {type(exc).__name__}: {exc}", [f], txn=txn, staged={})

        meta = {
            "key": built["key"], "so": built["so"], "build_s": built["build_s"],
            "load_s": loaded["load_s"], "cached": built["cached"],
            **({"timings": built["timings"]} if "timings" in built else {}),
        }
        self._staged[txn] = {**meta, "ops": ops}
        return F.ok(findings=decision["findings"], native={**meta, "ops": sorted(ops)})

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
