"""Shared conformance harness utilities: patch editing, patch/diff resolution, live-source stacks,
JSONL receipt appending, package overlays, fresh subprocess execution, standardized verb refusals,
and canonical kernel fault injection (`fail_apply`, `fail_commit`, `tamper`)."""
import contextlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def edit_once(source: str, old: str, new: str) -> str:
    """Replace `old` with `new` in `source`, asserting the anchor appears exactly once."""
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"anchor count={count} (expected 1): {old!r}")
    return source.replace(old, new)


def module_file(name_or_path: str) -> str | None:
    """Resolve a dotted module name or filesystem path to its `.py` source path, or None."""
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    spec = importlib.util.find_spec(name_or_path)
    return spec.origin if spec and spec.origin else None


def read_module(name_or_path: str) -> str:
    """Read UTF-8 source text of a dotted module name or filesystem path."""
    path = module_file(name_or_path)
    if not path:
        raise FileNotFoundError(f"module or file not found: {name_or_path!r}")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def module_relpath(site_root: str, module_name: str) -> str:
    """Relative path under `site_root` for `module_name`, handling both `__init__.py`
    packages and leaf `.py` modules."""
    parts = module_name.split(".")
    pkg_init = os.path.join(site_root, *parts, "__init__.py")
    if os.path.isfile(pkg_init) or os.path.isdir(os.path.join(site_root, *parts)):
        return os.path.join(*parts, "__init__.py")
    return os.path.join(*parts) + ".py"


def build_package_overlay(
    package_name: str,
    overrides: dict[str, str] | None = None,
    *,
    dst: str | None = None,
    copy_py: bool = False,
    src_dir: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build a symlink (or `.py`-copied) overlay of `package_name` in `dst` with `overrides`
    written in place.

    Handles:
    - `__init__.py` package modules as well as leaf `.py` modules;
    - stripping `__pycache__` and `.pyc` files so stale bytecode is never imported;
    - standalone filesystem paths in `overrides` (written into `dst/<basename>` and returned
      in the second element `{orig_path: overlaid_path}`).

    Returns `(overlay_root, standalone_overrides)`.
    """
    if src_dir is None:
        pkg = importlib.import_module(package_name)
        pkg_file = getattr(pkg, "__file__", None)
        if not pkg_file:
            raise RuntimeError(f"package {package_name!r} has no __file__")
        pkg_dir = os.path.dirname(os.path.abspath(pkg_file))
        site_root = os.path.dirname(pkg_dir)
    else:
        pkg_dir = os.path.abspath(src_dir)
        site_root = os.path.dirname(pkg_dir)

    overlay_root = os.path.abspath(dst) if dst else tempfile.mkdtemp(prefix="fastrsi-overlay-")
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, site_root)
        out_dir = os.path.join(overlay_root, rel)
        os.makedirs(out_dir, exist_ok=True)
        for fname in filenames:
            if fname.endswith(".pyc"):
                continue
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(out_dir, fname)
            if not os.path.lexists(dst_path):
                if copy_py and fname.endswith(".py"):
                    shutil.copy2(src_path, dst_path)
                else:
                    os.symlink(src_path, dst_path)

    standalone: dict[str, str] = {}
    for key, text in (overrides or {}).items():
        if os.path.isfile(key) or os.sep in key:
            target = os.path.join(overlay_root, os.path.basename(key))
            if os.path.lexists(target):
                os.unlink(target)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)
            standalone[key] = target
        else:
            rel_target = module_relpath(overlay_root, key)
            target = os.path.join(overlay_root, rel_target)
            if os.path.lexists(target):
                os.unlink(target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)

    return overlay_root, standalone


class PackageOverlay:
    """Context manager wrapping `build_package_overlay` with automatic temp-dir cleanup."""

    def __init__(
        self,
        package_name: str,
        overrides: dict[str, str] | None = None,
        *,
        copy_py: bool = False,
        src_dir: str | None = None,
    ):
        self.package_name = package_name
        self.overrides = overrides
        self.copy_py = copy_py
        self.src_dir = src_dir
        self.root: str | None = None
        self.standalone: dict[str, str] = {}

    def __enter__(self):
        self.root, self.standalone = build_package_overlay(
            self.package_name,
            self.overrides,
            copy_py=self.copy_py,
            src_dir=self.src_dir,
        )
        return self

    def __exit__(self, *_exc):
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None


def run_fresh_subprocess(
    fresh_module: str,
    spec: dict,
    overlay_root: str,
    *,
    extra_pythonpath: tuple[str, ...] | list[str] = (),
    timeout: float = 300.0,
    marker: str = "FRESH ",
    python: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Run `[python, '-m', fresh_module, spec.json]` with `overlay_root` first on `PYTHONPATH`
    and parse the last line prefixed with `marker` as JSON. Never raises."""
    spec_path = os.path.join(overlay_root, "fresh.json")
    try:
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, default=str)
        base_env = dict(os.environ if env is None else env)
        py_parts = [overlay_root, *[str(p) for p in extra_pythonpath if p]]
        if base_env.get("PYTHONPATH"):
            py_parts.append(base_env["PYTHONPATH"])
        base_env["PYTHONPATH"] = os.pathsep.join(py_parts)
        proc = subprocess.run(
            [python or sys.executable, "-m", fresh_module, spec_path],
            env=base_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or overlay_root,
        )
    except Exception as exc:
        return {"ok": False, "error": f"fresh_check: {type(exc).__name__}: {exc}", "output": None}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker)]
    if proc.returncode or not lines:
        tail = " | ".join(proc.stderr.strip().splitlines()[-4:]) or f"rc={proc.returncode}"
        return {"ok": False, "error": tail, "output": None}
    try:
        parsed = json.loads(lines[-1][len(marker):])
    except ValueError as exc:
        return {"ok": False, "error": f"malformed fresh JSON: {exc}", "output": None}
    return {"ok": True, **parsed}


def inject_participant_fault(participant, fault: str, **kw) -> dict:
    """Apply canonical kernel-level conformance faults directly to a `LocalParticipant`:
    - `fail_apply` (`after=1`): one-shot RuntimeError on `journal.set_code` after `after` swaps.
    - `fail_commit`: one-shot RuntimeError on `participant.quiesce`.
    - `tamper` (`module`, `qualname` or `function`, optional `source`): unjournaled
      `fn.__code__` replacement (changes code object identity without touching the journal).
    """
    if fault == "fail_apply":
        journal = participant.journal
        after = int(kw.get("after", 1))
        counter = [0]
        orig = journal.set_code

        def set_code(*args, **kwargs):
            counter[0] += 1
            if counter[0] > after:
                del journal.set_code
                raise RuntimeError(f"{participant.name}: injected fault after {after} swap(s)")
            return orig(*args, **kwargs)

        journal.set_code = set_code
        return {"ok": True, "fault": fault, "participant": participant.name}

    if fault == "fail_commit":
        orig_quiesce = participant.quiesce

        def failing(fn):
            participant.quiesce = orig_quiesce
            raise RuntimeError(f"{participant.name}: injected commit failure")

        participant.quiesce = failing
        return {"ok": True, "fault": fault, "participant": participant.name}

    if fault == "tamper":
        mod_name = kw.get("module")
        qualname = kw.get("qualname") or kw.get("function")
        if not mod_name or not qualname:
            return {"ok": False, "error": "tamper requires 'module' and 'qualname' (or 'function')"}
        actual = participant._actual(mod_name) if hasattr(participant, "_actual") else mod_name
        mod = sys.modules.get(actual) or sys.modules.get(mod_name)
        if mod is None:
            return {"ok": False, "error": f"module {mod_name!r} is not loaded"}
        obj = mod
        try:
            for part in str(qualname).split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            return {"ok": False, "error": f"tamper target not found: {exc}"}
        fn = getattr(obj, "__func__", obj)
        if not hasattr(fn, "__code__"):
            return {"ok": False, "error": f"{mod_name}:{qualname} has no __code__"}
        source = kw.get("source")
        if source:
            fn.__code__ = compile(source, fn.__code__.co_filename, "exec").co_consts[0]
        else:
            fn.__code__ = fn.__code__.replace(co_name=fn.__code__.co_name)
        return {"ok": True, "fault": fault, "participant": participant.name}

    return {"ok": False, "error": f"unknown participant fault {fault!r}"}


def append_jsonl(path: str | os.PathLike | None, record, *, suppress_errors: bool = False) -> None:
    """Append one UTF-8 JSON line to `path` (no-op when `path` is empty or `record` is None)."""
    if not path or record is None:
        return
    ctx = contextlib.suppress(Exception) if suppress_errors else contextlib.nullcontext()
    with ctx:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def resolve_patch_files(
    patch: dict | None,
    live_source,
    disk_source=None,
    *,
    against: str = "live",
    reject_mixed: bool = True,
) -> dict[str, str]:
    """Resolve `{module_or_path: text}` from a `{"files": ...}` or unified `{"diff": ...}` patch."""
    if not patch:
        return {}
    if patch.get("diff") is not None:
        if reject_mixed and patch.get("files"):
            raise ValueError("patch cannot supply both 'files' and 'diff'")
        from . import diffs

        reader = (disk_source or live_source) if against == "disk" else live_source

        def _safe_reader(key: str):
            try:
                return reader(key)
            except (FileNotFoundError, OSError, TypeError):
                return None

        return diffs.apply(patch["diff"], _safe_reader)
    return dict(patch.get("files") or {})


class SourceStack:
    """Layered live-source overlay stack over a base `disk_source(key)` reader."""

    def __init__(
        self,
        disk_source,
        frames: list[dict] | None = None,
        *,
        files_key: str | None = None,
        min_len: int = 1,
    ):
        self.disk_source = disk_source
        self.frames: list[dict] = frames if frames is not None else ([{}] if min_len else [])
        self.files_key = files_key
        self.min_len = min_len

    def live_source(self, key: str) -> str | None:
        for frame in reversed(self.frames):
            mapping = frame.get(self.files_key, {}) if self.files_key else frame
            if key in mapping:
                return mapping[key]
        return self.disk_source(key)

    def push(self, frame: dict) -> None:
        self.frames.append(frame)

    def pop(self, count: int = 1) -> int:
        drop = min(max(int(count), 0), max(len(self.frames) - self.min_len, 0))
        if drop:
            del self.frames[-drop:]
        return drop

    def clear(self) -> None:
        self.frames[:] = [{}] if self.min_len else []

    def resolve(self, patch: dict | None, against: str = "live") -> dict[str, str]:
        return resolve_patch_files(patch, self.live_source, self.disk_source, against=against)


def install_refusal(
    codes: list[str],
    findings: list[dict],
    *,
    outcome: str = "refused",
    t0: float | None = None,
) -> dict:
    """Standardized failed/refused `LabUnderTest.install()` reply."""
    res = {
        "ok": False,
        "id": None,
        "outcome": outcome,
        "codes": list(codes),
        "mechanisms": [],
        "findings": list(findings),
    }
    if t0 is not None:
        res["cost_s"] = round(time.perf_counter() - t0, 6)
    return res


def plain_finding(check: str, observed, expected, **extra) -> dict:
    """Build a lightweight finding dictionary (`check`, `observed`, `expected`, plus optional fields)."""
    return {"check": check, "observed": observed, "expected": expected, **extra}


def check_failure(error: str | None, findings: list[dict], *, t0: float | None = None) -> dict:
    """Standardized failed `LabUnderTest.check()` reply."""
    res = {
        "ok": False,
        "error": error,
        "output": None,
        "proven": None,
        "witness": {},
        "findings": list(findings),
    }
    if t0 is not None:
        res["cost_s"] = round(time.perf_counter() - t0, 6)
    return res
