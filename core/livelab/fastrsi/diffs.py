"""Apply a unified diff (`.patch` / `git diff`) in memory to produce `{module: new_text}`.

    files = apply(diff, current)     # {module: new full text}

`current(module)` returns the text the diff is made against (for example, a lab's live text:
base plus every live edit or rebuild so far), or `None` for a module created by the diff. The
diff is applied with `git apply` inside an empty temporary directory holding only those texts,
so engine files on disk are never read or modified, and a hunk that does not match the current
text raises `ValueError` (zero fuzz).
"""
import os
import re
import subprocess
import tempfile

_TARGET = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.M)
_SOURCE = re.compile(r"^--- (?:a/)?(\S+)", re.M)


def paths(diff: str) -> list[str]:
    """Python file paths a diff modifies or creates, in diff order (deleted files excluded)."""
    out: list[str] = []
    for path in _TARGET.findall(diff):
        if path != "/dev/null" and path.endswith(".py") and path not in out:
            out.append(path)
    return out


def module_of(path: str) -> str:
    """Convert `pkg/sub/x.py` -> `pkg.sub.x` (and `pkg/sub/__init__.py` -> `pkg.sub`)."""
    mod = path[:-3].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


def apply(diff: str, current) -> dict[str, str]:
    """Return `{module: new_text}` for every Python file `diff` touches.

    Raises `ValueError` when the diff touches no Python file, deletes a module, touches a
    non-Python file, or fails to apply cleanly to the texts returned by `current(module)`.
    """
    if "+++ /dev/null" in diff:
        raise ValueError("the diff deletes a module; a live lab cannot remove one")
    others = [
        path
        for path in _TARGET.findall(diff) + _SOURCE.findall(diff)
        if path != "/dev/null" and not path.endswith(".py")
    ]
    if others:
        raise ValueError(f"the diff touches files that are not modules: {sorted(set(others))}")
    targets = paths(diff)
    if not targets:
        raise ValueError("the diff changes no Python file")
    with tempfile.TemporaryDirectory(prefix="fastrsi-diff-") as tmp:
        for rel_path in targets:
            text = current(module_of(rel_path))
            if text is None:
                continue  # newly created by the diff
            full_path = os.path.join(tmp, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        # GIT_CEILING_DIRECTORIES prevents git apply from resolving paths against any
        # enclosing repository above the temporary directory.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": os.path.dirname(tmp)}
        res = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=diff,
            text=True,
            cwd=tmp,
            capture_output=True,
            env=env,
        )
        if res.returncode:
            raise ValueError(f"the diff does not apply to the live text: {res.stderr.strip()}")
        out: dict[str, str] = {}
        for rel_path in targets:
            with open(os.path.join(tmp, rel_path), encoding="utf-8") as fh:
                out[module_of(rel_path)] = fh.read()
        return out
