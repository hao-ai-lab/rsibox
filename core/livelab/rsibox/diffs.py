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

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_DIFF = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def _headers(diff: str) -> tuple[list[str], list[str], bool]:
    """Parse file headers outside `@@ ... @@` hunks.

    Returns `(sources, targets, deleted)` with `/dev/null` excluded and diff order preserved.
    Ignoring hunk bodies prevents added/removed lines beginning with `++ ` / `-- ` or literal
    `"+++ /dev/null"` strings inside Python source from being mistaken for file headers.
    """
    sources: list[str] = []
    targets: list[str] = []
    deleted = False
    old_rem = new_rem = 0
    git_pair: tuple[str, str] | None = None
    saw_file_hdr = False
    is_new_file = False
    is_deleted_file = False

    def flush_git_pair() -> None:
        nonlocal git_pair, saw_file_hdr, is_new_file, is_deleted_file
        if git_pair is not None and not saw_file_hdr:
            src, dst = git_pair
            if is_deleted_file:
                if src not in sources:
                    sources.append(src)
            elif is_new_file:
                if dst not in targets:
                    targets.append(dst)
            else:
                if src not in sources:
                    sources.append(src)
                if dst not in targets:
                    targets.append(dst)
        git_pair = None
        saw_file_hdr = False
        is_new_file = False
        is_deleted_file = False

    for line in diff.splitlines():
        if old_rem > 0 or new_rem > 0:
            if line.startswith("\\"):
                continue
            if line.startswith(" ") or (line == "" and old_rem > 0 and new_rem > 0):
                old_rem -= 1
                new_rem -= 1
                continue
            if line.startswith("-"):
                old_rem -= 1
                continue
            if line.startswith("+"):
                new_rem -= 1
                continue
            old_rem = new_rem = 0

        m = _HUNK.match(line)
        if m:
            old_rem = int(m.group(2)) if m.group(2) is not None else 1
            new_rem = int(m.group(4)) if m.group(4) is not None else 1
            continue

        if line.startswith("diff --git "):
            flush_git_pair()
            gm = _GIT_DIFF.match(line)
            if gm:
                git_pair = (gm.group(1), gm.group(2))
            continue

        if line.startswith("deleted file mode "):
            deleted = True
            is_deleted_file = True
            continue
        if line.startswith("new file mode "):
            is_new_file = True
            continue
        if line.startswith("rename from "):
            deleted = True
            src = line[len("rename from "):].strip()
            if src and src not in sources:
                sources.append(src)
            saw_file_hdr = True
            continue
        if line.startswith("rename to "):
            dst = line[len("rename to "):].strip()
            if dst and dst not in targets:
                targets.append(dst)
            saw_file_hdr = True
            continue

        if line.startswith("--- "):
            raw = line[4:].split("\t", 1)[0].strip()
            path = raw[2:] if raw.startswith("a/") else raw
            saw_file_hdr = True
            if path != "/dev/null" and path not in sources:
                sources.append(path)
            continue

        if line.startswith("+++ "):
            raw = line[4:].split("\t", 1)[0].strip()
            path = raw[2:] if raw.startswith("b/") else raw
            saw_file_hdr = True
            if path == "/dev/null":
                deleted = True
            elif path not in targets:
                targets.append(path)
            continue

    flush_git_pair()
    return sources, targets, deleted


def paths(diff: str) -> list[str]:
    """Python file paths a diff modifies or creates, in diff order (deleted files excluded)."""
    _, targets, _ = _headers(diff)
    return [path for path in targets if path.endswith(".py")]


def module_of(path: str) -> str:
    """Convert `pkg/sub/x.py` -> `pkg.sub.x` (and `pkg/sub/__init__.py` -> `pkg.sub`)."""
    mod = path[:-3].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


def apply(diff: str, current) -> dict[str, str]:
    """Return `{module: new_text}` for every Python file `diff` touches.

    Raises `ValueError` when the diff touches no Python file, deletes a module, touches a
    non-Python file, or fails to apply cleanly to the texts returned by `current(module)`.
    """
    sources, all_targets, deleted = _headers(diff)
    if deleted:
        raise ValueError("the diff deletes a module; a live lab cannot remove one")
    others = [path for path in all_targets + sources if not path.endswith(".py")]
    if others:
        raise ValueError(f"the diff touches files that are not modules: {sorted(set(others))}")
    targets = [path for path in all_targets if path.endswith(".py")]
    if not targets:
        raise ValueError("the diff changes no Python file")
    for rel_path in targets:
        if rel_path.startswith("/") or ".." in rel_path.split("/"):
            raise ValueError(f"unsafe diff path: {rel_path}")
    with tempfile.TemporaryDirectory(prefix="rsibox-diff-") as tmp:
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
