"""A unified diff (a `.patch` file, `git diff` output) as the `files` of an install or a rebuild.

    files = apply(diff, current)     # {module: new full text}

`current(module)` returns the text the diff is made against (the lab passes its live text: the
base plus every live edit or rebuild so far), or None for a module that does not exist yet. The
diff is applied with `git apply` in an empty temporary folder that holds only those texts, so
the engine's files on disk are never read or written, and a hunk that does not match the
current text is an error (no fuzz): a diff made against other code is refused, never applied
to the wrong lines. Raises ValueError with git's message when it does not apply.
"""
import os
import re
import subprocess
import tempfile

_TARGET = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.M)
_SOURCE = re.compile(r"^--- (?:a/)?(\S+)", re.M)


def paths(diff):
    """The Python files a diff changes or creates, in diff order (deleted files excluded)."""
    out = []
    for p in _TARGET.findall(diff):
        if p != "/dev/null" and p.endswith(".py") and p not in out:
            out.append(p)
    return out


def module_of(path):
    """`fastvideo/models/dits/x.py` -> `fastvideo.models.dits.x` (a package's `__init__.py`
    -> the package)."""
    mod = path[:-3].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


def apply(diff, current):
    """{module: new text} for every Python file the diff touches. ValueError when the diff
    deletes a module, touches a non-Python file, or does not apply to `current`'s texts."""
    targets = paths(diff)
    if not targets:
        raise ValueError("the diff changes no Python file")
    others = [p for p in _TARGET.findall(diff) + _SOURCE.findall(diff)
              if p != "/dev/null" and not p.endswith(".py")]
    if others:
        raise ValueError(f"the diff touches files that are not modules: {sorted(set(others))}")
    if "+++ /dev/null" in diff:
        raise ValueError("the diff deletes a module; a live lab cannot remove one")
    with tempfile.TemporaryDirectory(prefix="fvlab-diff-") as tmp:
        for p in targets:
            text = current(module_of(p))
            if text is None:
                continue                                  # created by the diff
            f = os.path.join(tmp, p)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w") as fh:
                fh.write(text)
        # GIT_CEILING_DIRECTORIES: never treat a repository around the temporary folder as
        # the target (git apply resolves paths from a work tree's root)
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": os.path.dirname(tmp)}
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=diff, text=True,
                           cwd=tmp, capture_output=True, env=env)
        if r.returncode:
            raise ValueError(f"the diff does not apply to the live text: {r.stderr.strip()}")
        out = {}
        for p in targets:
            with open(os.path.join(tmp, p)) as fh:
                out[module_of(p)] = fh.read()
        return out
