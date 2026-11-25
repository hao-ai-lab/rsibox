"""Shared stdlib helpers for RSIBox rstack deterministic gate scripts."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PR_TITLE_RE = re.compile(r"^(feat|fix|perf|docs|test|ci|refactor|chore)\([a-z0-9-]+\)!?: .+")
AI_ATTR_RE = re.compile(
    r"(co-authored-by:.*(claude|anthropic|codex|openai|gemini|antigravity|copilot))"
    r"|generated with \[?(claude|codex|gemini|antigravity)",
    re.IGNORECASE,
)
REQUIRED_HEADINGS = ("## What and why", "## Evidence", "## Loop record")
REQUIRED_LOOP_KEYS = (
    "Engine",
    "Upstream PRs",
    "Function groups",
    "Rule added",
    "Overheads measured",
    "Receipts",
    "Not verified",
)
REQUIRED_GPU_KEYS = ("Ideal full GPU test", "Actual tests run", "Why unrun GPU paths were skipped")


def find_repo_root(start: Path | str | None = None) -> Path:
    """Locate the enclosing git repository or worktree root."""
    cur = Path(start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".git").exists():
            return candidate
    res = run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=cur, timeout=10)
    if res.returncode == 0 and res.stdout.strip():
        return Path(res.stdout.strip()).resolve()
    raise RuntimeError(f"Not inside a git repository: {cur}")


def run_cmd(
    argv: list[str], cwd: Path | None = None, timeout: int = 60, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command without shell=True and with an explicit timeout."""
    merged_env = (os.environ | env) if env is not None else None
    return subprocess.run(
        argv, cwd=str(cwd) if cwd else None, env=merged_env, capture_output=True, text=True, timeout=timeout, check=False
    )


def which(name: str) -> str | None:
    """Resolve an executable on PATH without hardcoded paths."""
    return shutil.which(name)


def load_json(path: Path, default: Any = None) -> Any:
    """Read UTF-8 JSON from disk, returning default if absent."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON via a temp file in the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def emit(ok: bool, check: str, **details: Any) -> int:
    """Print structured JSON output and return 0 on success or 1 on failure."""
    sys.stdout.write(json.dumps({"ok": ok, "check": check, **details}, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0 if ok else 1
