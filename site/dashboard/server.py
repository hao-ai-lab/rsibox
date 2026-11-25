#!/usr/bin/env python3
"""Read-only HTTP server for the RSIBox Delivery Dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = Path(__file__).resolve().parent

_PR_CACHE_LOCK = threading.Lock()
_PR_CACHE: dict[str, Any] = {"ts": 0.0, "data": []}
_PR_CACHE_TTL_S = 30.0

mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def _run_cmd(args: list[str], timeout_s: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""


def _collect_git() -> dict[str, Any]:
    _, head_short = _run_cmd(["git", "rev-parse", "--short", "HEAD"], 2.0)
    _, head_full = _run_cmd(["git", "rev-parse", "HEAD"], 2.0)
    _, branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], 2.0)
    _, status_out = _run_cmd(["git", "status", "--porcelain"], 2.0)

    dirty_files: list[dict[str, str]] = []
    for line in status_out.splitlines():
        if not line.strip():
            continue
        code = line[:2].strip() or "?"
        rel_path = line[3:].strip()
        dirty_files.append({"status": code, "path": rel_path})

    log_target = "HEAD"
    rc_main, _ = _run_cmd(["git", "rev-parse", "--verify", "main"], 1.5)
    if rc_main == 0:
        log_target = "main"

    _, log_out = _run_cmd(
        ["git", "log", log_target, "-n", "15", "--pretty=format:%h\t%an\t%ar\t%s"],
        2.0,
    )
    recent_commits: list[dict[str, str]] = []
    for line in log_out.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            recent_commits.append(
                {
                    "sha": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "subject": parts[3],
                }
            )

    _, wt_out = _run_cmd(["git", "worktree", "list", "--porcelain"], 2.0)
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in wt_out.splitlines():
        if not line.strip():
            if current:
                worktrees.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            wt_path = line[len("worktree ") :].strip()
            current["path"] = wt_path
            current["name"] = Path(wt_path).name
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :].strip()[:8]
        elif line.startswith("branch "):
            ref = line[len("branch ") :].strip()
            current["branch"] = ref.removeprefix("refs/heads/")
        elif line == "detached":
            current["branch"] = "detached"
    if current:
        worktrees.append(current)

    return {
        "head": head_short.strip(),
        "head_full": head_full.strip(),
        "branch": branch.strip(),
        "dirty_count": len(dirty_files),
        "dirty_files": dirty_files[:50],
        "recent_commits": recent_commits,
        "worktrees": worktrees,
    }


def _collect_disk() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT)
    gib = 1024**3
    total_gb = round(usage.total / gib, 1)
    used_gb = round(usage.used / gib, 1)
    free_gb = round(usage.free / gib, 1)
    used_pct = round((usage.used / usage.total) * 100, 1) if usage.total else 0.0
    return {
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "used_pct": used_pct,
    }


def _collect_system() -> dict[str, Any]:
    load_1m = None
    load_5m = None
    try:
        l1, l5, _ = os.getloadavg()
        load_1m = round(l1, 2)
        load_5m = round(l5, 2)
    except OSError:
        pass

    mem_total_gb = None
    mem_available_gb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                num_str = rest.strip().split(" ")[0]
                if num_str.isdigit():
                    values[key] = int(num_str)
            if "MemTotal" in values:
                mem_total_gb = round(values["MemTotal"] / (1024 * 1024), 1)
            if "MemAvailable" in values:
                mem_available_gb = round(values["MemAvailable"] / (1024 * 1024), 1)
        except OSError:
            pass

    return {
        "cpus": os.cpu_count(),
        "load_1m": load_1m,
        "load_5m": load_5m,
        "mem_total_gb": mem_total_gb,
        "mem_available_gb": mem_available_gb,
    }


def _collect_gpus() -> list[dict[str, Any]]:
    rc, out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        2.0,
    )
    if rc != 0 or not out.strip():
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "utilization_gpu": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return gpus


def _collect_open_prs() -> list[dict[str, Any]]:
    now = time.monotonic()
    with _PR_CACHE_LOCK:
        if now - float(_PR_CACHE["ts"]) < _PR_CACHE_TTL_S:
            return list(_PR_CACHE["data"])

    rc, out = _run_cmd(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "20",
            "--json",
            "number,title,headRefName,author,updatedAt,url",
        ],
        3.0,
    )
    prs: list[dict[str, Any]] = []
    if rc == 0 and out.strip():
        try:
            parsed = json.loads(out)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    author = item.get("author")
                    author_login = (
                        author.get("login", "") if isinstance(author, dict) else str(author or "")
                    )
                    prs.append(
                        {
                            "number": item.get("number"),
                            "title": item.get("title", ""),
                            "headRefName": item.get("headRefName", ""),
                            "author": author_login,
                            "updatedAt": item.get("updatedAt", ""),
                            "url": item.get("url", ""),
                        }
                    )
        except json.JSONDecodeError:
            prs = []

    with _PR_CACHE_LOCK:
        _PR_CACHE["ts"] = time.monotonic()
        _PR_CACHE["data"] = prs
        return list(prs)


def collect_state() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": _collect_git(),
        "disk": _collect_disk(),
        "system": _collect_system(),
        "gpus": _collect_gpus(),
        "open_prs": _collect_open_prs(),
    }


def _safe_resolve(base: Path, rel: str) -> Path | None:
    cleaned = unquote(rel).lstrip("/")
    candidate = (base / cleaned).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RSIBoxDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep stdout clean; errors still surface cleanly
        return

    def _send_bytes(
        self,
        status: int,
        payload: bytes,
        content_type: str,
        cache_control: str = "no-cache",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        if ctype is None:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "application/javascript"):
            ctype = f"{ctype}; charset=utf-8"
        data = path.read_bytes()
        self._send_bytes(HTTPStatus.OK, data, ctype)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        req_path = parsed.path

        if req_path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard")
            self.end_headers()
            return

        if req_path in ("/dashboard", "/dashboard/"):
            self._serve_file(DASHBOARD_DIR / "index.html")
            return

        if req_path.startswith("/dashboard/"):
            rel = req_path[len("/dashboard/") :]
            target = _safe_resolve(DASHBOARD_DIR, rel)
            if target is None:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            self._serve_file(target)
            return

        if req_path in ("/site/dashboard", "/site/dashboard/"):
            self._serve_file(DASHBOARD_DIR / "index.html")
            return

        if req_path == "/api/state":
            body = json.dumps(collect_state(), ensure_ascii=False).encode("utf-8")
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "application/json; charset=utf-8",
                cache_control="no-store",
            )
            return

        target = _safe_resolve(REPO_ROOT, req_path)
        if target is None:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if target.is_dir():
            index_file = target / "index.html"
            if index_file.is_file():
                self._serve_file(index_file)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        self._serve_file(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the RSIBox Delivery Dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"RSIBox Delivery Dashboard listening on http://{args.host}:{args.port}/dashboard")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
