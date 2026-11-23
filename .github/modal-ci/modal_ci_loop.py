#!/usr/bin/env python3
"""Poll hao-ai-lab/rsibox and run the Modal GPU CI once per new trusted head.

Candidates are the head of main and the heads of open PRs whose head repository is
hao-ai-lab/rsibox itself (fork PRs are skipped here and refused again by the runner).
A SHA with a final result in `<work>/results/` is not rerun; a refusal or a runner error
before `--max-attempts` leaves a non-final result, so the SHA is tried again.

    python3 modal_ci_loop.py --once --dry-run   # list heads it would run
    python3 modal_ci_loop.py --interval 300     # poll loop; add --post to post statuses
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = "hao-ai-lab/rsibox"
HERE = Path(__file__).resolve().parent


def gh(path):
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout)


def candidates():
    main_sha = gh(f"repos/{REPO}/branches/main")["commit"]["sha"]
    out = [(main_sha, None, "main")]
    for pr in gh(f"repos/{REPO}/pulls?state=open&per_page=100"):
        head_repo = (pr["head"].get("repo") or {}).get("full_name")
        if head_repo != REPO:
            print(f"skip PR #{pr['number']}: head repo {head_repo!r} is not {REPO}", flush=True)
            continue
        out.append((pr["head"]["sha"], pr["number"], pr["head"]["ref"]))
    return out


def is_final(result: Path) -> bool:
    """A result written before attempts were counted has no `final` key and is final."""
    try:
        return bool(json.loads(result.read_text()).get("final", True))
    except (OSError, ValueError, AttributeError):
        return False


def runner_argv(a, sha, pr, work: Path) -> list:
    argv = [
        sys.executable,
        str(HERE / "run_modal_ci.py"),
        "--sha",
        sha,
        "--work",
        str(work),
        "--gpu",
        a.gpu,
        "--python",
        a.python,
        "--quickstart-pythons",
        a.quickstart_pythons,
        "--stages",
        a.stages,
        "--timeout",
        str(a.timeout),
        "--max-attempts",
        str(a.max_attempts),
        "--deadline-seconds",
        str(max(60, a.max_wait - 60)),
    ]
    if a.source:
        argv += ["--source", a.source]
    if pr:
        argv += ["--pr", str(pr)]
    if getattr(a, "vllm_model", ""):
        argv += ["--vllm-model", a.vllm_model]
    if getattr(a, "suites", ""):
        argv += ["--suites", a.suites]
    if getattr(a, "modal_cmd", ""):
        argv += ["--modal-cmd", a.modal_cmd]
    if a.logs_url_base:
        argv += ["--logs-url-base", a.logs_url_base]
    if a.post:
        argv.append("--post")
    return argv


def fresh(path: Path, since: float) -> bool:
    try:
        return path.stat().st_mtime >= since
    except OSError:
        return False


def wait_run(work: Path, sha: str, since: float, seconds: float, poll: float) -> bool:
    """True once the runner wrote its result for `sha` (any outcome, refusals included)."""
    result, deadline = work / "results" / f"{sha}.json", time.time() + seconds
    while not fresh(result, since):
        if time.time() >= deadline:
            return False
        time.sleep(poll)
    return True


def is_runner(pid: int, sha: str) -> bool:
    """True if `pid` is alive and its command line is run_modal_ci.py for `sha`."""
    # -ww disables column-width truncation so long paths/flags never hide --sha.
    found = subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True)
    if found.returncode != 0:
        return False
    cmdline = found.stdout
    return (bool(re.search(r"(?:^|[/\s])run_modal_ci\.py(?:\s|$)", cmdline))
            and bool(re.search(rf"(?:^|\s)--sha(?:=|\s+){re.escape(sha)}(?:\s|$)", cmdline)))


def stop_run(work: Path, sha: str, since: float) -> bool:
    """SIGTERM the runner of `sha` launched since `since`; True if a signal was sent."""
    pid_file = work / "active" / f"{sha}.pid"
    if not fresh(pid_file, since):
        return False
    try:
        pid = int(pid_file.read_text())
        if not is_runner(pid, sha):
            return False
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True


def run_one(a, sha, pr):
    work = Path(a.work).expanduser()
    result = work / "results" / f"{sha}.json"
    argv = runner_argv(a, sha, pr, work)
    print("launch:", shlex.join(argv), flush=True)
    if a.dry_run:
        return False
    since = time.time() - 1
    proc = subprocess.Popen(argv)
    try:
        deadline = time.time() + a.max_wait
        while proc.poll() is None and not fresh(result, since):
            if time.time() >= deadline:
                break
            time.sleep(a.poll)
        if proc.poll() is None and not fresh(result, since):
            print(f"no result for {sha} after {a.max_wait}s; stopping it", flush=True)
            stopped = stop_run(work, sha, since)
            if not stopped:
                proc.terminate()
            wait_run(work, sha, since, 120, a.poll)
        proc.wait(timeout=30)
        done = fresh(result, since)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        done = fresh(result, since)
    print(result.read_text() if done else f"no result for {sha}; tried again next pass", flush=True)
    return done


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="~/rsibox-ci")
    ap.add_argument("--source", default="", help="git URL to fetch branch heads from (runner default)")
    ap.add_argument("--gpu", default=os.environ.get("MODAL_GPU", "L4"))
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--quickstart-pythons", default="3.11,3.12")
    ap.add_argument("--stages", default="gpu")
    ap.add_argument("--timeout", type=int, default=1200, help="per step, seconds")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--vllm-model", default=os.environ.get("MODAL_VLLM_MODEL") or os.environ.get("RSIBOX_VLLM_MODEL", ""))
    ap.add_argument("--suites", default="")
    ap.add_argument("--modal-cmd", default="")
    ap.add_argument("--logs-url-base", default="")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--poll", type=float, default=15, help=argparse.SUPPRESS)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print launches, run nothing")
    ap.add_argument("--post", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    while True:
        done = Path(a.work).expanduser() / "results"
        for sha, pr, ref in candidates():
            if is_final(done / f"{sha}.json"):
                continue
            print(f"new head {sha[:12]} ({ref}{f', PR #{pr}' if pr else ''})", flush=True)
            run_one(a, sha, pr)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
