#!/usr/bin/env python3
"""Poll hao-ai-lab/fastrsi and run the B200 CI once per new trusted head.

Candidates are the head of main and the heads of open PRs whose head repository is
hao-ai-lab/fastrsi itself (fork PRs are skipped here and refused again by the runner).
Each candidate runs through Rack (`rack-run`, which supervises the job and records it),
one at a time. A SHA with a final result in <work>/results/ is not rerun; a refusal or a runner error
(before the last attempt) leaves a non-final result, so the SHA is tried again.
A run that has not finished after --max-wait seconds is stopped (SIGTERM to the pid in
<work>/active/<sha>.pid) before the next one starts, so two runs never overlap.

    python3 b200_ci_loop.py --once            # one pass, print what it would run
    python3 b200_ci_loop.py --interval 300    # loop; add --post to post statuses

Not installed anywhere by default; see fastrsi-b200-ci.service.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = "hao-ai-lab/fastrsi"
HERE = Path(__file__).resolve().parent


def gh(path):
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout)


def candidates():
    main = gh(f"repos/{REPO}/branches/main")["commit"]["sha"]
    out = [(main, None, "main")]
    for pr in gh(f"repos/{REPO}/pulls?state=open&per_page=100"):
        head_repo = (pr["head"].get("repo") or {}).get("full_name")
        if head_repo != REPO:
            print(f"skip PR #{pr['number']}: head repo {head_repo!r} is not {REPO}", flush=True)
            continue
        out.append((pr["head"]["sha"], pr["number"], pr["head"]["ref"]))
    return out


def is_final(result: Path) -> bool:
    """A result written before attempts were counted has no "final" key and is final."""
    try:
        return bool(json.loads(result.read_text()).get("final", True))
    except (OSError, ValueError, AttributeError):
        return False


def runner_argv(a, sha, pr, work: Path) -> list:
    argv = [sys.executable, str(HERE / "run_b200_ci.py"), "--sha", sha, "--work", str(work),
            "--gpus", a.gpus, "--gpu-pool", a.gpu_pool, "--python", a.python,
            "--quickstart-pythons", a.quickstart_pythons, "--stages", a.stages,
            "--timeout", str(a.timeout), "--max-attempts", str(a.max_attempts),
            # the runner stops itself first; the SIGTERM below is the fallback
            "--deadline-seconds", str(max(60, a.max_wait - 60))]
    if a.source:
        argv += ["--source", a.source]
    if pr:
        argv += ["--pr", str(pr)]
    if a.vllm_python:
        argv += ["--vllm-python", a.vllm_python]
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


def stop_run(work: Path, sha: str, since: float) -> bool:
    """SIGTERM the runner of `sha` launched since `since`; True if a signal was sent. A pid file
    older than the launch belongs to a dead run whose pid may have been reused: left alone."""
    pid_file = work / "active" / f"{sha}.pid"
    if not fresh(pid_file, since):
        return False
    try:
        os.kill(int(pid_file.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True


def run_one(a, sha, pr):
    work = Path(a.work).expanduser()
    result = work / "results" / f"{sha}.json"
    argv = runner_argv(a, sha, pr, work)
    launch = [str(Path(a.rack_run).expanduser()), "--agent", a.agent, "--title", f"b200-ci {sha[:12]}",
              "--cluster", "b200", "--node", "b200", "--"] + argv
    print("launch:", " ".join(launch), flush=True)
    if a.dry_run:
        return False
    since = time.time() - 1
    subprocess.run(launch, check=True)  # returns at once; rack-run supervises the job
    done = wait_run(work, sha, since, a.max_wait, a.poll)
    if not done:
        print(f"no result for {sha} after {a.max_wait}s; stopping it", flush=True)
        done = stop_run(work, sha, since) and wait_run(work, sha, since, 120, a.poll)
    print(result.read_text() if done else f"no result for {sha}; tried again next pass", flush=True)
    return done


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="~/fastrsi-ci")
    ap.add_argument("--source", default="", help="git URL to fetch branch heads from (runner default)")
    ap.add_argument("--gpus", default="auto:2")
    ap.add_argument("--gpu-pool", default="0,1,2,3")
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--quickstart-pythons", default="3.11,3.12")
    ap.add_argument("--stages", default="cpu,gpu,quickstart")
    ap.add_argument("--timeout", type=int, default=1200, help="per step, seconds")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--vllm-python", default="")
    ap.add_argument("--logs-url-base", default="")
    ap.add_argument("--agent", default="claude")
    ap.add_argument("--rack-run", default="~/.rack/bin/rack-run")
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
