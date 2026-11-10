#!/usr/bin/env python3
"""Poll GindaChen/fastrsi and run the B200 CI once per new trusted head.

Candidates are the head of main and the heads of open PRs whose head repository is
GindaChen/fastrsi itself (fork PRs are skipped here and refused again by the runner).
Each candidate runs through Rack (`rack-run`, which supervises the job and records it),
one at a time. A SHA with a result in <work>/results/ is not rerun.

    python3 b200_ci_loop.py --once            # one pass, print what it would run
    python3 b200_ci_loop.py --interval 300    # loop; add --post to post statuses

Not installed anywhere by default; see fastrsi-b200-ci.service.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "GindaChen/fastrsi"
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


def run_one(a, sha, pr):
    work = Path(a.work).expanduser()
    result = work / "results" / f"{sha}.json"
    argv = [sys.executable, str(HERE / "run_b200_ci.py"), "--sha", sha, "--work", str(work),
            "--gpus", a.gpus, "--gpu-pool", a.gpu_pool]
    if pr:
        argv += ["--pr", str(pr)]
    if a.vllm_python:
        argv += ["--vllm-python", a.vllm_python]
    if a.logs_url_base:
        argv += ["--logs-url-base", a.logs_url_base]
    if a.post:
        argv.append("--post")
    launch = [str(Path(a.rack_run).expanduser()), "--agent", a.agent, "--title", f"b200-ci {sha[:12]}",
              "--cluster", "b200", "--node", "b200", "--"] + argv
    print("launch:", " ".join(launch), flush=True)
    if a.dry_run:
        return
    subprocess.run(launch, check=True)  # returns at once; rack-run supervises the job
    deadline = time.time() + a.max_wait
    while not result.exists() and time.time() < deadline:
        time.sleep(15)
    print(result.read_text() if result.exists() else f"no result for {sha} after {a.max_wait}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="~/fastrsi-ci")
    ap.add_argument("--gpus", default="auto:2")
    ap.add_argument("--gpu-pool", default="0,1,2,3")
    ap.add_argument("--vllm-python", default="")
    ap.add_argument("--logs-url-base", default="")
    ap.add_argument("--agent", default="claude")
    ap.add_argument("--rack-run", default="~/.rack/bin/rack-run")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print launches, run nothing")
    ap.add_argument("--post", action="store_true")
    a = ap.parse_args()
    while True:
        done = Path(a.work).expanduser() / "results"
        for sha, pr, ref in candidates():
            if (done / f"{sha}.json").exists():
                continue
            print(f"new head {sha[:12]} ({ref}{f', PR #{pr}' if pr else ''})", flush=True)
            run_one(a, sha, pr)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
