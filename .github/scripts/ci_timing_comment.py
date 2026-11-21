#!/usr/bin/env python3
"""Post one comment per pull request that says where its CI time went.

Reads the workflow runs of one head commit from the GitHub Actions API (their jobs, steps and
timestamps; no logs, no code from the pull request) and renders a table: wall-clock from the
first queue to the last job done, runner time, and per job its queue time, run time, runner and
slowest step. The comment carries a hidden marker and is updated in place on every push, so a
pull request has one timing comment, not one per run.

Resolution is the API's: whole seconds per step. A step is a test suite here, so this says
which suite cost how much, not which test case.

    python3 .github/scripts/ci_timing_comment.py --repo owner/name --sha <sha> --pr <n>          # print
    python3 .github/scripts/ci_timing_comment.py --repo owner/name --sha <sha> --pr <n> --post   # comment

Needs the `gh` CLI with a token in GH_TOKEN (actions: read; pull-requests: write to post).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime

MARKER = "<!-- fastrsi-ci-timing -->"
DETAIL_STEP_SECONDS = 2
STEP_NAME_WIDTH = 70


def parse_time(text):
    """Seconds since the epoch for an API timestamp, or None when the field is empty."""
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def span(start, end):
    """Whole seconds from start to end; None when either is missing; never negative."""
    start_seconds, end_seconds = parse_time(start), parse_time(end)
    if start_seconds is None or end_seconds is None:
        return None
    return max(0, round(end_seconds - start_seconds))


def human(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def cell(text, width=None):
    """Text safe inside a markdown table cell: no pipes, no raw angle brackets, one line."""
    text = str(text).replace("\n", " ")
    if width is not None and len(text) > width:
        text = text[:width - 3].rstrip() + "..."
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")


def job_row(job):
    """What the table shows for one job, computed from its API record."""
    steps = [
        {"name": step["name"], "seconds": span(step.get("started_at"), step.get("completed_at")),
         "conclusion": step.get("conclusion")}
        for step in job.get("steps") or []
        if step.get("conclusion") != "skipped"
    ]
    timed = [step for step in steps if step["seconds"] is not None]
    slowest = max(timed, key=lambda step: step["seconds"]) if timed else None
    return {
        "name": job["name"],
        "result": job.get("conclusion") or job.get("status") or "unknown",
        "runner": job.get("runner_name") or "n/a",
        "queued": span(job.get("created_at"), job.get("started_at")),
        "ran": span(job.get("started_at"), job.get("completed_at")),
        "slowest": slowest,
        "steps": steps,
    }


def render(sha, runs):
    """Markdown for the comment. `runs` is a list of {"run": run record, "jobs": [job records]}."""
    rows, workflow_lines, starts, ends = [], [], [], []
    for entry in runs:
        run, jobs = entry["run"], entry["jobs"]
        job_rows = [job_row(job) for job in jobs]
        rows.extend(job_rows)
        run_start = parse_time(run.get("created_at"))
        job_ends = [parse_time(job.get("completed_at")) for job in jobs]
        job_ends = [end for end in job_ends if end is not None]
        if run_start is not None:
            starts.append(run_start)
        ends.extend(job_ends)
        wall = round(max(job_ends) - run_start) if job_ends and run_start is not None else None
        queues = [row["queued"] for row in job_rows if row["queued"] is not None]
        workflow_lines.append(
            f"| [{cell(run['name'])}]({run['html_url']}) | {run.get('conclusion') or run.get('status')} "
            f"| {human(wall)} | {human(max(queues) if queues else None)} | {len(job_rows)} |")

    total_wall = round(max(ends) - min(starts)) if starts and ends else None
    runner_seconds = sum(row["ran"] or 0 for row in rows)
    not_passed = [row for row in rows if row["result"] not in ("success", "skipped")]
    verdict = "all passed" if not not_passed else f"{len(not_passed)} not passed"

    lines = [
        MARKER,
        f"### CI timing for `{sha[:7]}`",
        "",
        f"**{human(total_wall)} wall-clock** from the first queue to the last job done; "
        f"{len(rows)} jobs, {human(runner_seconds)} of runner time, {verdict}.",
        "",
        "| Workflow | Result | Wall-clock | Longest queue | Jobs |",
        "|---|---|---:|---:|---:|",
        *workflow_lines,
        "",
        "| Job | Result | Runner | Queued | Ran | Slowest step |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda row: -(row["ran"] or 0)):
        slowest = row["slowest"]
        slowest_text = f"{cell(slowest['name'], STEP_NAME_WIDTH)} ({human(slowest['seconds'])})" if slowest else "n/a"
        lines.append(
            f"| {cell(row['name'])} | {row['result']} | {cell(row['runner'])} "
            f"| {human(row['queued'])} | {human(row['ran'])} | {slowest_text} |")

    detail = []
    for row in sorted(rows, key=lambda row: -(row["ran"] or 0)):
        steps = [step for step in row["steps"] if (step["seconds"] or 0) >= DETAIL_STEP_SECONDS]
        if not steps:
            continue
        detail.append(f"- **{cell(row['name'])}** ({human(row['ran'])})")
        for step in sorted(steps, key=lambda step: -step["seconds"]):
            failed = "" if step["conclusion"] in ("success", None) else f", {step['conclusion']}"
            detail.append(f"  - {human(step['seconds'])}: {cell(step['name'])}{failed}")
    if detail:
        lines += ["", f"<details><summary>Steps of {DETAIL_STEP_SECONDS} s or more, per job</summary>",
                  "", *detail, "", "</details>"]
    lines += ["", "Times are whole seconds from the Actions API; a step is one test suite. "
                  "Queued is from job creation to its start on a runner."]
    return "\n".join(lines) + "\n"


def gh_api(*arguments, body=None):
    command = ["gh", "api", *arguments]
    if body is not None:
        command += ["--input", "-"]
    done = subprocess.run(command, input=None if body is None else json.dumps(body),
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"gh api {' '.join(arguments)} failed: {done.stderr.strip()}")
    return json.loads(done.stdout) if done.stdout.strip() else None


def collect(repo, sha):
    """The latest run of each workflow for this commit, with its jobs (latest attempt)."""
    listed = gh_api(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100")["workflow_runs"]
    latest = {}
    for run in listed:
        if run["name"] not in latest or run["created_at"] > latest[run["name"]]["created_at"]:
            latest[run["name"]] = run
    runs = []
    for run in sorted(latest.values(), key=lambda run: run["name"]):
        jobs = gh_api(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
        runs.append({"run": run, "jobs": jobs})
    return runs


def find_pull(repo, sha):
    """The pull request whose head is this commit (an open one first), or None. A pull request
    merged right after its checks still gets its comment."""
    pulls = [pull for pull in gh_api(f"repos/{repo}/commits/{sha}/pulls?per_page=100") or []
             if pull.get("head", {}).get("sha") == sha]
    pulls.sort(key=lambda pull: pull.get("state") != "open")
    return pulls[0]["number"] if pulls else None


def post(repo, pull_number, body):
    """Update the pull request's timing comment, or create it. Returns 'updated' or 'created'."""
    comments = gh_api(f"repos/{repo}/issues/{pull_number}/comments?per_page=100")
    for comment in comments:
        if MARKER in (comment.get("body") or ""):
            gh_api("-X", "PATCH", f"repos/{repo}/issues/comments/{comment['id']}", body={"body": body})
            return "updated"
    gh_api("-X", "POST", f"repos/{repo}/issues/{pull_number}/comments", body={"body": body})
    return "created"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--sha", required=True, help="head commit of the pull request")
    parser.add_argument("--pr", type=int,
                        help="pull request number; default: the pull request whose head is --sha")
    parser.add_argument("--only", action="append", default=[],
                        help="workflow name to include (repeatable); default: every workflow of the commit")
    parser.add_argument("--post", action="store_true", help="create or update the comment; default prints it")
    arguments = parser.parse_args(argv)

    runs = collect(arguments.repo, arguments.sha)
    if arguments.only:
        runs = [entry for entry in runs if entry["run"]["name"] in arguments.only]
    if not runs:
        print(f"no workflow runs for {arguments.sha}; nothing to report")
        return 0
    body = render(arguments.sha, runs)
    if not arguments.post:
        print(body, end="")
        return 0
    pull_number = arguments.pr or find_pull(arguments.repo, arguments.sha)
    if pull_number is None:
        print(f"no pull request has head {arguments.sha}; nothing to comment on")
        return 0
    print(f"{post(arguments.repo, pull_number, body)} the timing comment on #{pull_number}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
