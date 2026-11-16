#!/usr/bin/env python3
"""B200 CI: test one trusted commit of hao-ai-lab/fastrsi on the B200 machine.

Stdlib only. Run it on B200, inside a Rack-supervised job (see README.md here):

    ~/.rack/bin/rack-run --agent claude --title "b200-ci <sha>" --cluster b200 --node b200 -- \
        python3 run_b200_ci.py --sha <sha> --gpus auto:2

Steps: fetch branch heads (never refs/pull/*) into a mirror, refuse a SHA that no
branch of the repo contains (fork PRs), check it out into a fresh run directory,
build fresh virtualenvs, run

  cpu         the cpu-tests.yml suites with no torch and no GPU (GitHub parity),
  gpu         the same suites with torch + triton, FASTRSI_TEST_GPU=1, GPUs visible
              and (optionally) a vLLM python, so the torch/GPU/vLLM cases run,
  quickstart  the README quickstart block, verbatim, in a fresh venv,

then write logs, summary.json, summary.md and the commit status it would post
(context `b200-gpu`). It posts that status only with --post.

Retries: <work>/results/<sha>.json says whether the result is final; the loop reruns a SHA
until it is. A refusal (untrusted SHA, no free GPUs) is never final and posts no final
status. A runner error before the SHA is verified (fetch) is never final; after that it is
one attempt of --max-attempts, and only the last is final. A step cut by
--deadline-seconds or SIGTERM is a runner error. A test result is always final.
Exit code: 0 success, 1 test failure, 2 infrastructure error or refusal.
"""
import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = "hao-ai-lab/fastrsi"
CONTEXT = "b200-gpu"
HERE = Path(__file__).resolve().parent
TORCH_SPEC = ["torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128",
              "--extra-index-url", "https://pypi.org/simple"]
GPU_EXTRA = ["pytest", "numpy", "nvidia-ml-py"]
CPU_EXTRA = ["pytest"]  # cpu-tests.yml installs only pytest, and only for vllm-services


class Refused(Exception):
    """Not run, and not a result: the SHA is not trusted or the GPUs are not free."""


class Stopped(Exception):
    """The runner was told to stop (SIGTERM) or ran past --deadline-seconds."""


class Deadline:
    """Caps a whole run (--deadline-seconds). A step cut by it, or any later step, raises
    Stopped, which is recorded as a runner error, never as a test result."""

    def __init__(self, seconds: int):
        self.end = time.monotonic() + seconds if seconds else None

    def expired(self) -> bool:
        return self.end is not None and time.monotonic() >= self.end

    def timeout(self, per_step: int) -> float:
        if self.end is None:
            return per_step
        left = self.end - time.monotonic()
        if left <= 0:
            raise Stopped("runner deadline reached")
        return min(per_step, left)


def run_group(argv, per_step: int, deadline: Deadline, **kw):
    """Run one step in its own process group; return its exit code, or "timeout" when the
    per-step timeout cut it. On a timeout, the run deadline, SIGTERM (Stopped) or any other
    exception the whole group is killed, grandchildren included, before this returns or
    raises. A process that leaves the group (setsid) is not covered."""
    proc = subprocess.Popen(argv, start_new_session=True, **kw)
    try:
        return proc.wait(timeout=deadline.timeout(per_step))
    except subprocess.TimeoutExpired:
        kill_group(proc)
        if deadline.expired():
            raise Stopped("runner deadline reached during a step")
        return "timeout"
    except BaseException:
        kill_group(proc)
        raise


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def sh(argv, **kw):
    kw.setdefault("check", True)
    kw.setdefault("text", True)
    return subprocess.run(argv, **kw)


# ---------------------------------------------------------------- source + trust

def sync_mirror(work: Path, source: str) -> Path:
    mirror = work / "mirror.git"
    if not mirror.exists():
        sh(["git", "init", "--quiet", "--bare", str(mirror)])
    # Branch heads only. GitHub serves fork PR commits only under refs/pull/*, which we
    # never fetch, so a fork PR head cannot enter this mirror through this fetch.
    sh(["git", "-C", str(mirror), "fetch", "--quiet", "--prune", "--no-tags", source,
        "+refs/heads/*:refs/heads/*"])
    return mirror


def trusted_branches(mirror: Path, sha: str) -> list:
    r = sh(["git", "-C", str(mirror), "cat-file", "-e", sha + "^{commit}"], check=False,
           capture_output=True)
    if r.returncode != 0:
        return []
    out = sh(["git", "-C", str(mirror), "branch", "--format=%(refname:short)", "--contains", sha],
             capture_output=True).stdout.split()
    return out


def check_pr(pr: int, sha: str) -> dict:
    """Same-repo PR whose head is exactly `sha`; refuses fork PRs."""
    r = sh(["gh", "api", f"repos/{REPO}/pulls/{pr}"], capture_output=True)
    d = json.loads(r.stdout)
    head_repo = (d.get("head") or {}).get("repo") or {}
    if head_repo.get("full_name") != REPO:
        raise Refused(f"PR #{pr} head is {head_repo.get('full_name')!r}, not {REPO}: fork PRs are not run")
    if d["head"]["sha"] != sha:
        raise Refused(f"PR #{pr} head is {d['head']['sha']}, not {sha}")
    return {"pr": pr, "head_ref": d["head"]["ref"], "base_ref": d["base"]["ref"]}


# ---------------------------------------------------------------- GPUs

def idle_gpus(nvidia_smi: str) -> list:
    out = sh([nvidia_smi, "--query-gpu=index,memory.used,utilization.gpu",
              "--format=csv,noheader,nounits"], capture_output=True).stdout
    idle = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        if int(mem) < 500 and int(util) == 0:
            idle.append(idx)
    return idle


def lock_gpu(lock_dir: Path, idx: str):
    """An exclusive, non-blocking lock on one GPU index, or None if another run holds it.
    The kernel drops the lock when the holder exits, however it exits."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = open(lock_dir / f"gpu{idx}.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def pick_gpus(spec: str, pool: str, lock_dir: Path, nvidia_smi: str = "nvidia-smi"):
    """`spec` is an explicit CUDA_VISIBLE_DEVICES value or auto:N. auto picks N idle GPUs,
    only from `pool` (the indices this runner is allowed on the shared machine).
    Returns (value, lock handles). Each chosen GPU is locked, so two runs never share one;
    an auto pick is checked idle again after its lock is taken. Keep the handles open for
    as long as the GPUs are in use."""
    allowed = set(pool.split(",")) if pool else None
    if not spec.startswith("auto"):
        chosen = spec.split(",")
        if allowed and not set(chosen) <= allowed:
            raise Refused(f"--gpus {spec} is outside the allowed pool {pool}")
        handles = []
        for idx in chosen:
            handle = lock_gpu(lock_dir, idx)
            if handle is None:
                for held in handles:
                    held.close()
                raise Refused(f"GPU {idx} is locked by another CI run")
            handles.append(handle)
        return spec, handles
    want = int(spec.split(":", 1)[1]) if ":" in spec else 1
    idle = [idx for idx in idle_gpus(nvidia_smi) if allowed is None or idx in allowed]
    chosen, handles = [], []
    for idx in idle:
        if len(chosen) == want:
            break
        handle = lock_gpu(lock_dir, idx)
        if handle is not None:
            chosen.append(idx)
            handles.append(handle)
    still_idle = set(idle_gpus(nvidia_smi)) if chosen else set()
    if len(chosen) < want or not set(chosen) <= still_idle:
        for held in handles:
            held.close()
        raise Refused(f"need {want} idle, unlocked GPUs in pool {pool or 'all'}, found {chosen} "
                      f"(idle: {idle})")
    return ",".join(chosen), handles


def check_still_idle(gpus: str, spec: str, nvidia_smi: str):
    """Right before the gpu stage: an auto-picked GPU that someone outside CI started using
    since it was picked (the cpu stage takes minutes) is a refusal, not a test failure."""
    if spec.startswith("auto"):
        busy = set(gpus.split(",")) - set(idle_gpus(nvidia_smi))
        if busy:
            raise Refused(f"GPUs {sorted(busy)} became busy before the gpu stage")


# ---------------------------------------------------------------- envs

def make_env(uv: str, path: Path, python: str, pkgs: list, log, deadline: Deadline) -> float:
    t0 = time.monotonic()
    commands = [[uv, "venv", "--quiet", "--python", python, str(path)]]
    commands += [[uv, "pip", "install", "--quiet", "--python", str(path / "bin" / "python"), *group]
                 for group in pkgs]
    for argv in commands:
        exit_code = run_group(argv, 1800, deadline, stdout=log, stderr=subprocess.STDOUT)
        if exit_code != 0:
            raise RuntimeError(f"{shlex.join(argv[:3])} exited {exit_code}")
    return time.monotonic() - t0


# ---------------------------------------------------------------- suites

def suites(src: Path, py: str, res: Path):
    """The cpu-tests.yml steps, in order. Each yields machine-readable results."""
    uj = str(HERE / "unittest_json.py")

    def ut(name, s, t=None, p=None, cwd="."):
        argv = [py, uj, "--json", str(res / f"{name}.json"), "-s", s]
        if t:
            argv += ["-t", t]
        if p:
            argv += ["-p", p]
        return {"name": name, "kind": "unittest", "argv": argv, "cwd": cwd}

    def cmd(name, argv, cwd=".", env=None):
        return {"name": name, "kind": "cmd", "argv": argv, "cwd": cwd, "env": env or {}}

    fv = str(src / "repos/fastvideo/livelab")
    return [
        ut("core-conformance-cases", "conformance", ".", cwd="core/livelab"),
        ut("core-kernel", "core/livelab/kernel", "core/livelab"),
        ut("core-forksafe", "core/livelab", p="test_forksafe.py"),
        cmd("core-conformance-toylab", [py, "-m", "conformance", "--lab", "kernel.toy_lab:ToyLab"],
            cwd="core/livelab"),
        ut("fastvideo-adapter", "repos/fastvideo/livelab/adapter", "repos/fastvideo/livelab"),
        ut("fastvideo-services", "repos/fastvideo/livelab/services", "repos/fastvideo/livelab"),
        cmd("fastvideo-conformance-toylab",
            [py, "-m", "conformance", "--lab", "adapter.toy_lab:ToyFastVideoLab"],
            cwd="core/livelab", env={"PYTHONPATH": fv}),
        ut("vllm-adapter", "repos/vllm/livelab/adapter", "repos/vllm/livelab"),
        {"name": "vllm-services", "kind": "pytest",
         "argv": [py, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider",
                  f"--junitxml={res / 'vllm-services.xml'}", "repos/vllm/livelab/services"],
         "cwd": "."},
        ut("megatron-livelab", "repos/megatron/livelab/fastrsi_megatron", "repos/megatron/livelab"),
        ut("megatron-tools", "repos/megatron/tools"),
        ut("review-checkers", ".github/review", ".github/review"),
    ]


def parse_results(step, res: Path) -> list:
    if step["kind"] == "unittest":
        f = res / f"{step['name']}.json"
        return json.loads(f.read_text())["tests"] if f.exists() else []
    if step["kind"] == "pytest":
        f = res / f"{step['name']}.xml"
        if not f.exists():
            return []
        tests = []
        for tc in ET.parse(f).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            outcome, detail = "passed", ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    outcome, detail = ("failed" if child.tag == "failure" else "error"), \
                        (child.get("message") or "")[:2000]
                elif child.tag == "skipped":
                    xf = (child.get("type") or "") == "pytest.xfail"
                    outcome, detail = ("xfailed" if xf else "skipped"), child.get("message") or ""
            tests.append({"id": tid, "outcome": outcome, "seconds": float(tc.get("time") or 0),
                          "detail": detail})
        return tests
    return []


def run_stage(stage, src: Path, py: str, env_extra: dict, run_dir: Path, timeout: int,
              deadline: Deadline) -> dict:
    res = run_dir / "results" / stage
    logs = run_dir / "logs" / stage
    res.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    out = {"stage": stage, "steps": []}
    t_stage = time.monotonic()
    for step in suites(src, py, res):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("FASTRSI_", "PYTHON", "VIRTUAL_ENV"))}
        env["PATH"] = str(Path(py).parent) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
        env.update(env_extra)
        env.update(step.get("env", {}))
        log_path = logs / f"{step['name']}.log"
        t0 = time.monotonic()
        with open(log_path, "w") as log:
            log.write("$ (cd %s && %s)\n" % (step["cwd"], shlex.join(step["argv"])))
            log.flush()
            exit_code = run_group(step["argv"], timeout, deadline, cwd=src / step["cwd"], env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        tests = parse_results(step, res)
        if step["kind"] == "cmd":
            tests = [{"id": step["name"], "outcome": "passed" if exit_code == 0 else "failed",
                      "seconds": round(time.monotonic() - t0, 3), "detail": ""}]
        counts = {}
        for t in tests:
            counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
        ok = exit_code == 0 and not counts.get("failed") and not counts.get("error")
        out["steps"].append({"name": step["name"], "rc": exit_code, "ok": ok, "counts": counts,
                             "seconds": round(time.monotonic() - t0, 1),
                             "log": str(log_path.relative_to(run_dir)), "tests": tests})
        print(f"[{stage}] {step['name']}: rc={exit_code} {counts} {time.monotonic() - t0:.1f}s", flush=True)
    out["seconds"] = round(time.monotonic() - t_stage, 1)
    out["ok"] = all(s["ok"] for s in out["steps"])
    out["counts"] = totals(out["steps"])
    return out


def totals(steps) -> dict:
    c = {}
    for s in steps:
        for k, v in s["counts"].items():
            c[k] = c.get(k, 0) + v
    return c


def quickstart_script(readme: str):
    """Every ```sh or ```bash block of the `## Quickstart` section, in order, joined; None if
    there is none. The section ends at the next `## ` heading, so a block further down the
    README is never run as the quickstart."""
    section = re.search(r"^## Quickstart\b.*?$(.*?)(?=^## |\Z)", readme, flags=re.S | re.M)
    if not section:
        return None
    blocks = re.findall(r"^```(?:sh|bash)[ \t]*\n(.*?)^```", section.group(1), flags=re.S | re.M)
    return "".join(blocks) if blocks else None


def quickstart(src: Path, uv: str, python: str, run_dir: Path, deadline: Deadline) -> dict:
    """Run the README quickstart code blocks verbatim in a fresh venv."""
    block = quickstart_script((src / "README.md").read_text())
    log_path = run_dir / "logs" / f"quickstart-py{python}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if block is None:
        log_path.write_text("no ```sh or ```bash block under ## Quickstart in README.md\n")
        return {"python": python, "ok": False, "rc": "no-block", "log": str(log_path.relative_to(run_dir))}
    venv = run_dir / "envs" / f"quickstart-{python}"
    # Each Python gets its own clean clone: the block makes .venv/ in the tree it runs in.
    qsrc = run_dir / "envs" / f"quickstart-src-{python}"
    sh(["git", "clone", "--quiet", str(src), str(qsrc)])
    script = run_dir / f"quickstart-py{python}.sh"
    script.write_text("set -euxo pipefail\nsource %s/bin/activate\ncd %s\n%s" %
                      (shlex.quote(str(venv)), shlex.quote(str(qsrc)), block))
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        # A plain `python -m venv` with pip, as a user would make it (no uv, no extras).
        r = subprocess.run([uv, "python", "find", "--no-project", python], capture_output=True, text=True)
        base = r.stdout.strip() or f"python{python}"
        exit_code = run_group([base, "-m", "venv", str(venv)], 600, deadline, stdout=log,
                       stderr=subprocess.STDOUT)
        if exit_code == 0:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "FASTRSI_"))}
            env["CUDA_VISIBLE_DEVICES"] = ""
            exit_code = run_group(["bash", str(script)], 1800, deadline, stdout=log,
                           stderr=subprocess.STDOUT, env=env, cwd=qsrc)
    return {"python": python, "ok": exit_code == 0, "rc": exit_code, "seconds": round(time.monotonic() - t0, 1),
            "log": str(log_path.relative_to(run_dir))}


# ---------------------------------------------------------------- status

def fmt_counts(c: dict) -> str:
    order = ["passed", "failed", "error", "skipped", "xfailed"]
    return ", ".join(f"{c[k]} {k}" for k in order if c.get(k))


def build_status(summary: dict, logs_url_base: str) -> dict:
    if summary.get("refused"):
        state, desc = "error", "refused: " + summary["refused"]
    elif summary.get("infra_error"):
        state, desc = "error", "runner error: " + summary["infra_error"]
    else:
        stages = summary["stages"]
        state = "success" if all(s["ok"] for s in stages.values()) and \
            all(q["ok"] for q in summary.get("quickstart", [])) else "failure"
        parts = [f"{name}: {fmt_counts(s['counts'])}" for name, s in stages.items()]
        if summary.get("quickstart"):
            qs = summary["quickstart"]
            parts.append("quickstart %d/%d ok" % (sum(q["ok"] for q in qs), len(qs)))
        m, s = divmod(int(summary["seconds"]), 60)
        desc = "; ".join(parts) + f"; {m}m{s:02d}s"
    target = f"{logs_url_base.rstrip('/')}/{summary['run_id']}/summary.md" if logs_url_base else None
    st = {"state": state, "context": CONTEXT, "description": desc[:140]}
    if target:
        st["target_url"] = target
    return st


def post_status(sha: str, status: dict):
    argv = ["gh", "api", "--method", "POST", f"repos/{REPO}/statuses/{sha}"]
    for k, v in status.items():
        argv += ["-f", f"{k}={v}"]
    sh(argv, capture_output=True)


def write_md(summary: dict, status: dict, path: Path):
    L = [f"# b200-gpu CI {summary['sha'][:12]}", "",
         f"- sha: `{summary['sha']}`", f"- trusted via branches: {', '.join(summary.get('branches', [])) or '-'}",
         f"- host: {summary['host']}  GPUs: `{summary.get('gpus', '-')}`  python: {summary.get('python')}",
         f"- started: {summary['started']}  wall: {summary.get('seconds')} s", ""]
    for name, st in summary.get("stages", {}).items():
        L += [f"## {name} ({'ok' if st['ok'] else 'FAILED'}, {st['seconds']} s): {fmt_counts(st['counts'])}", "",
              "| step | rc | passed | failed | error | skipped | s |", "|---|---|---|---|---|---|---|"]
        for s in st["steps"]:
            c = s["counts"]
            L.append(f"| {s['name']} | {s['rc']} | {c.get('passed', 0)} | {c.get('failed', 0)} | "
                     f"{c.get('error', 0)} | {c.get('skipped', 0)} | {s['seconds']} |")
        bad = [(s["name"], t) for s in st["steps"] for t in s["tests"] if t["outcome"] in ("failed", "error")]
        if bad:
            L += ["", "Failures:", ""] + [f"- `{n}` {t['outcome']}: `{t['id']}`" for n, t in bad]
        L.append("")
    if "newly_run" in summary:
        L += [f"## Cases skipped in cpu that ran in gpu: {len(summary['newly_run'])}", ""]
        L += [f"- `{i}`: {o}" for i, o in summary["newly_run"]] + [""]
        L += [f"## Still skipped in gpu: {len(summary['still_skipped'])}", ""]
        L += [f"- `{i}`: {r}" for i, r in summary["still_skipped"]] + [""]
    for q in summary.get("quickstart", []):
        L.append(f"- quickstart python {q['python']}: {'ok' if q['ok'] else 'FAILED'} (rc {q['rc']}, log `{q['log']}`)")
    L += ["", "## Status of this run", "", "```json", json.dumps(status, indent=1), "```", ""]
    if "posted_status" in summary:
        posted = summary["posted_status"]
        L += [f"- status to post: {posted['state'] + ': ' + posted['description'] if posted else 'none'}",
              f"- result record: `{json.dumps(summary['result_record'])}`",
              ""]
    path.write_text("\n".join(L))


# ---------------------------------------------------------------- results

def write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def previous_attempts(result: Path) -> int:
    try:
        return int(json.loads(result.read_text()).get("attempts", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def settle(summary: dict, status: dict, result: Path, max_attempts: int):
    """What this run leaves behind: (result record, status to post or None).

    The loop reruns a SHA until its record is final:
    - refused: not final, attempts unchanged. No final status; a pending status already
      posted is updated to say the run will be retried.
    - runner error before the SHA was verified as trusted (fetch, mirror): not final and not
      counted, so an outage never retires a head.
    - runner error after that: one more attempt; final (and its error status posted) only on
      the last of max_attempts, otherwise the status stays pending.
    - test result: final, with its status.
    Nothing is posted for a SHA that was not verified as trusted (no pending posted)."""
    trusted = summary.get("pending_posted", False)
    attempts = previous_attempts(result)
    record = {"run_dir": summary["run_dir"], "status": status, "attempts": attempts, "final": True}
    if summary.get("refused"):
        retry = {"state": "pending", "context": CONTEXT,
                 "description": ("not run yet, will retry: " + summary["refused"])[:140]}
        return dict(record, refused=summary["refused"], final=False), (retry if trusted else None)
    if summary.get("infra_error") and "branches" not in summary:
        return dict(record, final=False), None
    if summary.get("infra_error"):
        record.update(attempts=attempts + 1, final=attempts + 1 >= max_attempts)
        if record["final"]:
            return record, (status if trusted else None)
        retry = {"state": "pending", "context": CONTEXT,
                 "description": (f"runner error, will retry ({attempts + 1}/{max_attempts}): "
                                 + summary["infra_error"])[:140]}
        return record, (retry if trusted else None)
    return record, status


def on_sigterm(signum, frame):
    raise Stopped("stopped by SIGTERM")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--source", default=f"https://github.com/{REPO}.git",
                    help="git URL or bundle to fetch branch heads from")
    ap.add_argument("--work", default=str(Path.home() / "fastrsi-ci"))
    ap.add_argument("--pr", type=int, help="also require this PR to be same-repo with head == sha (uses gh)")
    ap.add_argument("--gpus", default="auto:2", help="CUDA_VISIBLE_DEVICES value, or auto:N idle GPUs")
    ap.add_argument("--gpu-pool", default="0,1,2,3", help="GPU indices this runner may use (B200 split)")
    ap.add_argument("--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS)
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--quickstart-pythons", default="3.11,3.12", help="requires-python is >=3.11")
    ap.add_argument("--vllm-python", default="", help="python with vLLM for FASTRSI_VLLM_PYTHON (gpu stage)")
    ap.add_argument("--stages", default="cpu,gpu,quickstart")
    ap.add_argument("--uv", default=shutil.which("uv") or str(Path.home() / ".local/bin/uv"))
    ap.add_argument("--timeout", type=int, default=1200, help="per step, seconds")
    ap.add_argument("--deadline-seconds", type=int, default=0,
                    help="stop the whole run after this many seconds (0: no limit)")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="runner errors on one SHA before its error status is final")
    ap.add_argument("--logs-url-base", default="", help="public base URL where run dirs are served")
    ap.add_argument("--keep-envs", action="store_true")
    ap.add_argument("--post", action="store_true", help="post the status to GitHub (off by default)")
    a = ap.parse_args(argv)

    if not re.fullmatch(r"[0-9a-f]{40}", a.sha):
        sys.exit("--sha must be a full 40-hex commit id")
    signal.signal(signal.SIGTERM, on_sigterm)
    work = Path(a.work).expanduser()
    started = now()
    run_id = f"{a.sha[:12]}-{started:%Y%m%dT%H%M%SZ}"
    run_dir = work / "runs" / run_id
    run_dir.mkdir(parents=True)
    # The loop reads this to stop a run that outlived its wait (SIGTERM, handled above).
    pid_file = work / "active" / f"{a.sha}.pid"
    write_atomic(pid_file, str(os.getpid()))
    summary = {"sha": a.sha, "run_id": run_id, "run_dir": str(run_dir), "host": socket.gethostname(),
               "python": a.python, "started": started.isoformat(timespec="seconds"),
               "argv": sys.argv if argv is None else argv, "stages": {}}
    t0 = time.monotonic()
    deadline = Deadline(a.deadline_seconds)
    stages = a.stages.split(",")
    gpu_locks = []
    try:
        mirror = sync_mirror(work, a.source)
        branches = trusted_branches(mirror, a.sha)
        if not branches:
            raise Refused(f"{a.sha[:12]} is on no branch of {REPO} (fork PR or unknown commit)")
        summary["branches"] = branches
        if a.pr:
            summary["pr"] = check_pr(a.pr, a.sha)
        if "gpu" in stages:
            # Locked before anything is posted or built, and held until the run ends.
            summary["gpus"], gpu_locks = pick_gpus(a.gpus, a.gpu_pool, work / "locks", a.nvidia_smi)
        if a.post:
            post_status(a.sha, {"state": "pending", "context": CONTEXT, "description": "running on B200"})
            summary["pending_posted"] = True
        src = run_dir / "src"
        sh(["git", "clone", "--quiet", "--no-checkout", str(mirror), str(src)])
        sh(["git", "-C", str(src), "-c", "advice.detachedHead=false", "checkout", "--quiet", a.sha])
        envs = run_dir / "envs"
        with open(run_dir / "env-setup.log", "w") as log:
            if "cpu" in stages:
                summary["cpu_env_s"] = round(make_env(a.uv, envs / "cpu", a.python, [CPU_EXTRA], log,
                                                      deadline), 1)
            if "gpu" in stages:
                summary["gpu_env_s"] = round(make_env(a.uv, envs / "gpu", a.python,
                                                      [TORCH_SPEC, GPU_EXTRA], log, deadline), 1)
        base_env = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if "cpu" in stages:
            summary["stages"]["cpu"] = run_stage(
                "cpu", src, str(envs / "cpu/bin/python"),
                {**base_env, "CUDA_VISIBLE_DEVICES": "", "FASTRSI_VLLM_PYTHON": ""}, run_dir, a.timeout,
                deadline)
        if "gpu" in stages:
            check_still_idle(summary["gpus"], a.gpus, a.nvidia_smi)
            env = {**base_env, "CUDA_VISIBLE_DEVICES": summary["gpus"], "FASTRSI_TEST_GPU": "1"}
            if a.vllm_python:
                # The vLLM python may live in a user site; give it a shim that drops our isolation.
                shim = run_dir / "vllm-python"
                shim.write_text("#!/bin/sh\nunset PYTHONNOUSERSITE\nexec %s \"$@\"\n" % shlex.quote(a.vllm_python))
                shim.chmod(0o755)
                env["FASTRSI_VLLM_PYTHON"] = str(shim)
                summary["vllm_python"] = a.vllm_python
            summary["stages"]["gpu"] = run_stage("gpu", src, str(envs / "gpu/bin/python"), env, run_dir,
                                                 a.timeout, deadline)
        if "cpu" in stages and "gpu" in stages:
            cpu = {t["id"]: t for s in summary["stages"]["cpu"]["steps"] for t in s["tests"]}
            gpu = {t["id"]: t for s in summary["stages"]["gpu"]["steps"] for t in s["tests"]}
            summary["newly_run"] = sorted((i, gpu[i]["outcome"]) for i, t in cpu.items()
                                          if t["outcome"] == "skipped" and i in gpu
                                          and gpu[i]["outcome"] != "skipped")
            summary["still_skipped"] = sorted((i, t["detail"]) for i, t in gpu.items() if t["outcome"] == "skipped")
        if "quickstart" in stages:
            summary["quickstart"] = [quickstart(src, a.uv, version, run_dir, deadline)
                                     for version in a.quickstart_pythons.split(",")]
    except Refused as e:
        summary["refused"] = str(e)
    except Exception as e:  # infrastructure (or Stopped), not a test result
        summary["infra_error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        for handle in gpu_locks:
            handle.close()
        summary["seconds"] = round(time.monotonic() - t0, 1)
        if not a.keep_envs:
            shutil.rmtree(run_dir / "envs", ignore_errors=True)
    status = build_status(summary, a.logs_url_base)
    summary["status"] = status
    result = work / "results" / f"{a.sha}.json"
    record, to_post = settle(summary, status, result, a.max_attempts)
    summary["result_record"] = record
    summary["posted_status"] = to_post
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_md(summary, status, run_dir / "summary.md")
    write_atomic(result, json.dumps(record))
    pid_file.unlink(missing_ok=True)
    print(f"run dir: {run_dir}")
    print("status for repos/%s/statuses/%s:" % (REPO, a.sha))
    print(json.dumps(status, indent=1))
    if not record["final"]:
        print("result is not final: the loop will run this SHA again")
    if a.post and to_post is not None:
        post_status(a.sha, to_post)
        print("posted:", to_post["state"])
    sys.exit({"success": 0, "failure": 1}.get(status["state"], 2))


if __name__ == "__main__":
    main()
