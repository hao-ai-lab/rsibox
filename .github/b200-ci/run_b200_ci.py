#!/usr/bin/env python3
"""B200 CI: test one trusted commit of GindaChen/fastrsi on the B200 machine.

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
Exit code: 0 success, 1 test failure, 2 infrastructure error or refusal.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = "GindaChen/fastrsi"
CONTEXT = "b200-gpu"
HERE = Path(__file__).resolve().parent
TORCH_SPEC = ["torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128",
              "--extra-index-url", "https://pypi.org/simple"]
GPU_EXTRA = ["pytest", "numpy", "nvidia-ml-py"]
CPU_EXTRA = ["pytest"]  # cpu-tests.yml installs only pytest, and only for vllm-services


class Refused(Exception):
    pass


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

def pick_gpus(spec: str, pool: str) -> str:
    """`spec` is an explicit CUDA_VISIBLE_DEVICES value or auto:N. auto picks N idle GPUs,
    only from `pool` (the indices this runner is allowed on the shared machine)."""
    allowed = set(pool.split(",")) if pool else None
    if not spec.startswith("auto"):
        if allowed and not set(spec.split(",")) <= allowed:
            raise Refused(f"--gpus {spec} is outside the allowed pool {pool}")
        return spec
    want = int(spec.split(":", 1)[1]) if ":" in spec else 1
    out = sh(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
              "--format=csv,noheader,nounits"], capture_output=True).stdout
    idle = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        if (allowed is None or idx in allowed) and int(mem) < 500 and int(util) == 0:
            idle.append(idx)
    if len(idle) < want:
        raise Refused(f"need {want} idle GPUs in pool {pool or 'all'}, found {idle}")
    return ",".join(idle[:want])


# ---------------------------------------------------------------- envs

def make_env(uv: str, path: Path, python: str, pkgs: list, log) -> float:
    t0 = time.monotonic()
    sh([uv, "venv", "--quiet", "--python", python, str(path)], stdout=log, stderr=subprocess.STDOUT)
    for group in pkgs:
        sh([uv, "pip", "install", "--quiet", "--python", str(path / "bin" / "python"), *group],
           stdout=log, stderr=subprocess.STDOUT)
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


def run_stage(stage, src: Path, py: str, env_extra: dict, run_dir: Path, timeout: int) -> dict:
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
            try:
                rc = subprocess.run(step["argv"], cwd=src / step["cwd"], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        tests = parse_results(step, res)
        if step["kind"] == "cmd":
            tests = [{"id": step["name"], "outcome": "passed" if rc == 0 else "failed",
                      "seconds": round(time.monotonic() - t0, 3), "detail": ""}]
        counts = {}
        for t in tests:
            counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
        ok = rc == 0 and not counts.get("failed") and not counts.get("error")
        out["steps"].append({"name": step["name"], "rc": rc, "ok": ok, "counts": counts,
                             "seconds": round(time.monotonic() - t0, 1),
                             "log": str(log_path.relative_to(run_dir)), "tests": tests})
        print(f"[{stage}] {step['name']}: rc={rc} {counts} {time.monotonic() - t0:.1f}s", flush=True)
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


def quickstart(src: Path, uv: str, python: str, run_dir: Path) -> dict:
    """Run the README quickstart code block verbatim in a fresh venv."""
    text = (src / "README.md").read_text()
    m = re.search(r"^## Quickstart.*?^```sh\n(.*?)^```", text, flags=re.S | re.M)
    log_path = run_dir / "logs" / f"quickstart-py{python}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not m:
        log_path.write_text("no ```sh block under ## Quickstart in README.md\n")
        return {"python": python, "ok": False, "rc": "no-block", "log": str(log_path.relative_to(run_dir))}
    venv = run_dir / "envs" / f"quickstart-{python}"
    # Each Python gets its own clean clone: the block makes .venv/ in the tree it runs in.
    qsrc = run_dir / "envs" / f"quickstart-src-{python}"
    sh(["git", "clone", "--quiet", str(src), str(qsrc)])
    script = run_dir / f"quickstart-py{python}.sh"
    script.write_text("set -euxo pipefail\nsource %s/bin/activate\ncd %s\n%s" %
                      (shlex.quote(str(venv)), shlex.quote(str(qsrc)), m.group(1)))
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        # A plain `python -m venv` with pip, as a user would make it (no uv, no extras).
        r = subprocess.run([uv, "python", "find", "--no-project", python], capture_output=True, text=True)
        base = r.stdout.strip() or f"python{python}"
        rc = subprocess.run([base, "-m", "venv", str(venv)], stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "FASTRSI_"))}
            env["CUDA_VISIBLE_DEVICES"] = ""
            rc = subprocess.run(["bash", str(script)], stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=qsrc).returncode
    return {"python": python, "ok": rc == 0, "rc": rc, "seconds": round(time.monotonic() - t0, 1),
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
    L += ["", "## Status it would post", "", "```json", json.dumps(status, indent=1), "```", ""]
    path.write_text("\n".join(L))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--source", default=f"https://github.com/{REPO}.git",
                    help="git URL or bundle to fetch branch heads from")
    ap.add_argument("--work", default=str(Path.home() / "fastrsi-ci"))
    ap.add_argument("--pr", type=int, help="also require this PR to be same-repo with head == sha (uses gh)")
    ap.add_argument("--gpus", default="auto:2", help="CUDA_VISIBLE_DEVICES value, or auto:N idle GPUs")
    ap.add_argument("--gpu-pool", default="0,1,2,3", help="GPU indices this runner may use (B200 split)")
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--quickstart-pythons", default="3.11,3.12", help="requires-python is >=3.11")
    ap.add_argument("--vllm-python", default="", help="python with vLLM for FASTRSI_VLLM_PYTHON (gpu stage)")
    ap.add_argument("--stages", default="cpu,gpu,quickstart")
    ap.add_argument("--uv", default=shutil.which("uv") or str(Path.home() / ".local/bin/uv"))
    ap.add_argument("--timeout", type=int, default=1200, help="per step, seconds")
    ap.add_argument("--logs-url-base", default="", help="public base URL where run dirs are served")
    ap.add_argument("--keep-envs", action="store_true")
    ap.add_argument("--post", action="store_true", help="post the status to GitHub (off by default)")
    a = ap.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", a.sha):
        sys.exit("--sha must be a full 40-hex commit id")
    work = Path(a.work).expanduser()
    started = now()
    run_id = f"{a.sha[:12]}-{started:%Y%m%dT%H%M%SZ}"
    run_dir = work / "runs" / run_id
    run_dir.mkdir(parents=True)
    summary = {"sha": a.sha, "run_id": run_id, "host": socket.gethostname(), "python": a.python,
               "started": started.isoformat(timespec="seconds"), "argv": sys.argv, "stages": {}}
    t0 = time.monotonic()
    stages = a.stages.split(",")
    try:
        mirror = sync_mirror(work, a.source)
        branches = trusted_branches(mirror, a.sha)
        if not branches:
            raise Refused(f"{a.sha[:12]} is on no branch of {REPO} (fork PR or unknown commit)")
        summary["branches"] = branches
        if a.pr:
            summary["pr"] = check_pr(a.pr, a.sha)
        if a.post:
            post_status(a.sha, {"state": "pending", "context": CONTEXT, "description": "running on B200"})
        src = run_dir / "src"
        sh(["git", "clone", "--quiet", "--no-checkout", str(mirror), str(src)])
        sh(["git", "-C", str(src), "-c", "advice.detachedHead=false", "checkout", "--quiet", a.sha])
        envs = run_dir / "envs"
        with open(run_dir / "env-setup.log", "w") as log:
            if "cpu" in stages:
                summary["cpu_env_s"] = round(make_env(a.uv, envs / "cpu", a.python, [CPU_EXTRA], log), 1)
            if "gpu" in stages:
                summary["gpus"] = pick_gpus(a.gpus, a.gpu_pool)
                summary["gpu_env_s"] = round(make_env(a.uv, envs / "gpu", a.python,
                                                      [TORCH_SPEC, GPU_EXTRA], log), 1)
        base_env = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if "cpu" in stages:
            summary["stages"]["cpu"] = run_stage(
                "cpu", src, str(envs / "cpu/bin/python"),
                {**base_env, "CUDA_VISIBLE_DEVICES": "", "FASTRSI_VLLM_PYTHON": ""}, run_dir, a.timeout)
        if "gpu" in stages:
            env = {**base_env, "CUDA_VISIBLE_DEVICES": summary["gpus"], "FASTRSI_TEST_GPU": "1"}
            if a.vllm_python:
                # The vLLM python may live in a user site; give it a shim that drops our isolation.
                shim = run_dir / "vllm-python"
                shim.write_text("#!/bin/sh\nunset PYTHONNOUSERSITE\nexec %s \"$@\"\n" % shlex.quote(a.vllm_python))
                shim.chmod(0o755)
                env["FASTRSI_VLLM_PYTHON"] = str(shim)
                summary["vllm_python"] = a.vllm_python
            summary["stages"]["gpu"] = run_stage("gpu", src, str(envs / "gpu/bin/python"), env, run_dir, a.timeout)
        if "cpu" in stages and "gpu" in stages:
            cpu = {t["id"]: t for s in summary["stages"]["cpu"]["steps"] for t in s["tests"]}
            gpu = {t["id"]: t for s in summary["stages"]["gpu"]["steps"] for t in s["tests"]}
            summary["newly_run"] = sorted((i, gpu[i]["outcome"]) for i, t in cpu.items()
                                          if t["outcome"] == "skipped" and i in gpu
                                          and gpu[i]["outcome"] != "skipped")
            summary["still_skipped"] = sorted((i, t["detail"]) for i, t in gpu.items() if t["outcome"] == "skipped")
        if "quickstart" in stages:
            summary["quickstart"] = [quickstart(src, a.uv, p, run_dir) for p in a.quickstart_pythons.split(",")]
    except Refused as e:
        summary["refused"] = str(e)
    except Exception as e:  # infrastructure, not a test result
        summary["infra_error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        summary["seconds"] = round(time.monotonic() - t0, 1)
        if not a.keep_envs:
            shutil.rmtree(run_dir / "envs", ignore_errors=True)
    status = build_status(summary, a.logs_url_base)
    summary["status"] = status
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_md(summary, status, run_dir / "summary.md")
    (work / "results").mkdir(exist_ok=True)
    (work / "results" / f"{a.sha}.json").write_text(json.dumps({"run_dir": str(run_dir), "status": status}))
    print(f"run dir: {run_dir}")
    print("status it would post to repos/%s/statuses/%s:" % (REPO, a.sha))
    print(json.dumps(status, indent=1))
    if a.post:
        post_status(a.sha, status)
        print("posted")
    sys.exit({"success": 0, "failure": 1}.get(status["state"], 2))


if __name__ == "__main__":
    main()
