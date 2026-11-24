#!/usr/bin/env python3
"""Modal GPU CI: test one trusted commit of hao-ai-lab/rsibox on Modal GPUs.

Stdlib only. Fetches branch heads (never refs/pull/*) into a bare mirror, refuses
any SHA not contained on a branch of the repo (fork PRs), exports only that trusted
SHA as a clean `git archive` tarball (zero GitHub tokens enter Modal containers),
and invokes `.github/modal-ci/modal_app.py` to run the core + multi-engine suites
in parallel with `vllm-services` on cached Modal GPU images.

Optional stages `--stages cpu,gpu,quickstart` preserve local CPU/quickstart parity.
Context posted with `--post`: `modal-gpu`.

Retries: `<work>/results/<sha>.json` records `final`; the loop reruns a SHA until
it is final. A refusal is never final. A runner error before trust is verified is
never final and not counted; after trust it increments `attempts` up to
`--max-attempts` before posting `error`.
Exit code: 0 success, 1 test failure, 2 refusal or infrastructure error.
"""
import argparse
import datetime
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

REPO = "hao-ai-lab/rsibox"
CONTEXT = "modal-gpu"
HERE = Path(__file__).resolve().parent
CPU_EXTRA = ["pytest"]


class Refused(Exception):
    """Not run, and not a result: the SHA or PR failed trust checks."""


class Stopped(Exception):
    """The runner was told to stop (SIGTERM) or ran past --deadline-seconds."""


class Deadline:
    """Caps a whole run (--deadline-seconds). A step cut by it raises Stopped."""

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
    """Run one subprocess in its own process group; kill the whole group on timeout/signal."""
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
    # Branch heads only. Fork PR commits live only under refs/pull/*, never fetched here.
    sh(["git", "-C", str(mirror), "fetch", "--quiet", "--prune", "--no-tags", source,
        "+refs/heads/*:refs/heads/*"])
    return mirror


def trusted_branches(mirror: Path, sha: str) -> list:
    r = sh(["git", "-C", str(mirror), "cat-file", "-e", sha + "^{commit}"],
           check=False, capture_output=True)
    if r.returncode != 0:
        return []
    out = sh(["git", "-C", str(mirror), "branch", "--format=%(refname:short)", "--contains", sha],
             capture_output=True).stdout.split()
    return out


def check_pr(pr: int, sha: str) -> dict:
    """Require same-repo PR whose head SHA equals `sha`; refuses fork PRs."""
    r = sh(["gh", "api", f"repos/{REPO}/pulls/{pr}"], capture_output=True)
    d = json.loads(r.stdout)
    head_repo = (d.get("head") or {}).get("repo") or {}
    if head_repo.get("full_name") != REPO:
        raise Refused(f"PR #{pr} head is {head_repo.get('full_name')!r}, not {REPO}: fork PRs are not run")
    if d["head"]["sha"] != sha:
        raise Refused(f"PR #{pr} head is {d['head']['sha']}, not {sha}")
    return {"pr": pr, "head_ref": d["head"]["ref"], "base_ref": d["base"]["ref"]}


def export_archive(mirror: Path, sha: str, tarball: Path) -> Path:
    tarball.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "-C", str(mirror), "archive", "--format=tar.gz", "-o", str(tarball), sha])
    return tarball


# ---------------------------------------------------------------- Modal runner

def resolve_modal_cmd(modal_cmd: str, uv: str) -> list[str]:
    if modal_cmd.strip():
        return shlex.split(modal_cmd)
    on_path = shutil.which("modal")
    if on_path:
        return [on_path]
    uv_bin = shutil.which(uv) if uv else None
    if not uv_bin and uv and Path(uv).is_file():
        uv_bin = uv
    if uv_bin:
        return [uv_bin, "run", "--no-project", "--with", "modal", "modal"]
    raise RuntimeError("modal CLI not found (install modal or uv, or pass --modal-cmd)")


def run_modal_gpu(
    modal_cmd: list[str],
    tarball: Path,
    run_dir: Path,
    gpu: str,
    timeout: int,
    deadline: Deadline,
    vllm_model: str = "",
    suites_csv: str = "",
) -> tuple[dict, dict]:
    """Invoke Modal GPU CI app on the trusted tarball and collect structured results + logs."""
    modal_json = run_dir / "modal_results.json"
    logs_dir = run_dir / "logs" / "gpu"
    logs_dir.mkdir(parents=True, exist_ok=True)
    modal_log_path = run_dir / "logs" / "modal-run.log"

    remote_timeout = max(1800, min(7200, timeout * 3))
    argv = [
        *modal_cmd,
        "run",
        str(HERE / "modal_app.py"),
        "--tarball-path",
        str(tarball),
        "--output-json",
        str(modal_json),
        "--gpu",
        gpu,
        "--timeout",
        str(timeout),
    ]
    if vllm_model:
        argv += ["--vllm-model", vllm_model]
    if suites_csv:
        argv += ["--suites", suites_csv]

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("RSIBOX_", "PYTHONPATH", "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API"))
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["RSIBOX_MODAL_GPU"] = gpu
    env["RSIBOX_MODAL_TIMEOUT"] = str(remote_timeout)

    t_stage = time.monotonic()
    modal_timeout = max(timeout * 3, 2400)
    with open(modal_log_path, "w") as mlog:
        mlog.write("$ %s\n" % shlex.join(argv))
        mlog.flush()
        exit_code = run_group(
            argv,
            modal_timeout,
            deadline,
            env=env,
            stdout=mlog,
            stderr=subprocess.STDOUT,
        )

    if exit_code == "timeout":
        raise RuntimeError(f"modal run timed out after {modal_timeout}s")
    if not modal_json.exists():
        tail = ""
        try:
            lines = [ln.strip() for ln in modal_log_path.read_text(errors="replace").splitlines() if ln.strip()]
            if lines:
                tail = f": {lines[-1][:160]}"
        except OSError:
            pass
        raise RuntimeError(f"modal run exited {exit_code} without writing {modal_json.name}{tail}")

    try:
        data = json.loads(modal_json.read_text())
    except ValueError as exc:
        raise RuntimeError(f"invalid {modal_json.name}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise RuntimeError(f"malformed {modal_json.name}: missing steps list")
    if data.get("infra_error"):
        raise RuntimeError(str(data["infra_error"]))
    if exit_code not in (0, 1):
        raise RuntimeError(f"modal run exited {exit_code}")

    steps = []
    for raw_step in data["steps"]:
        step = dict(raw_step)
        log_text = step.pop("log_text", "")
        log_path = logs_dir / f"{step['name']}.log"
        log_path.write_text(log_text)
        step["log"] = str(log_path.relative_to(run_dir))
        steps.append(step)
        print(
            f"[gpu] {step['name']}: rc={step['rc']} {step.get('counts', {})} {step.get('seconds', 0)}s",
            flush=True,
        )

    stage = {
        "stage": "gpu",
        "steps": steps,
        "seconds": round(time.monotonic() - t_stage, 1),
        "ok": bool(steps) and all(s["ok"] for s in steps),
        "counts": totals(steps),
    }
    return stage, data.get("gpu_info", {})


# ---------------------------------------------------------------- optional CPU / quickstart

def make_env(uv: str, path: Path, python: str, pkgs: list, log, deadline: Deadline) -> float:
    t0 = time.monotonic()
    commands = [[uv, "venv", "--quiet", "--python", python, str(path)]]
    commands += [
        [uv, "pip", "install", "--quiet", "--python", str(path / "bin" / "python"), *group]
        for group in pkgs
    ]
    for argv in commands:
        exit_code = run_group(argv, 1800, deadline, stdout=log, stderr=subprocess.STDOUT)
        if exit_code != 0:
            raise RuntimeError(f"{shlex.join(argv[:3])} exited {exit_code}")
    return time.monotonic() - t0


def cpu_suites(src: Path, py: str, res: Path):
    """Local CPU parity suites."""
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
    steps = [
        ut("core-conformance-cases", "conformance", ".", cwd="core/livelab"),
        ut("core-kernel", "core/livelab/kernel", "core/livelab"),
        ut("core-forksafe", "core/livelab", p="test_forksafe.py"),
        ut("core-rsibox", "core/livelab/rsibox", "core/livelab"),
        ut("core-survey", "core/livelab/survey", "core/livelab"),
        cmd("core-doctor", [py, "-m", "rsibox", "doctor"], env={"PYTHONPATH": str(src / "core/livelab")}),
        cmd("core-conformance-toylab", [py, "-m", "conformance", "--lab", "kernel.toy_lab:ToyLab"],
            cwd="core/livelab"),
        ut("template-adapter", "template/livelab/adapter", "template/livelab"),
        ut("fastvideo-adapter", "repos/fastvideo/livelab/adapter", "repos/fastvideo/livelab"),
        ut("fastvideo-services", "repos/fastvideo/livelab/services", "repos/fastvideo/livelab"),
        cmd("fastvideo-conformance",
            [py, "-m", "conformance", "--lab", "adapter.toy_lab:ToyFastVideoLab"],
            cwd="core/livelab", env={"PYTHONPATH": fv}),
        ut("vllm-adapter", "repos/vllm/livelab/adapter", "repos/vllm/livelab"),
        {"name": "vllm-services", "kind": "pytest",
         "argv": [py, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider",
                  f"--junitxml={res / 'vllm-services.xml'}", "repos/vllm/livelab/services"],
         "cwd": "."},
        ut("megatron-livelab", "repos/megatron/livelab/rsibox_megatron", "repos/megatron/livelab"),
    ]
    if (src / "repos/megatron/tools").is_dir():
        steps.append(ut("megatron-tools", "repos/megatron/tools"))
    steps += [
        ut("verl-adapter", "repos/verl/livelab/adapter", "repos/verl/livelab"),
        ut("verl-tools", "repos/verl/tools", "repos/verl"),
        cmd("verl-conformance", [py, "-m", "rsibox", "conformance", "verl"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("miles-adapter", "repos/miles/livelab/adapter", "repos/miles/livelab"),
        cmd("miles-conformance", [py, "-m", "rsibox", "conformance", "miles"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("sglang-adapter", "repos/sglang/livelab/adapter", "repos/sglang/livelab"),
        ut("sglang-services", "repos/sglang/livelab/services", "repos/sglang/livelab"),
        cmd("sglang-conformance", [py, "-m", "rsibox", "conformance", "sglang"],
            env={"PYTHONPATH": str(src / "core/livelab")}),
        ut("review-checkers", ".github/review", ".github/review"),
        ut("scripts-tests", "scripts", p="test_*.py"),
    ]
    return steps


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
                    outcome = "failed" if child.tag == "failure" else "error"
                    detail = (child.get("message") or "")[:2000]
                elif child.tag == "skipped":
                    xf = (child.get("type") or "") == "pytest.xfail"
                    outcome, detail = ("xfailed" if xf else "skipped"), child.get("message") or ""
            tests.append({"id": tid, "outcome": outcome, "seconds": float(tc.get("time") or 0),
                          "detail": detail})
        return tests
    return []


def run_stage(stage, src: Path, py: str, env_extra: dict, run_dir: Path, timeout: int,
              deadline: Deadline, suites_filter: set[str] | None = None) -> dict:
    res = run_dir / "results" / stage
    logs = run_dir / "logs" / stage
    res.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    out = {"stage": stage, "steps": []}
    t_stage = time.monotonic()
    steps = cpu_suites(src, py, res)
    if suites_filter is not None:
        known = {step["name"] for step in steps}
        unknown = sorted(suites_filter - known)
        if unknown:
            raise ValueError(f"unknown suite(s): {', '.join(unknown)}")
        steps = [step for step in steps if step["name"] in suites_filter]
    for step in steps:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("RSIBOX_", "PYTHON", "VIRTUAL_ENV",
                                    "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API", "MODAL_TOKEN_"))}
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
    out["ok"] = bool(out["steps"]) and all(s["ok"] for s in out["steps"])
    out["counts"] = totals(out["steps"])
    return out


def totals(steps) -> dict:
    c = {}
    for s in steps:
        for k, v in s.get("counts", {}).items():
            c[k] = c.get(k, 0) + v
    return c


def quickstart_script(readme: str):
    """Every ```sh or ```bash block of the `## Quickstart` section, in order, joined."""
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
    qsrc = run_dir / "envs" / f"quickstart-src-{python}"
    sh(["git", "clone", "--quiet", str(src), str(qsrc)])
    script = run_dir / f"quickstart-py{python}.sh"
    script.write_text("set -euxo pipefail\nsource %s/bin/activate\ncd %s\n%s" %
                      (shlex.quote(str(venv)), shlex.quote(str(qsrc)), block))
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        r = subprocess.run([uv, "python", "find", "--no-project", python], capture_output=True, text=True)
        base = r.stdout.strip() or f"python{python}"
        exit_code = run_group([base, "-m", "venv", str(venv)], 600, deadline, stdout=log,
                              stderr=subprocess.STDOUT)
        if exit_code == 0:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("PYTHON", "RSIBOX_",
                                        "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API", "MODAL_TOKEN_"))}
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
    gpu_info = summary.get("gpu_info", {})
    core_gi = gpu_info.get("core", {})
    vllm_gi = gpu_info.get("vllm", {})
    gi = core_gi or vllm_gi
    gpu_desc = f"`{summary.get('gpu', '-')}`"
    if gi.get("gpu_name"):
        gpu_desc += f" ({gi['gpu_name']}, driver {gi.get('driver_version', '?')})"
    L = [f"# modal-gpu CI {summary['sha'][:12]}", "",
         f"- sha: `{summary['sha']}`",
         f"- trusted via branches: {', '.join(summary.get('branches', [])) or '-'}",
         f"- host: {summary['host']}  Modal GPU: {gpu_desc}  python: {summary.get('python')}",
         f"- started: {summary['started']}  wall: {summary.get('seconds')} s", ""]
    if core_gi or vllm_gi:
        for role, info in (("core", core_gi), ("vllm", vllm_gi)):
            if info:
                L.append(
                    f"- {role} container: gpu={info.get('gpu_name', '?')} "
                    f"driver={info.get('driver_version', '?')} "
                    f"torch={info.get('torch_version', '?')} "
                    f"cuda={info.get('cuda_version', '?')}"
                    + (f" vllm={info['vllm_version']}" if info.get("vllm_version") else "")
                )
        L.append("")
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
    write_atomic(path, "\n".join(L))


# ---------------------------------------------------------------- results

def write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def previous_attempts(result: Path) -> int:
    try:
        return int(json.loads(result.read_text()).get("attempts", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def settle(summary: dict, status: dict, result: Path, max_attempts: int):
    """Return `(result_record, status_to_post_or_None)` implementing retry/finality rules."""
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
    ap.add_argument("--work", default=str(Path.home() / "rsibox-ci"))
    ap.add_argument("--pr", type=int, help="also require this PR to be same-repo with head == sha (uses gh)")
    ap.add_argument("--gpu", default=os.environ.get("MODAL_GPU", "L4"),
                    help="Modal GPU spec (default: L4)")
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--quickstart-pythons", default="3.11,3.12", help="requires-python is >=3.11")
    ap.add_argument("--vllm-model", default=os.environ.get("MODAL_VLLM_MODEL") or os.environ.get("RSIBOX_VLLM_MODEL", ""),
                    help="model path/id for RSIBOX_VLLM_MODEL (vllm-services E2E)")
    ap.add_argument("--suites", default="", help="optional comma-separated suite filter for Modal GPU stage")
    ap.add_argument("--stages", default="gpu", help="comma-separated stages: gpu (default), cpu, quickstart")
    ap.add_argument("--modal-cmd", default="", help="override Modal CLI command (auto-detected if empty)")
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
    work = Path(a.work).expanduser()
    started = now()
    run_id = f"{a.sha[:12]}-{started:%Y%m%dT%H%M%SZ}"
    run_dir = work / "runs" / run_id
    run_dir.mkdir(parents=True)
    pid_file = work / "active" / f"{a.sha}.pid"
    summary = {
        "sha": a.sha,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "host": socket.gethostname(),
        "gpu": a.gpu,
        "python": a.python,
        "started": started.isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else argv,
        "stages": {},
    }
    t0 = time.monotonic()
    deadline = Deadline(a.deadline_seconds)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    suites_list = [s.strip() for s in a.suites.split(",") if s.strip()]
    suites_filter = set(suites_list) if suites_list else None

    signal.signal(signal.SIGTERM, on_sigterm)
    write_atomic(pid_file, str(os.getpid()))
    completed_run = False
    stopped_during_finalize = False

    def defer_sigterm(signum, frame):
        nonlocal stopped_during_finalize
        stopped_during_finalize = True

    try:
        try:
            mirror = sync_mirror(work, a.source)
            branches = trusted_branches(mirror, a.sha)
            if not branches:
                raise Refused(f"{a.sha[:12]} is on no branch of {REPO} (fork PR or unknown commit)")
            summary["branches"] = branches
            if a.pr:
                summary["pr"] = check_pr(a.pr, a.sha)
            if a.post:
                post_status(a.sha, {"state": "pending", "context": CONTEXT,
                                    "description": f"running on Modal ({a.gpu})"})
                summary["pending_posted"] = True

            src = run_dir / "src"
            tarball = run_dir / "src.tar.gz"
            if "gpu" in stages:
                export_archive(mirror, a.sha, tarball)
            if "cpu" in stages or "quickstart" in stages:
                sh(["git", "clone", "--quiet", "--no-checkout", str(mirror), str(src)])
                sh(["git", "-C", str(src), "-c", "advice.detachedHead=false", "checkout", "--quiet", a.sha])

            base_env = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            if "cpu" in stages:
                envs = run_dir / "envs"
                with open(run_dir / "env-setup.log", "w") as log:
                    summary["cpu_env_s"] = round(make_env(a.uv, envs / "cpu", a.python, [CPU_EXTRA], log, deadline), 1)
                summary["stages"]["cpu"] = run_stage(
                    "cpu",
                    src,
                    str(envs / "cpu/bin/python"),
                    {**base_env, "CUDA_VISIBLE_DEVICES": "", "RSIBOX_VLLM_PYTHON": "",
                     "RSIBOX_VLLM_MODEL": "", "RSIBOX_GPU_E2E": "0"},
                    run_dir,
                    a.timeout,
                    deadline,
                    suites_filter=suites_filter,
                )

            if "gpu" in stages:
                modal_cmd = resolve_modal_cmd(a.modal_cmd, a.uv)
                gpu_stage, gpu_info = run_modal_gpu(
                    modal_cmd,
                    tarball,
                    run_dir,
                    a.gpu,
                    a.timeout,
                    deadline,
                    vllm_model=a.vllm_model,
                    suites_csv=a.suites,
                )
                summary["stages"]["gpu"] = gpu_stage
                summary["gpu_info"] = gpu_info
                if a.vllm_model:
                    summary["vllm_model"] = a.vllm_model

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
            completed_run = True
        except Refused as e:
            summary["refused"] = str(e)
        except Exception as e:
            summary["infra_error"] = f"{type(e).__name__}: {e}"[:300]
    except Stopped as e:
        if not summary.get("refused") and not summary.get("infra_error"):
            summary["infra_error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        # Defer SIGTERM until result records and summaries are persisted atomically.
        signal.signal(signal.SIGTERM, defer_sigterm)
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            if not a.keep_envs:
                shutil.rmtree(run_dir / "envs", ignore_errors=True)
                (run_dir / "src.tar.gz").unlink(missing_ok=True)
            if (not completed_run and stopped_during_finalize
                    and not summary.get("refused") and not summary.get("infra_error")):
                summary["infra_error"] = "Stopped: stopped by SIGTERM"
            summary["seconds"] = round(time.monotonic() - t0, 1)
            status = build_status(summary, a.logs_url_base)
            summary["status"] = status
            result = work / "results" / f"{a.sha}.json"
            record, to_post = settle(summary, status, result, a.max_attempts)
            summary["result_record"] = record
            summary["posted_status"] = to_post
            write_atomic(run_dir / "summary.json", json.dumps(summary, indent=1))
            write_md(summary, status, run_dir / "summary.md")
            write_atomic(result, json.dumps(record))
            pid_file.unlink(missing_ok=True)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

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
