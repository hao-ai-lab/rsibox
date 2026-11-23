#!/usr/bin/env python3
"""Offline golden tests for the Modal GPU CI runner, loop, and app structure.

Zero live Modal or GitHub calls. Uses a local git repository, a stand-in `--modal-cmd`
script, and mocked GitHub status/PR API calls.

Run: python3 .github/modal-ci/test_modal_ci.py -v
"""
import ast
import contextlib
import importlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal_ci_loop as loop   # noqa: E402
import run_modal_ci as runner  # noqa: E402

README = """# demo

## Quickstart

```sh
echo first >> quickstart.out
```

Then:

```bash
echo second >> quickstart.out
```

## Tests

```sh
echo WRONG_SECTION >> quickstart.out
```
"""


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def py_script(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)
    return str(path)


def load_modal_app_with_stub():
    """Import `modal_app` using an in-memory Modal 1.x stub (no modal package needed)."""
    modal_stub = types.ModuleType("modal")

    class FakeImage:
        def __init__(self, ops=None):
            self.ops = list(ops or [])

        @classmethod
        def debian_slim(cls, python_version="3.12"):
            return cls([("debian_slim", python_version)])

        def apt_install(self, *pkgs):
            return FakeImage([*self.ops, ("apt_install", pkgs)])

        def pip_install(self, *pkgs, **kw):
            return FakeImage([*self.ops, ("pip_install", pkgs, kw)])

    class FakeCall:
        def __init__(self, fn, args, kwargs):
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def get(self):
            return self.fn(*self.args, **self.kwargs)

    class FakeFunction:
        def __init__(self, fn, opts, parent=None):
            self.fn = fn
            self.opts = dict(opts)
            self._parent = parent
            self.with_options_calls = [] if parent is None else parent.with_options_calls
            self.spawn_calls = [] if parent is None else parent.spawn_calls

        def with_options(self, **kw):
            self.with_options_calls.append(kw)
            return FakeFunction(self.fn, {**self.opts, **kw}, parent=self._parent or self)

        def spawn(self, *args, **kwargs):
            if self._parent is not None and "spawn" in self._parent.__dict__:
                return self._parent.__dict__["spawn"](*args, **kwargs)
            self.spawn_calls.append((self.opts, args, kwargs))
            return FakeCall(self.fn, args, kwargs)

        def local(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    class FakeApp:
        def __init__(self, name):
            self.name = name

        def function(self, **opts):
            def deco(fn):
                return FakeFunction(fn, opts)
            return deco

        def local_entrypoint(self):
            def deco(fn):
                return fn
            return deco

    modal_stub.App = FakeApp
    modal_stub.Image = FakeImage
    with mock.patch.dict(sys.modules, {"modal": modal_stub}):
        sys.modules.pop("modal_app", None)
        return importlib.import_module("modal_app")


class Lab(unittest.TestCase):
    """A source repo with one commit on main and one commit on no branch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src, self.work, self.bin = root / "src", root / "work", root / "bin"
        self.src.mkdir()
        self.bin.mkdir()
        git("init", "--quiet", "-b", "main", cwd=self.src)
        git("config", "user.email", "ci@example.invalid", cwd=self.src)
        git("config", "user.name", "ci", cwd=self.src)
        (self.src / "README.md").write_text(README)
        git("add", "README.md", cwd=self.src)
        git("commit", "--quiet", "-m", "main", cwd=self.src)
        self.trusted = git("rev-parse", "HEAD", cwd=self.src)
        git("checkout", "--quiet", "-b", "tmp", cwd=self.src)
        git("commit", "--quiet", "--allow-empty", "-m", "fork-like", cwd=self.src)
        self.untrusted = git("rev-parse", "HEAD", cwd=self.src)
        git("checkout", "--quiet", "main", cwd=self.src)
        git("branch", "--quiet", "-D", "tmp", cwd=self.src)
        self.uv = script(self.bin / "uv", f'echo "{sys.executable}"\n')
        self.posted = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_modal(self, name="fake-modal", *, exit_code=0, fail_test=False, write_json=True,
                   record_file: Path | None = None, sleep_cmd: str = ""):
        """Stand-in `modal` CLI script handling `modal run modal_app.py ...`."""
        rec_expr = repr(str(record_file)) if record_file else "None"
        body = f"""
import json, os, sys, tarfile, time
args = sys.argv[1:]
opts = {{}}
i = 0
while i < len(args):
    if args[i].startswith("--") and i + 1 < len(args):
        opts[args[i]] = args[i + 1]
        i += 2
    else:
        i += 1
if {sleep_cmd!r}:
    os.system({sleep_cmd!r})
tar_path = opts.get("--tarball-path")
tar_members = []
if tar_path and os.path.exists(tar_path):
    with tarfile.open(tar_path, "r:gz") as tf:
        tar_members = tf.getnames()
if {rec_expr} is not None:
    with open({rec_expr}, "w") as fh:
        json.dump({{
            "argv": args,
            "opts": opts,
            "tar_members": tar_members,
            "has_gh_token": any(k in os.environ for k in ("GH_TOKEN", "GITHUB_TOKEN")),
            "modal_gpu_env": os.environ.get("RSIBOX_MODAL_GPU", ""),
        }}, fh)
out_json = opts.get("--output-json")
if {write_json!r} and out_json:
    outcome = "failed" if {fail_test!r} else "passed"
    payload = {{
        "ok": not {fail_test!r},
        "gpu": opts.get("--gpu", "L4"),
        "gpu_info": {{
            "core": {{"gpu_name": "NVIDIA L4", "driver_version": "570.124", "torch_version": "2.9.1+cu128", "cuda_version": "12.8"}},
            "vllm": {{"gpu_name": "NVIDIA L4", "driver_version": "570.124", "vllm_version": "0.19.0"}},
        }},
        "seconds": 1.2,
        "counts": {{outcome: 1, "skipped": 1}} if not {fail_test!r} else {{outcome: 1}},
        "steps": [
            {{
                "name": "core-kernel",
                "kind": "unittest",
                "group": "core",
                "rc": 1 if {fail_test!r} else 0,
                "ok": not {fail_test!r},
                "counts": {{outcome: 1, "skipped": 1}} if not {fail_test!r} else {{outcome: 1}},
                "seconds": 1.2,
                "log_text": "synthetic modal log\\n",
                "tests": [
                    {{"id": "test_demo.Case.test_one", "outcome": outcome, "seconds": 0.1, "detail": ""}},
                    *([] if {fail_test!r} else [{{"id": "test_demo.Case.test_skip", "outcome": "skipped", "seconds": 0.0, "detail": "optional"}}]),
                ],
            }}
        ],
    }}
    with open(out_json, "w") as fh:
        json.dump(payload, fh)
sys.exit({exit_code})
"""
        return py_script(self.bin / name, body)

    def run_ci(self, sha, *extra):
        argv = ["--sha", sha, "--source", str(self.src), "--work", str(self.work), "--uv", self.uv,
                "--post", *extra]
        with mock.patch.object(runner, "post_status", lambda s, st: self.posted.append(st)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_:
                runner.main(argv)
        return exit_.exception.code

    def result(self, sha):
        return json.loads((self.work / "results" / f"{sha}.json").read_text())


class Refusals(Lab):
    def test_untrusted_sha_posts_nothing_and_is_not_final(self):
        """A SHA on no branch (fork PR head) is refused without posting and stays non-final."""
        self.assertEqual(self.run_ci(self.untrusted, "--stages", "quickstart"), 2)
        rec = self.result(self.untrusted)
        self.assertEqual((rec["final"], rec["attempts"], self.posted), (False, 0, []))
        self.assertIn("on no branch", rec["refused"])
        self.assertFalse(loop.is_final(self.work / "results" / "missing.json"))
        self.assertFalse(loop.is_final(self.work / "results" / f"{self.untrusted}.json"))

    def test_fork_pr_refused_without_posting(self):
        """A PR from a fork repository is refused by check_pr before any status is posted."""
        modal_bin = self.fake_modal()
        fake_pr = json.dumps({"head": {"repo": {"full_name": "other/rsibox"}, "sha": self.trusted, "ref": "patch"}})
        orig_sh = runner.sh

        def fake_sh(argv, **kw):
            if argv[:2] == ["gh", "api"]:
                return subprocess.CompletedProcess(argv, 0, stdout=fake_pr, stderr="")
            return orig_sh(argv, **kw)

        with mock.patch.object(runner, "sh", side_effect=fake_sh):
            code = self.run_ci(self.trusted, "--pr", "42", "--modal-cmd", modal_bin)
        rec = self.result(self.trusted)
        self.assertEqual((code, rec["final"], self.posted), (2, False, []))
        self.assertIn("fork PRs are not run", rec["refused"])

    def test_pr_head_mismatch_refused(self):
        """If the PR head moved away from `--sha`, the run is refused before posting."""
        modal_bin = self.fake_modal()
        fake_pr = json.dumps({"head": {"repo": {"full_name": runner.REPO}, "sha": "0" * 40, "ref": "feat"}})
        orig_sh = runner.sh

        def fake_sh(argv, **kw):
            if argv[:2] == ["gh", "api"]:
                return subprocess.CompletedProcess(argv, 0, stdout=fake_pr, stderr="")
            return orig_sh(argv, **kw)

        with mock.patch.object(runner, "sh", side_effect=fake_sh):
            code = self.run_ci(self.trusted, "--pr", "7", "--modal-cmd", modal_bin)
        rec = self.result(self.trusted)
        self.assertEqual((code, rec["final"], self.posted), (2, False, []))
        self.assertIn("not " + self.trusted, rec["refused"])

    def test_old_result_without_final_key_is_final(self):
        old = self.work / "results" / "old.json"
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps({"run_dir": "old-run", "status": {"state": "success"}}))
        self.assertTrue(loop.is_final(old))


class ModalHandoffAndStatus(Lab):
    def test_trusted_sha_exports_git_archive_and_posts_modal_gpu_status(self):
        """Trusted SHA creates a clean git archive tarball, strips GH tokens, runs modal-cmd,
        persists per-step logs + summary.md/json, and posts `modal-gpu` status."""
        rec_file = self.work / "modal-call.json"
        modal_bin = self.fake_modal(record_file=rec_file)
        with mock.patch.dict(os.environ, {"GH_TOKEN": "dummy-gh-token-for-test", "GITHUB_TOKEN": "dummy-github-tok"}):
            code = self.run_ci(
                self.trusted,
                "--modal-cmd", modal_bin,
                "--gpu", "A10G",
                "--suites", "core-kernel,vllm-services",
                "--vllm-model", "Qwen/Qwen3-0.6B",
            )
        self.assertEqual(code, 0)
        call = json.loads(rec_file.read_text())
        self.assertEqual(call["argv"][:2], ["run", str(HERE / "modal_app.py")])
        self.assertEqual(call["opts"]["--gpu"], "A10G")
        self.assertEqual(call["opts"]["--suites"], "core-kernel,vllm-services")
        self.assertEqual(call["opts"]["--vllm-model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(call["modal_gpu_env"], "A10G")
        self.assertFalse(call["has_gh_token"])
        self.assertIn("README.md", call["tar_members"])
        self.assertFalse(any(m.startswith(".git") for m in call["tar_members"]))

        rec = self.result(self.trusted)
        self.assertTrue(rec["final"])
        self.assertEqual([s["state"] for s in self.posted], ["pending", "success"])
        self.assertEqual(self.posted[-1]["context"], "modal-gpu")
        self.assertIn("gpu: 1 passed, 1 skipped", self.posted[-1]["description"])

        run_dir = Path(rec["run_dir"])
        self.assertFalse((run_dir / "src.tar.gz").exists())
        self.assertTrue((run_dir / "logs" / "gpu" / "core-kernel.log").is_file())
        self.assertIn("NVIDIA L4", (run_dir / "summary.md").read_text())

    def test_modal_test_failure_exits_1_and_posts_failure_final(self):
        modal_bin = self.fake_modal(exit_code=1, fail_test=True)
        code = self.run_ci(self.trusted, "--modal-cmd", modal_bin)
        rec = self.result(self.trusted)
        self.assertEqual((code, rec["final"]), (1, True))
        self.assertEqual([s["state"] for s in self.posted], ["pending", "failure"])
        self.assertEqual(self.posted[-1]["context"], "modal-gpu")

    def test_modal_missing_output_json_is_retried_infra_error(self):
        modal_bin = self.fake_modal(exit_code=2, write_json=False)
        args = ("--modal-cmd", modal_bin, "--max-attempts", "2")
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec1 = self.result(self.trusted)
        self.assertEqual((rec1["attempts"], rec1["final"]), (1, False))
        self.assertEqual([s["state"] for s in self.posted], ["pending", "pending"])
        self.assertIn("will retry (1/2)", self.posted[-1]["description"])

        time.sleep(1.1)
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec2 = self.result(self.trusted)
        self.assertEqual((rec2["attempts"], rec2["final"]), (2, True))
        self.assertEqual(self.posted[-1]["state"], "error")

    def test_resolve_modal_cmd_auto_detects_path_and_uv_fallback(self):
        fake = script(self.bin / "modal", "exit 0\n")
        self.assertEqual(runner.resolve_modal_cmd("custom --flag", self.uv), ["custom", "--flag"])
        with mock.patch.dict(os.environ, {"PATH": str(self.bin)}):
            self.assertEqual(runner.resolve_modal_cmd("", self.uv), [fake])
        with mock.patch.object(runner.shutil, "which", side_effect=lambda name: None if name == "modal" else self.uv):
            self.assertEqual(
                runner.resolve_modal_cmd("", self.uv),
                [self.uv, "run", "--no-project", "--with", "modal", "modal"],
            )


class RunnerErrors(Lab):
    def test_runner_error_is_retried_then_final(self):
        self.uv = str(self.bin / "no-such-uv")
        args = ("--stages", "quickstart", "--quickstart-pythons", "3", "--max-attempts", "2")
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec = self.result(self.trusted)
        self.assertEqual((rec["attempts"], rec["final"]), (1, False))
        self.assertEqual([status["state"] for status in self.posted], ["pending", "pending"])
        self.assertIn("will retry (1/2)", self.posted[-1]["description"])
        time.sleep(1.1)
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec = self.result(self.trusted)
        self.assertEqual((rec["attempts"], rec["final"]), (2, True))
        self.assertEqual(self.posted[-1]["state"], "error")

    def test_runner_error_before_trust_posts_nothing_and_is_never_final(self):
        self.src = self.src.parent / "missing-repo"
        for _ in range(2):
            self.assertEqual(self.run_ci(self.trusted, "--stages", "quickstart", "--max-attempts", "1"), 2)
            time.sleep(1.1)
        rec = self.result(self.trusted)
        self.assertEqual((rec["final"], rec["attempts"], self.posted), (False, 0, []))

    def assert_dead(self, pid_path: Path):
        pid = int(pid_path.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(pid, 9)
        self.fail(f"grandchild {pid} survived the stopped run")

    def test_deadline_on_modal_step_is_a_runner_error_and_kills_the_group(self):
        grandchild = self.work.parent / "grandchild.pid"
        modal_bin = self.fake_modal(
            write_json=False,
            sleep_cmd=f"sleep 300 & echo $! > {grandchild}; wait",
        )
        started = time.monotonic()
        code = self.run_ci(self.trusted, "--modal-cmd", modal_bin, "--deadline-seconds", "3")
        self.assertLess(time.monotonic() - started, 25)
        rec = self.result(self.trusted)
        self.assertEqual((code, rec["attempts"], rec["final"]), (2, 1, False))
        self.assertIn("deadline", rec["status"]["description"])
        self.assertEqual([status["state"] for status in self.posted], ["pending", "pending"])
        self.assert_dead(grandchild)

    def test_sigterm_stops_the_run_and_counts_as_a_runner_error(self):
        grandchild = self.work.parent / "grandchild.pid"
        modal_bin = self.fake_modal(
            write_json=False,
            sleep_cmd=f"sleep 60 & echo $! > {grandchild}; wait",
        )
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "run_modal_ci.py"), "--sha", self.trusted,
             "--source", str(self.src), "--work", str(self.work), "--modal-cmd", modal_bin],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            pid_file = self.work / "active" / f"{self.trusted}.pid"
            deadline = time.time() + 30
            while time.time() < deadline and not (grandchild.exists() and grandchild.read_text().strip()):
                time.sleep(0.05)
            self.assertEqual(pid_file.read_text(), str(proc.pid))
            self.assertTrue(loop.is_runner(proc.pid, self.trusted))
            t0 = time.monotonic()
            proc.terminate()
            self.assertEqual(proc.wait(timeout=30), 2)
            self.assertLess(time.monotonic() - t0, 20)
        finally:
            proc.kill()
        rec = self.result(self.trusted)
        self.assertEqual((rec["attempts"], rec["final"]), (1, False))
        self.assertIn("Stopped", rec["status"]["description"])
        self.assertFalse(pid_file.exists())
        self.assert_dead(grandchild)

    def test_sigterm_during_finally_and_except_still_writes_attempt_record(self):
        for phase in ("finally", "except"):
            with self.subTest(phase=phase):
                if phase == "finally":
                    hook = (
                        "orig = shutil.rmtree\n"
                        "def sig_rmtree(path, ignore_errors=False):\n"
                        "    os.kill(os.getpid(), signal.SIGTERM)\n"
                        "    return orig(path, ignore_errors=ignore_errors)\n"
                        "shutil.rmtree = sig_rmtree\n"
                    )
                    sha = self.trusted
                    uv_arg = str(self.bin / "no-such-uv")
                else:
                    hook = (
                        "class SigRefused(runner.Refused):\n"
                        "    def __str__(self):\n"
                        "        os.kill(os.getpid(), signal.SIGTERM)\n"
                        "        return 'refused-for-test'\n"
                        "def sig_refused(mirror, sha):\n"
                        "    raise SigRefused()\n"
                        "runner.trusted_branches = sig_refused\n"
                    )
                    sha = self.untrusted
                    uv_arg = self.uv
                code = (
                    "import os, shutil, signal, sys\n"
                    f"sys.path.insert(0, {str(HERE)!r})\n"
                    "import run_modal_ci as runner\n"
                    "runner.post_status = lambda sha, st: None\n"
                    f"{hook}"
                    "runner.main([\n"
                    f"  '--sha', {sha!r}, '--source', {str(self.src)!r},\n"
                    f"  '--work', {str(self.work)!r}, '--uv', {uv_arg!r},\n"
                    "  '--stages', 'quickstart', '--quickstart-pythons', '9.99',\n"
                    "])\n"
                )
                r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, r.stderr + r.stdout)
                rec = self.result(sha)
                self.assertFalse(rec["final"])
                self.assertFalse((self.work / "active" / f"{sha}.pid").exists())
                if phase == "finally":
                    self.assertEqual(rec["attempts"], 1)
                else:
                    self.assertIn("Stopped", rec["status"]["description"])

    def test_deadline_and_settle_contract(self):
        clock = runner.Deadline(1)
        time.sleep(1.05)
        with self.assertRaises(runner.Stopped):
            clock.timeout(1200)
        self.assertEqual(runner.Deadline(0).timeout(1200), 1200)

        res_path = self.work / "results" / "settle.json"
        res_path.parent.mkdir(parents=True, exist_ok=True)
        # Pre-trust infra error does not increment attempts and never posts.
        rec, post = runner.settle(
            {"run_dir": "r1", "infra_error": "fetch failed", "pending_posted": False},
            {"state": "error", "context": runner.CONTEXT, "description": "runner error"},
            res_path,
            max_attempts=1,
        )
        self.assertEqual((rec["final"], rec["attempts"], post), (False, 0, None))


class Quickstart(Lab):
    def test_all_sh_and_bash_blocks_of_the_section_and_nothing_after_it(self):
        self.assertEqual(runner.quickstart_script(README),
                         "echo first >> quickstart.out\necho second >> quickstart.out\n")
        bash_only = "## Quickstart\n```bash\necho safe\n```\n## Later\n```sh\necho WRONG\n```\n"
        self.assertEqual(runner.quickstart_script(bash_only), "echo safe\n")
        self.assertIsNone(runner.quickstart_script("## Quickstart\nprose\n## Later\n```sh\nx\n```\n"))
        self.assertIsNone(runner.quickstart_script("# no quickstart here\n"))

    def test_quickstart_runs_both_blocks_in_a_fresh_venv(self):
        version = "%d.%d" % sys.version_info[:2]
        code = self.run_ci(self.trusted, "--stages", "quickstart", "--quickstart-pythons", version,
                           "--keep-envs")
        rec = self.result(self.trusted)
        out = Path(rec["run_dir"], "envs", f"quickstart-src-{version}", "quickstart.out").read_text()
        self.assertEqual((code, rec["final"], out), (0, True, "first\nsecond\n"))
        self.assertEqual([status["state"] for status in self.posted], ["pending", "success"])
        self.assertFalse((self.work / "active" / f"{self.trusted}.pid").exists())

    def test_quickstart_and_cpu_stage_strip_tokens_and_validate_suites(self):
        """[190-6]: local quickstart and cpu stages strip GitHub/Modal tokens from subprocess env,
        reject unknown --suites, and never report ok=True for 0 executed steps."""
        # Quickstart strips tokens from its subprocess environment.
        readme_env = (
            "# demo\n\n## Quickstart\n\n```sh\n"
            "python3 -c 'import os; "
            "leaked=[k for k in (\"GH_TOKEN\",\"GITHUB_TOKEN\",\"GITHUB_API\",\"MODAL_TOKEN_ID\",\"MODAL_TOKEN_SECRET\") if k in os.environ]; "
            "assert not leaked, leaked'\n"
            "```\n"
        )
        (self.src / "README.md").write_text(readme_env)
        git("add", "README.md", cwd=self.src)
        git("commit", "--quiet", "-m", "check env", cwd=self.src)
        sha = git("rev-parse", "HEAD", cwd=self.src)
        version = "%d.%d" % sys.version_info[:2]
        tok_env = {
            "GH_TOKEN": "dummy-gh",
            "GITHUB_TOKEN": "dummy-github",
            "GITHUB_API": "https://api.example.invalid",
            "MODAL_TOKEN_ID": "ak-dummy",
            "MODAL_TOKEN_SECRET": "as-dummy",
        }
        with mock.patch.dict(os.environ, tok_env):
            code = self.run_ci(sha, "--stages", "quickstart", "--quickstart-pythons", version)
        self.assertEqual(code, 0)

        # run_stage strips tokens, rejects unknown suites, and requires >= 1 step for ok=True.
        run_dir = self.work / "cpu-stage-check"
        run_dir.mkdir(parents=True)
        seen_envs = []

        def fake_run_group(argv, per_step, deadline, **kw):
            seen_envs.append(dict(kw.get("env") or {}))
            return 0

        with mock.patch.dict(os.environ, tok_env), \
                mock.patch.object(runner, "run_group", side_effect=fake_run_group), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                runner.run_stage("cpu", self.src, sys.executable, {}, run_dir, 60,
                                 runner.Deadline(0), suites_filter={"typo-suite"})
            empty_out = runner.run_stage("cpu", self.src, sys.executable, {}, run_dir, 60,
                                         runner.Deadline(0), suites_filter=set())
            self.assertFalse(empty_out["ok"])
            ok_out = runner.run_stage("cpu", self.src, sys.executable, {}, run_dir, 60,
                                      runner.Deadline(0), suites_filter={"core-doctor"})
        self.assertTrue(ok_out["ok"])
        self.assertEqual(len(seen_envs), 1)
        for key in tok_env:
            self.assertNotIn(key, seen_envs[0])


class Loop(Lab):
    def args(self, *extra):
        return [
            "--work", str(self.work),
            "--source", str(self.src),
            "--gpu", "A100",
            "--stages", "gpu",
            "--timeout", "77",
            "--max-attempts", "4",
            "--vllm-model", "Qwen/Qwen3-0.6B",
            "--suites", "core-kernel",
            "--max-wait", "600",
            "--poll", "0.05",
            *extra,
        ]

    def test_loop_forwards_runner_flags(self):
        modal_bin = self.fake_modal()
        argv = self.args("--modal-cmd", modal_bin)
        parsed = loop.build_parser().parse_args(argv)
        got = loop.runner_argv(parsed, "a" * 40, 7, self.work)
        joined = " ".join(got)
        for flag in (
            "--gpu A100",
            "--stages gpu",
            "--timeout 77",
            "--max-attempts 4",
            "--vllm-model Qwen/Qwen3-0.6B",
            "--suites core-kernel",
            "--source " + str(self.src),
            "--modal-cmd " + modal_bin,
            "--deadline-seconds 540",
            "--pr 7",
        ):
            self.assertIn(flag, joined)

    def test_loop_reads_modal_env_defaults(self):
        with mock.patch.dict(os.environ, {
            "MODAL_GPU": "H100",
            "MODAL_VLLM_MODEL": "Qwen/Qwen3-0.6B",
        }):
            parsed = loop.build_parser().parse_args([])
        argv = loop.runner_argv(parsed, "e" * 40, None, self.work)
        self.assertEqual(argv[argv.index("--gpu") + 1], "H100")
        self.assertEqual(argv[argv.index("--vllm-model") + 1], "Qwen/Qwen3-0.6B")

    def test_loop_retries_a_refused_sha_and_skips_a_final_one(self):
        launches = self.work.parent / "launches.txt"
        rec_file = self.work / "fake-modal-out.json"
        modal_bin = self.fake_modal(record_file=rec_file)
        modal_wrapper = py_script(
            self.bin / "modal-wrap",
            f"import os, sys\nopen({str(launches)!r}, 'a').write('called\\n')\nos.execv({modal_bin!r}, [{modal_bin!r}, *sys.argv[1:]])\n",
        )
        argv = ["--work", str(self.work), "--source", str(self.src), "--stages", "gpu",
                "--modal-cmd", modal_wrapper, "--poll", "0.05", "--once"]
        heads = [(self.untrusted, None, "fork-like"), (self.trusted, None, "main")]
        with mock.patch.object(loop, "candidates", lambda: heads), \
                contextlib.redirect_stdout(io.StringIO()):
            loop.main(argv)
            self.assertTrue(loop.is_final(self.work / "results" / f"{self.trusted}.json"))
            self.assertFalse(loop.is_final(self.work / "results" / f"{self.untrusted}.json"))
            loop.main(argv)
        # Untrusted SHA was refused before calling Modal; trusted SHA ran once (skipped on pass 2).
        self.assertEqual(launches.read_text().splitlines(), ["called"])
        self.assertEqual(self.result(self.trusted)["status"]["state"], "success")

    def test_loop_signals_only_the_runner_named_in_the_pid_file(self):
        sha = "c" * 40
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        other = subprocess.Popen(sleeper)
        runner_like = subprocess.Popen(sleeper + ["run_modal_ci.py", "--sha", sha])
        try:
            pid_file = self.work / "active" / f"{sha}.pid"
            pid_file.parent.mkdir(parents=True)
            since = time.time() - 1
            pid_file.write_text(str(other.pid))
            self.assertFalse(loop.stop_run(self.work, sha, since))
            self.assertIsNone(other.poll())
            pid_file.write_text(str(runner_like.pid))
            os.utime(pid_file, (since - 100, since - 100))
            self.assertFalse(loop.stop_run(self.work, sha, since))
            pid_file.touch()
            self.assertTrue(loop.stop_run(self.work, sha, since))
            self.assertEqual(runner_like.wait(timeout=10), -15)
        finally:
            for proc in (other, runner_like):
                proc.kill()
                proc.wait()

    def test_is_runner_uses_ps_ww_and_matches_long_command_lines(self):
        sha = "d" * 40
        long_url = "https://logs.example.invalid/" + ("segment/" * 30)
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        long_runner = subprocess.Popen(
            sleeper + ["/opt/runner/modal-ci/run_modal_ci.py", "--logs-url-base", long_url, "--sha", sha]
        )
        not_runner = subprocess.Popen(
            sleeper + ["/tmp/not_run_modal_ci.py", "--other", sha]
        )
        try:
            self.assertGreater(len(" ".join([*sleeper, "/opt/runner/modal-ci/run_modal_ci.py",
                                             "--logs-url-base", long_url, "--sha", sha])), 250)
            self.assertTrue(loop.is_runner(long_runner.pid, sha))
            self.assertFalse(loop.is_runner(not_runner.pid, sha))

            seen_argv = []
            def fake_run(argv, **kw):
                seen_argv.append(argv)
                return subprocess.CompletedProcess(
                    argv, 0,
                    stdout=f"python3 /very/long/prefix/{'x' * 220}/run_modal_ci.py --sha {sha}\n",
                )
            with mock.patch.object(loop.subprocess, "run", side_effect=fake_run):
                self.assertTrue(loop.is_runner(12345, sha))
            self.assertIn("-ww", seen_argv[0])
        finally:
            for proc in (long_runner, not_runner):
                proc.kill()
                proc.wait()


class ModalAppStructure(unittest.TestCase):
    def test_modal_app_uses_v1_api_and_cached_images(self):
        src_text = (HERE / "modal_app.py").read_text()
        self.assertNotIn("modal.stub", src_text)
        self.assertNotIn("@stub", src_text)
        tree = ast.parse(src_text)
        fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"run_core_gpu", "run_vllm_gpu", "main", "select_suites"} <= fn_names)

        app_mod = load_modal_app_with_stub()
        self.assertEqual(app_mod.app.name, "rsibox-gpu-ci")
        core_ops = app_mod.core_gpu_image.ops
        vllm_ops = app_mod.vllm_gpu_image.ops
        self.assertEqual(core_ops[0], ("debian_slim", "3.12"))
        self.assertEqual(vllm_ops[0], ("debian_slim", "3.12"))
        self.assertIn("torch==2.9.1", core_ops[2][1])
        self.assertIn("ray", core_ops[3][1])
        self.assertIn("vllm", vllm_ops[2][1])

    def test_select_suites_and_isolated_pythonpaths(self):
        app_mod = load_modal_app_with_stub()
        self.assertEqual(app_mod.select_suites(""), (None, True))
        self.assertEqual(app_mod.select_suites("vllm-services"), ([], True))
        self.assertEqual(app_mod.select_suites("core-kernel,vllm-services"), (["core-kernel"], True))
        with self.assertRaises(ValueError):
            app_mod.select_suites("core-kernel,does-not-exist")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "repos/megatron/tools").mkdir(parents=True)
            steps = app_mod.build_core_steps(root, sys.executable, root / "res", HERE / "unittest_json.py")
            names = [s["name"] for s in steps]
            self.assertIn("core-kernel", names)
            self.assertIn("core-rsibox", names)
            self.assertIn("megatron-tools", names)
            self.assertIn("sglang-services", names)
            self.assertNotIn("vllm-services", names)
            self.assertIn("core-rsibox", app_mod.CORE_SUITE_ORDER)
            self.assertIn("core-rsibox", app_mod.ALL_SUITE_ORDER)
            cpu_names = [s["name"] for s in runner.cpu_suites(root, sys.executable, root / "res")]
            self.assertIn("core-rsibox", cpu_names)
            for s in steps:
                pypath = s.get("env", {}).get("PYTHONPATH", "")
                self.assertFalse("sglang" in pypath and "vllm" in pypath, s)

    def test_local_entrypoint_spawns_selected_functions_in_parallel(self):
        app_mod = load_modal_app_with_stub()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar = root / "src.tar.gz"
            with tarfile.open(tar, "w:gz") as tf:
                info = tarfile.TarInfo("README.md")
                data = b"# demo\n"
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            out_json = root / "out.json"

            fake_core = mock.MagicMock()
            fake_core.get.return_value = {
                "gpu_info": {"gpu_name": "NVIDIA L4"},
                "steps": [{"name": "core-kernel", "kind": "unittest", "group": "core",
                           "rc": 0, "ok": True, "counts": {"passed": 2}, "seconds": 0.5,
                           "tests": [], "log_text": "ok\n"}],
            }
            fake_vllm = mock.MagicMock()
            fake_vllm.get.return_value = {
                "gpu_info": {"gpu_name": "NVIDIA L4"},
                "steps": [{"name": "vllm-services", "kind": "pytest", "group": "vllm",
                           "rc": 0, "ok": True, "counts": {"passed": 3}, "seconds": 0.7,
                           "tests": [], "log_text": "ok\n"}],
            }

            with mock.patch.object(app_mod.run_core_gpu, "spawn", return_value=fake_core) as sp_core, \
                    mock.patch.object(app_mod.run_vllm_gpu, "spawn", return_value=fake_vllm) as sp_vllm:
                with self.assertRaises(SystemExit) as ex:
                    app_mod.main(str(tar), str(out_json), gpu="A10G", timeout=900)
                self.assertEqual(ex.exception.code, 0)
                self.assertEqual(sp_core.call_count, 1)
                self.assertEqual(sp_vllm.call_count, 1)
                self.assertEqual(app_mod.run_core_gpu.with_options_calls, [{"gpu": "A10G", "timeout": 2700}])
                self.assertEqual(app_mod.run_vllm_gpu.with_options_calls, [{"gpu": "A10G", "timeout": 2700}])

            payload = json.loads(out_json.read_text())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["counts"], {"passed": 5})
            self.assertEqual([s["name"] for s in payload["steps"]], ["core-kernel", "vllm-services"])

            # Filter to vllm-services only -> run_core_gpu is never spawned.
            with mock.patch.object(app_mod.run_core_gpu, "spawn", return_value=fake_core) as sp_core, \
                    mock.patch.object(app_mod.run_vllm_gpu, "spawn", return_value=fake_vllm) as sp_vllm:
                with self.assertRaises(SystemExit) as ex:
                    app_mod.main(str(tar), str(out_json), suites="vllm-services")
                self.assertEqual(ex.exception.code, 0)
                self.assertEqual(sp_core.call_count, 0)
                self.assertEqual(sp_vllm.call_count, 1)

    def test_local_entrypoint_cancels_sibling_call_on_error(self):
        """[190-8]: if core_call.get() fails (or vllm spawn fails), the sibling Modal call is
        cancelled instead of left running."""
        app_mod = load_modal_app_with_stub()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar = root / "src.tar.gz"
            with tarfile.open(tar, "w:gz") as tf:
                info = tarfile.TarInfo("README.md")
                data = b"# demo\n"
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            out_json = root / "out.json"

            failing_core = mock.MagicMock()
            failing_core.get.side_effect = RuntimeError("core container crashed")
            sibling_vllm = mock.MagicMock()
            with mock.patch.object(app_mod.run_core_gpu, "spawn", return_value=failing_core), \
                    mock.patch.object(app_mod.run_vllm_gpu, "spawn", return_value=sibling_vllm):
                with self.assertRaises(RuntimeError):
                    app_mod.main(str(tar), str(out_json))
            sibling_vllm.cancel.assert_called_once()

            spawned_core = mock.MagicMock()
            with mock.patch.object(app_mod.run_core_gpu, "spawn", return_value=spawned_core), \
                    mock.patch.object(app_mod.run_vllm_gpu, "spawn", side_effect=RuntimeError("vllm spawn failed")):
                with self.assertRaises(RuntimeError):
                    app_mod.main(str(tar), str(out_json))
            spawned_core.cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
