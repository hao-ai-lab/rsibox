#!/usr/bin/env python3
"""Golden cases for the B200 CI runner and loop, on CPU with no GPU and no network.

The runner runs for real against a local git repository (real fetch, trust check, clone and
quickstart venv). Only the edges are replaced: `nvidia-smi` by a script that prints a fixed
GPU table, `rack-run` by a script that runs its command, and the GitHub status call by a
recorder. Each case names the pain point it guards.

Run: python3 .github/b200-ci/test_b200_ci.py -v
"""
import contextlib, io, json, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b200_ci_loop as loop   # noqa: E402
import run_b200_ci as runner  # noqa: E402

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
        # `uv python find` answers with this interpreter, so quickstart needs no uv install.
        self.uv = script(self.bin / "uv", f'echo "{sys.executable}"\n')
        self.posted = []

    def tearDown(self):
        self.tmp.cleanup()

    def smi(self, *rows):
        """A stand-in nvidia-smi printing `index, memory.used, utilization.gpu` rows."""
        return script(self.bin / "nvidia-smi", "cat <<'EOF'\n" + "\n".join(rows) + "\nEOF\n")

    def run_ci(self, sha, *extra):
        argv = ["--sha", sha, "--source", str(self.src), "--work", str(self.work), "--uv", self.uv,
                "--post", *extra]
        with mock.patch.object(runner, "post_status", lambda sha, status: self.posted.append(status)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_:
                runner.main(argv)
        return exit_.exception.code

    def result(self, sha):
        return json.loads((self.work / "results" / f"{sha}.json").read_text())


class Refusals(Lab):
    def test_untrusted_sha_posts_nothing_and_is_not_final(self):
        """Pain point: a SHA on no branch (a fork PR head) got an `error` status posted on it
        and a result that made the loop skip it for good."""
        self.assertEqual(self.run_ci(self.untrusted, "--stages", "quickstart"), 2)
        rec = self.result(self.untrusted)
        self.assertEqual((rec["final"], rec["attempts"], self.posted), (False, 0, []))
        self.assertIn("on no branch", rec["refused"])
        self.assertFalse(loop.is_final(self.work / "results" / "missing.json"))
        self.assertFalse(loop.is_final(self.work / "results" / f"{self.untrusted}.json"))

    def test_busy_gpus_post_nothing_and_are_retried(self):
        """Pain point: two busy GPUs (someone else's job) blacklisted a trusted commit for
        good and posted `error` on it. Now: refused before the pending status, nothing posted,
        a non-final result the loop retries."""
        smi = self.smi("0, 80000, 97", "1, 3, 0", "2, 70000, 100", "3, 60000, 50")
        code = self.run_ci(self.trusted, "--stages", "gpu", "--gpus", "auto:2", "--nvidia-smi", smi)
        rec = self.result(self.trusted)
        self.assertEqual((code, rec["final"], self.posted), (2, False, []))
        self.assertIn("found ['1']", rec["refused"])

    def test_old_result_without_final_key_is_final(self):
        """Results written before this change have no "final" key; they stay skipped."""
        old = self.work / "results" / "old.json"
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps({"run_dir": "old-run", "status": {"state": "success"}}))
        self.assertTrue(loop.is_final(old))


class RunnerErrors(Lab):
    def test_runner_error_is_retried_then_final(self):
        """Pain point: one transient runner error (here: the quickstart's uv is missing)
        became the commit's permanent result. Now attempt 1 of 2 keeps the status pending and
        the result non-final; attempt 2 is final and posts the error."""
        self.uv = str(self.bin / "no-such-uv")
        args = ("--stages", "quickstart", "--quickstart-pythons", "3", "--max-attempts", "2")
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec = self.result(self.trusted)
        self.assertEqual((rec["attempts"], rec["final"]), (1, False))
        self.assertEqual([status["state"] for status in self.posted], ["pending", "pending"])
        self.assertIn("will retry (1/2)", self.posted[-1]["description"])
        time.sleep(1.1)                      # run ids are per second
        self.assertEqual(self.run_ci(self.trusted, *args), 2)
        rec = self.result(self.trusted)
        self.assertEqual((rec["attempts"], rec["final"]), (2, True))
        self.assertEqual(self.posted[-1]["state"], "error")

    def test_runner_error_before_trust_posts_nothing_and_is_never_final(self):
        """A failed fetch means trust is unknown: nothing may be posted on that SHA. Pain
        point: three such errors in a row (a short network outage) retired the head of main
        for good. They are now not counted and never final."""
        self.src = self.src.parent / "missing-repo"
        for _ in range(2):
            self.assertEqual(self.run_ci(self.trusted, "--stages", "quickstart", "--max-attempts", "1"), 2)
            time.sleep(1.1)                  # run ids are per second
        rec = self.result(self.trusted)
        self.assertEqual((rec["final"], rec["attempts"], self.posted), (False, 0, []))

    def slow_quickstart(self, block):
        (self.src / "README.md").write_text(f"## Quickstart\n```sh\n{block}\n```\n")
        git("commit", "--quiet", "-am", "slow quickstart", cwd=self.src)
        return git("rev-parse", "HEAD", cwd=self.src)

    def assert_dead(self, pid_path: Path):
        """The process whose pid the quickstart wrote to `pid_path` is gone."""
        pid = int(pid_path.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(pid, 9)
        self.fail(f"grandchild {pid} survived the stopped run")

    def test_deadline_on_the_last_step_is_a_runner_error_and_kills_the_group(self):
        """Pain point: a deadline that cut the last step was recorded as a final `failure`
        (posted), and the step's own children (here `sleep 300`) kept running after the
        runner exited. Now: a non-final runner error, status still pending, group killed."""
        grandchild = self.work.parent / "grandchild.pid"
        sha = self.slow_quickstart(f"sleep 300 & echo $! > {grandchild}; wait")
        version = "%d.%d" % sys.version_info[:2]
        started = time.monotonic()
        code = self.run_ci(sha, "--stages", "quickstart", "--quickstart-pythons", version,
                           "--deadline-seconds", "8")
        self.assertLess(time.monotonic() - started, 30)
        rec = self.result(sha)
        self.assertEqual((code, rec["attempts"], rec["final"]), (2, 1, False))
        self.assertIn("deadline", rec["status"]["description"])
        self.assertEqual([status["state"] for status in self.posted], ["pending", "pending"])
        self.assert_dead(grandchild)

    def test_sigterm_stops_the_run_and_counts_as_a_runner_error(self):
        """Pain point: the loop gave up waiting on a run, which kept its GPUs and went on next
        to the following one. SIGTERM (what the loop sends) now ends the running step and
        leaves a non-final runner-error result."""
        grandchild = self.work.parent / "grandchild.pid"
        sha = self.slow_quickstart(f"sleep 60 & echo $! > {grandchild}; wait")
        version = "%d.%d" % sys.version_info[:2]
        proc = subprocess.Popen([sys.executable, str(HERE / "run_b200_ci.py"), "--sha", sha,
                                 "--source", str(self.src), "--work", str(self.work), "--uv", self.uv,
                                 "--stages", "quickstart", "--quickstart-pythons", version],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            pid_file = self.work / "active" / f"{sha}.pid"
            deadline = time.time() + 60
            while time.time() < deadline and not (grandchild.exists() and grandchild.read_text().strip()):
                time.sleep(0.1)                      # until the step is inside `sleep 60`
            self.assertEqual(pid_file.read_text(), str(proc.pid))
            self.assertTrue(loop.is_runner(proc.pid, sha))
            t0 = time.monotonic()
            proc.terminate()
            self.assertEqual(proc.wait(timeout=30), 2)
            self.assertLess(time.monotonic() - t0, 20)
        finally:
            proc.kill()
        rec = self.result(sha)
        self.assertEqual((rec["attempts"], rec["final"]), (1, False))
        self.assertIn("Stopped", rec["status"]["description"])
        self.assertFalse(pid_file.exists())
        self.assert_dead(grandchild)

    def test_deadline(self):
        clock = runner.Deadline(1)
        time.sleep(1.05)
        with self.assertRaises(runner.Stopped):
            clock.timeout(1200)
        self.assertEqual(runner.Deadline(0).timeout(1200), 1200)


class Quickstart(Lab):
    def test_all_sh_and_bash_blocks_of_the_section_and_nothing_after_it(self):
        """Pain point: only the first ```sh block ran (a second block was silently skipped),
        and a ```bash quickstart made the regex run a ```sh block from a later section."""
        self.assertEqual(runner.quickstart_script(README),
                         "echo first >> quickstart.out\necho second >> quickstart.out\n")
        bash_only = "## Quickstart\n```bash\necho safe\n```\n## Later\n```sh\necho WRONG\n```\n"
        self.assertEqual(runner.quickstart_script(bash_only), "echo safe\n")
        self.assertIsNone(runner.quickstart_script("## Quickstart\nprose\n## Later\n```sh\nx\n```\n"))
        self.assertIsNone(runner.quickstart_script("# no quickstart here\n"))

    def test_quickstart_runs_both_blocks_in_a_fresh_venv(self):
        """End to end on CPU: trusted SHA, real clone, real `python -m venv`, both blocks run,
        a final success result and a success status."""
        version = "%d.%d" % sys.version_info[:2]
        code = self.run_ci(self.trusted, "--stages", "quickstart", "--quickstart-pythons", version,
                           "--keep-envs")
        rec = self.result(self.trusted)
        out = Path(rec["run_dir"], "envs", f"quickstart-src-{version}", "quickstart.out").read_text()
        self.assertEqual((code, rec["final"], out), (0, True, "first\nsecond\n"))
        self.assertEqual([status["state"] for status in self.posted], ["pending", "success"])
        self.assertFalse((self.work / "active" / f"{self.trusted}.pid").exists())


class Gpus(Lab):
    def test_two_runs_never_get_the_same_gpu(self):
        """Pain point: a GPU checked idle minutes before use, with no lock, could be picked by
        two runs at once. Locks are held until the handles close."""
        smi = self.smi("0, 1, 0", "1, 1, 0", "2, 1, 0", "3, 1, 0")
        locks = self.work / "locks"
        first, held = runner.pick_gpus("auto:2", "0,1,2,3", locks, smi)
        second, held2 = runner.pick_gpus("auto:2", "0,1,2,3", locks, smi)
        self.assertEqual((first, second), ("0,1", "2,3"))
        with self.assertRaises(runner.Refused):
            runner.pick_gpus("auto:1", "0,1,2,3", locks, smi)
        with self.assertRaises(runner.Refused):
            runner.pick_gpus("1", "0,1,2,3", locks, smi)       # explicit, but locked
        for handle in held:
            handle.close()
        again, held3 = runner.pick_gpus("auto:2", "0,1,2,3", locks, smi)
        self.assertEqual(again, "0,1")
        for handle in held2 + held3:
            handle.close()

    def test_pool_and_busy_checks(self):
        smi = self.smi("0, 1, 0", "1, 9000, 0", "2, 1, 0", "4, 1, 0")
        locks = self.work / "locks"
        with self.assertRaises(runner.Refused):
            runner.pick_gpus("4", "0,1,2,3", locks, smi)       # outside the pool
        value, held = runner.pick_gpus("auto:2", "0,1,2,3", locks, smi)
        self.assertEqual(value, "0,2")                         # 1 is busy, 4 outside the pool
        self.smi("0, 1, 0", "1, 9000, 0", "2, 30000, 88", "4, 1, 0")
        with self.assertRaises(runner.Refused):                # 2 got busy before the gpu stage
            runner.check_still_idle(value, "auto:2", smi)
        for handle in held:
            handle.close()


class Loop(Lab):
    def args(self, *extra):
        # Like rack-run: start the command in the background and return at once.
        rack = script(self.bin / "rack-run",
                      'while [ "$1" != "--" ]; do shift; done; shift\n"$@" >/dev/null 2>&1 &\n')
        return ["--work", str(self.work), "--source", str(self.src), "--rack-run", rack, "--python", "3.13",
                "--quickstart-pythons", "3.11,3.13", "--timeout", "77", "--stages", "quickstart",
                "--max-wait", "600", "--poll", "0.05", *extra]

    def test_loop_forwards_runner_flags(self):
        """Pain point: --python, --quickstart-pythons and --timeout given to the loop never
        reached the runner, which silently used its defaults."""
        argv = self.args()
        parsed = loop.build_parser().parse_args(argv)
        got = loop.runner_argv(parsed, "a" * 40, 7, self.work)
        joined = " ".join(got)
        for flag in ("--python 3.13", "--quickstart-pythons 3.11,3.13", "--timeout 77",
                     "--stages quickstart", "--max-attempts 3", "--deadline-seconds 540", "--pr 7"):
            self.assertIn(flag, joined)

    def test_loop_retries_a_refused_sha_and_skips_a_final_one(self):
        """Two passes of the real loop, through a stand-in rack-run that starts the real
        runner. Pass 1 runs both heads: the untrusted one is refused (non-final), the trusted
        one passes its quickstart (final). Pass 2 launches only the untrusted one again."""
        version = "%d.%d" % sys.version_info[:2]
        argv = self.args("--once", "--quickstart-pythons", version)
        launches = self.work.parent / "launches.txt"
        script(self.bin / "rack-run", f'echo "$4" >> {launches}\n'
               'while [ "$1" != "--" ]; do shift; done; shift\n"$@" >/dev/null 2>&1 &\n')
        heads = [(self.untrusted, None, "fork-like"), (self.trusted, None, "main")]
        with mock.patch.object(loop, "candidates", lambda: heads), \
                mock.patch.dict(os.environ, {"PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}"}), \
                contextlib.redirect_stdout(io.StringIO()):
            loop.main(argv)
            self.assertTrue(loop.is_final(self.work / "results" / f"{self.trusted}.json"))
            self.assertFalse(loop.is_final(self.work / "results" / f"{self.untrusted}.json"))
            loop.main(argv)
        self.assertEqual(launches.read_text().splitlines(),
                         [f"b200-ci {sha[:12]}" for sha in (self.untrusted, self.trusted, self.untrusted)])
        self.assertEqual(self.result(self.trusted)["status"]["state"], "success")


    def test_loop_signals_only_the_runner_named_in_the_pid_file(self):
        """Pain point: a SIGKILLed runner leaves a fresh pid file, and the loop could SIGTERM
        whatever process reused that pid. Only a live process whose command line is the
        runner for that SHA is signalled; a stale pid file is ignored."""
        sha = "c" * 40
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        other = subprocess.Popen(sleeper)                                  # pid reused by someone else
        runner_like = subprocess.Popen(sleeper + ["run_b200_ci.py", "--sha", sha])
        try:
            pid_file = self.work / "active" / f"{sha}.pid"
            pid_file.parent.mkdir(parents=True)
            since = time.time() - 1
            pid_file.write_text(str(other.pid))
            self.assertFalse(loop.stop_run(self.work, sha, since))
            self.assertIsNone(other.poll())
            pid_file.write_text(str(runner_like.pid))
            os.utime(pid_file, (since - 100, since - 100))
            self.assertFalse(loop.stop_run(self.work, sha, since))         # older than the launch
            pid_file.touch()
            self.assertTrue(loop.stop_run(self.work, sha, since))
            self.assertEqual(runner_like.wait(timeout=10), -15)
        finally:
            for proc in (other, runner_like):
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    unittest.main()
