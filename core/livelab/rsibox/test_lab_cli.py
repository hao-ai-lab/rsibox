"""Golden tests for `core/livelab/rsibox/lab.py` and `cli.py`:
- `RoleEndpoint` multi-role filtering and abstain bookkeeping,
- `RecaptureCoordinator` post-commit rebuild + bounded in-gap rollback on failure,
- `BaseLab` subclass (`MiniToyLab`) passing all 11/11 `core/livelab/conformance` cases (c0-c10),
- Unified `rsibox` CLI (`list`, `doctor`, `conformance`, `ctl`).

Run from the repo root:
    python3 -m unittest discover -s core/livelab/rsibox -p 'test_lab_cli.py' -t core/livelab -v
"""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from conformance import cases as conformance_cases
from rsibox import lab_cli as cli, harness, ipc, lab
from kernel import coordinator as C
from kernel import toy

LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEQ = itertools.count(1)


class RoleEndpointAndRecaptureTests(unittest.TestCase):
    def test_role_endpoint_filters_subplan_and_abstains_cleanly(self):
        """Input: two `RoleEndpoint`s (`worker` and `core`); plan targets only `worker`.
        Expected: `core` abstains through prepare/commit/status/proof/witness_check/undo while
        `worker` swaps and witnesses the code, and dispatch never raises on unknown methods."""
        pkg = f"re_{next(_SEQ)}"
        toy.load_copy(pkg)
        self.addCleanup(
            lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")]
        )
        worker_ep = lab.RoleEndpoint("worker", rank=0, alias={"toy": pkg})
        core_ep = lab.RoleEndpoint("core", alias={"toy": pkg})
        bad = worker_ep.dispatch("nonexistent_method")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["role"], "worker")
        self.assertEqual(bad["rank"], 0)

        ops_src = toy.source("ops")
        patched = harness.edit_once(ops_src, "    return x + 1", "    return x + 5")
        plan = {"id": "txn-re1", "modules": {"toy.ops": {"source": patched, "on": ["worker"]}}}
        co = C.Coordinator(
            [
                C.CallParticipant("worker", worker_ep.dispatch),
                C.CallParticipant("core", core_ep.dispatch),
            ],
            phase2_deadline_s=None,
        )
        res = co.run(plan)
        self.assertEqual(res["outcome"], "committed", res)
        self.assertTrue(res["receipt"]["participants"]["core"]["prepare"]["abstain"])
        self.assertEqual(sys.modules[pkg + ".ops"].activate(10), 15)
        proof = co.prove("txn-re1")
        self.assertTrue(proof["ok"], proof["receipt"]["findings"])
        self.assertEqual(proof["receipt"].get("abstained"), ["core"])
        self.assertEqual(co.undo("txn-re1")["status"], "undone")
        self.assertEqual(sys.modules[pkg + ".ops"].activate(10), 11)

    def test_recapture_coordinator_commits_on_ok_and_rolls_back_in_gap_on_failure(self):
        """Input: `RecaptureCoordinator` over a `LocalParticipant` with a `rebuild` callback.
        Action: first commit with failing rebuild -> must roll back code in the same gap and
        report `rebuild_failed`; second commit with succeeding rebuild + `post_commit_reset` ->
        commits, runs rebuild and reset, and records `rebuild` details in the receipt."""
        pkg = f"rc_{next(_SEQ)}"
        toy.load_copy(pkg)
        self.addCleanup(
            lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")]
        )
        resets_seen = []
        lp = C.LocalParticipant("rank0", alias={"toy": pkg}, resets={"state": lambda: resets_seen.append("state")})
        ops_src = toy.source("ops")
        patched = harness.edit_once(ops_src, "    return x + 1", "    return x + 7")
        plan = {
            "modules": {"toy.ops": {"source": patched}},
            "rebuild": ["recapture"],
            "recapture": {"sizes": [1, 2]},
        }

        fail_co = lab.RecaptureCoordinator(
            [lp],
            gap=lambda fn: fn(),
            phase2_deadline_s=5.0,
            rebuild=lambda txn, spec: {
                "ok": False,
                "error": "OOM during capture",
                "ranks": [{"rank": 0, "ok": False, "error": "OOM"}],
            },
        )
        failed = fail_co.run(plan)
        self.assertEqual(failed["outcome"], "rolled_back", failed)
        self.assertEqual(sys.modules[pkg + ".ops"].activate(10), 11)
        checks = [f["check"] for f in failed["receipt"]["findings"]]
        self.assertIn("rebuild_failed", checks)

        ok_co = lab.RecaptureCoordinator(
            [lp],
            gap=lambda fn: fn(),
            phase2_deadline_s=5.0,
            post_commit_reset="state",
            rebuild=lambda txn, spec: {
                "ok": True,
                "ranks": [{"rank": 0, "ok": True, "captured": spec["sizes"]}],
            },
        )
        ok_res = ok_co.run(plan)
        self.assertEqual(ok_res["outcome"], "committed", ok_res)
        self.assertEqual(sys.modules[pkg + ".ops"].activate(10), 17)
        self.assertEqual(resets_seen, ["state"])
        self.assertEqual(ok_res["receipt"]["rebuild"]["routes"], ["recapture"])
        self.assertEqual(ok_co.undo(ok_res["txn"])["status"], "undone")
        self.assertEqual(sys.modules[pkg + ".ops"].activate(10), 11)


class MiniToyLab(lab.BaseLab):
    """Minimal `BaseLab` subclass over `kernel.toy` (2 ranks) proving `BaseLab` +
    `inject_participant_fault` + `edit_once` pass all 11/11 conformance cases (c0-c10)."""

    name = "mini-toy"
    MODULES = ("toy.ops", "toy.engine")
    WORKLOAD = {"inputs": [0, 1, 2, 3, 5, 8]}

    def __init__(self, ranks=2, receipts=None, check_timeout=0.5, hang_s=1.0):
        self.ranks = [f"rank{i}" for i in range(ranks)]
        self.hang_s = hang_s
        super().__init__(receipts=receipts, check_timeout=check_timeout, auto_spawn=False)
        self.base_text = {m: toy.source(m.split(".", 1)[1]) for m in self.MODULES}
        self._spawn()

    def _start_engine(self):
        tag = f"mt{next(_SEQ)}"
        self.faults = {}
        self.engines, self.locals, self.pkgs, parts = {}, {}, {}, []
        for r in self.ranks:
            pkg = self.pkgs[r] = f"{tag}_{r}"
            toy.load_copy(pkg)
            eng = self.engines[r] = sys.modules[pkg + ".engine"].Engine()
            lp = self.locals[r] = C.LocalParticipant(
                r, alias={"toy": pkg}, scope={"engine": eng},
                resets={"R2": eng.drain, "R3": eng.cache.clear},
            )
            parts.append(C.CallParticipant(r, self._rpc(r, lp)))
        toy.load_copy(f"{tag}_core")
        self.cluster = sys.modules[f"{tag}_core.engine"].Cluster(self.engines)
        co = C.StepGapCoordinator(
            parts, self.cluster.at_gap, writer=self.writer, interface_version="1-draft",
            engine={"name": self.name, "version": "0", "commit": None},
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.cluster.run, args=(self._stop,), daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while len(self.cluster.log) < 2 and time.time() < deadline:
            time.sleep(0.001)
        return co, self.MODULES

    def _rpc(self, rank, lp):
        def call(method, payload):
            if self.faults.get("lost") == rank:
                raise ConnectionError(f"{rank}: process lost")
            return getattr(lp, method)(payload)
        return call

    def _stop_engine(self):
        if hasattr(self, "_stop"):
            self._stop.set()
            self._thread.join(self.hang_s + 5.0)

    def _run_workload(self, workload):
        inputs = (workload or self.WORKLOAD)["inputs"]
        faults = self.faults

        def job():
            if faults.pop("crash", None):
                raise RuntimeError(f"{self.ranks[-1]}: worker died in step")
            if faults.pop("hang", None):
                time.sleep(self.hang_s)
            return {r: [e.step(x) for x in inputs] for r, e in self.engines.items()}

        return json.loads(json.dumps(self.cluster.at_gap(job, self.check_timeout)))

    def _reset_state(self):
        def job():
            for eng in self.engines.values():
                eng.drain()
                eng.cache.clear()
        self.cluster.at_gap(job, self.check_timeout)
        return {"ok": True, "reset": ["tokens", "cache"]}

    def patches(self):
        ops, eng = self.base_text["toy.ops"], self.base_text["toy.engine"]
        a2 = harness.edit_once(ops, "    return x + 1", "    return x + 2")
        a3 = harness.edit_once(a2, "    return x + 2", "    return x + 3")
        p = lambda **files: {"files": {"toy." + k: v for k, v in files.items()}}
        return {
            "noop": p(ops=harness.edit_once(ops, "    return x * STEP_SCALE", "    y = x * STEP_SCALE\n    return y")),
            "visible": p(ops=a2),
            "chain": [p(ops=a2), p(ops=a3), p(engine=harness.edit_once(eng, "        return y\n", "        return y * 10\n"))],
            "multi": p(ops=harness.edit_once(a2, "    return x * STEP_SCALE", "    return x * STEP_SCALE + 1"),
                       engine=harness.edit_once(eng, "        return y\n", "        return y - 1\n")),
            "unreached": {**p(engine=harness.edit_once(eng, "make_shift(10)", "make_shift(11)")), "accept": ["constructor_edit"]},
            "refused": {
                "syntax": p(ops=harness.edit_once(ops, "    return x + 1", "    return x +")),
                "closure": p(ops=harness.edit_once(ops, "        return x + k", "        return x + k + bias")),
                "module_state": p(ops=harness.edit_once(ops, "STEP_SCALE = 2", "STEP_SCALE = 3")),
                "empty": p(ops=harness.edit_once(ops, "def scale(x):", "def scale(x):  # faster")),
            },
        }

    def fresh_check(self, patch, workload=None):
        from kernel.toy_lab import FRESH
        inputs = (workload or self.WORKLOAD)["inputs"]
        tmp = tempfile.mkdtemp(prefix="minitoy-fresh-")
        try:
            for fname in ("__init__.py", "ops.py", "engine.py"):
                shutil.copy(os.path.join(toy.HERE, fname), tmp)
            for m, text in ((patch or {}).get("files") or {}).items():
                with open(os.path.join(tmp, m.split(".", 1)[1] + ".py"), "w", encoding="utf-8") as fh:
                    fh.write(text)
            proc_res = subprocess.run(
                [sys.executable, "-I", "-c", FRESH, LIVELAB, tmp, json.dumps(inputs), json.dumps(self.ranks)],
                capture_output=True, text=True, timeout=60,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if proc_res.returncode:
            return {"ok": False, "error": proc_res.stderr.strip().splitlines()[-1], "output": None}
        return {"ok": True, "output": json.loads(proc_res.stdout)}

    def inject(self, fault, **kw):
        rank = kw.get("participant", self.ranks[-1])
        if fault in ("fail_apply", "fail_commit"):
            return harness.inject_participant_fault(self.locals[rank], fault, **kw)
        if fault == "tamper":
            return harness.inject_participant_fault(self.locals[rank], "tamper", module="toy.ops", qualname="scale")
        if fault in ("crash", "hang"):
            self.faults[fault] = True
            return {"ok": True, "fault": fault, "participant": rank}
        if fault == "lost":
            self.faults["lost"] = rank
            return {"ok": True, "fault": fault, "participant": rank}
        return {"ok": False, "error": f"unknown fault {fault}"}


class BaseLabConformanceTests(unittest.TestCase):
    def test_baselab_subclass_passes_all_11_conformance_cases(self):
        """Proves `BaseLab` + `inject_participant_fault` + `edit_once` pass every conformance
        case c0-c10 end-to-end without re-implementing install/check/reset/uninstall/status."""
        for case_name in conformance_cases.CASES:
            with self.subTest(case=case_name):
                verdict = conformance_cases.run(case_name, MiniToyLab)
                self.assertEqual(verdict["result"], "pass", f"{case_name}: {verdict}")

    def test_baselab_clean_witness_and_rebuild_mechanisms_hooks(self):
        """Input: a `BaseLab` subclass using `RecaptureCoordinator` and overriding `_clean_witness`.
        Expected: committed rebuild appends planned rebuild recipe (`R1 recapture`) to `mechanisms`,
        and `_clean_witness` filters participant witness entries returned by `check()`."""
        class HookedMiniLab(MiniToyLab):
            def _start_engine(self):
                co, mods = super()._start_engine()
                rc = lab.RecaptureCoordinator(
                    co.participants,
                    gap=co.gap,
                    rebuild=lambda txn, spec: {"ok": True, "ranks": [{"rank": 0, "ok": True, "captured": [1]}]},
                )
                return rc, mods

            def _build_plan(self, patch, live_map):
                plan, fs = super()._build_plan(patch, live_map)
                plan["rebuild"] = ["recapture"]
                plan["recipes"] = ["R0 swap", "R1 recapture"]
                return plan, fs

            def _clean_witness(self, raw_witness, proof_ok):
                return {k: v for k, v in raw_witness.items() if k == "rank0"}

        hl = HookedMiniLab()
        self.addCleanup(hl.close)
        ins = hl.install(hl.patches()["visible"])
        self.assertTrue(ins["ok"], ins)
        self.assertEqual(ins["mechanisms"], ["R0 swap", "R1 recapture"])
        chk = hl.check()
        self.assertTrue(chk["proven"])
        self.assertEqual(list(chk["witness"]), ["rank0"])

    def test_baselab_accepts_stacked_unified_diffs_and_refuses_stale_diff(self):
        """Input: `MiniToyLab` (generic `BaseLab` subclass) receiving `install({"diff": ...})`
        for two stacked unified diffs on `toy.ops`, then an out-of-order stale diff.
        Expected: stacked diffs commit against live text, output updates, stale diff is refused
        with `patch_invalid`, and uninstall restores base output."""
        import difflib

        ml = MiniToyLab()
        self.addCleanup(ml.close)
        base_out = ml.check()["output"]
        ops0 = ml.base_text["toy.ops"]
        ops1 = harness.edit_once(ops0, "    return x + 1", "    return x + 2")
        ops2 = harness.edit_once(ops1, "    return x + 2", "    return x + 4")

        def make_diff(old, new):
            return "".join(difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), fromfile="a/toy/ops.py", tofile="b/toy/ops.py",
            ))

        d1, d2 = make_diff(ops0, ops1), make_diff(ops1, ops2)
        stale = ml.install({"diff": d2})
        self.assertEqual((stale["ok"], stale["outcome"], stale["codes"]), (False, "refused", ["patch_invalid"]))

        r1 = ml.install({"diff": d1})
        self.assertEqual((r1["ok"], r1["outcome"]), (True, "committed"), r1)
        r2 = ml.install({"diff": d2})
        self.assertEqual((r2["ok"], r2["outcome"]), (True, "committed"), r2)
        chk = ml.check()
        self.assertTrue(chk["ok"] and chk["proven"])
        self.assertNotEqual(chk["output"], base_out)
        self.assertTrue(ml.uninstall(r1["id"])["ok"])
        self.assertEqual(ml.check()["output"], base_out)


class CliTests(unittest.TestCase):
    def test_real_launch_args_keep_training_flags_after_delimiter(self):
        """The worker receives its launch options even when training reuses a CLI flag."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "launch_lab.py").write_text(
                "import json, os\n"
                "from kernel.toy_lab import ToyLab\n"
                "class LaunchLab(ToyLab):\n"
                "    @staticmethod\n"
                "    def launch_environment(argv):\n"
                "        return {'TRAINING_ARGS': json.dumps(argv)}\n"
                "    def __init__(self, receipts=None):\n"
                "        assert json.loads(os.environ['TRAINING_ARGS']) == "
                "['--root', 'tree with spaces', '--nproc', '2', '--', '--only', 'train']\n"
                "        super().__init__(receipts=receipts)\n"
            )
            result = subprocess.run(
                [sys.executable, "-m", "rsibox", "conformance", "launch_lab:LaunchLab",
                 "--real", "--path", tmp, "--root", "tree with spaces", "--nproc", "2",
                 "--only", "c0_repeatable", "--json", "--", "--only", "train"],
                cwd=LIVELAB, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1/1 as expected", result.stdout)
            self.assertIn('"lab": "launch_lab:LaunchLab"', result.stdout)

    def test_cli_list_doctor_conformance_and_ctl_end_to_end(self):
        """Input: invoke `rsibox.cli.main` for `list --json`, `doctor --json`,
        `conformance toy --only c0_repeatable,c2_visible`, and `ctl` over a real Unix socket.
        Expected: dynamic discovery finds `toy`, `template`, and repo engines without any
        hardcoded engine names; conformance passes; `ctl` maps ok/refused to exit 0/2."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["list", "--json"])
        self.assertEqual(rc, 0)
        names = {item["name"] for item in json.loads(buf.getvalue())["labs"]}
        self.assertIn("toy", names)
        self.assertIn("template", names)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["doctor", "--json"])
        self.assertEqual(rc, 0)
        doc = json.loads(buf.getvalue())
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["max_unix_sock_bytes"], ipc.MAX_SOCK_PATH)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["conformance", "toy", "--only", "c0_repeatable,c2_visible"])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("2/2 as expected", buf.getvalue())

        sock = ipc.short_sock_path("cli-ctl")

        def ctl_handler(req):
            if req.get("verb") == "install":
                return {"ok": False, "outcome": "refused", "codes": ["empty_change"]}
            return {"ok": True, "state": "base", "steps": req.get("steps")}

        with ipc.UnixJsonServer(sock, handler=ctl_handler):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_ok = cli.main(["ctl", "--sock", sock, "status", "--json-arg", "steps=3"])
            self.assertEqual(rc_ok, 0)
            self.assertEqual(json.loads(buf.getvalue()), {"ok": True, "state": "base", "steps": 3})

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_ref = cli.main(["ctl", "--sock", sock, "install"])
            self.assertEqual(rc_ref, 2)

    def test_resolve_lab_requires_real_flag_when_toy_spec_missing(self):
        """Input: a discovered lab exposing only a real-engine conformance class (`RealOnlyLab`)
        and no toy class. Expected: `resolve_lab(name, real=False)` raises `ValueError` telling
        the caller to pass `--real`, `cli.main(["conformance", name])` exits 2 without starting
        a real lab, and `resolve_lab(name, real=True)` returns the real spec."""
        tmp = tempfile.mkdtemp(prefix="rsibox-cli-realonly-")
        self.addCleanup(shutil.rmtree, tmp, True)
        os.makedirs(os.path.join(tmp, "core", "livelab"))
        real_dir = os.path.join(tmp, "repos", "realonly", "livelab")
        real_pkg = os.path.join(real_dir, "realpkg")
        toy_pkg = os.path.join(tmp, "repos", "toyonly", "livelab", "toypkg")
        os.makedirs(real_pkg)
        os.makedirs(toy_pkg)
        with open(os.path.join(real_pkg, "__init__.py"), "w", encoding="utf-8") as handle:
            handle.write("")
        with open(os.path.join(real_pkg, "conformance_lab.py"), "w", encoding="utf-8") as handle:
            handle.write("class RealOnlyLab:\n    pass\n")
        with open(os.path.join(toy_pkg, "__init__.py"), "w", encoding="utf-8") as handle:
            handle.write("")
        with open(os.path.join(toy_pkg, "toy_lab.py"), "w", encoding="utf-8") as handle:
            handle.write("class ToyOnlyLab:\n    pass\n")

        with self.assertRaisesRegex(ValueError, "--real.*realpkg\\.conformance_lab:RealOnlyLab"):
            cli.resolve_lab("realonly", real=False, repo_root=tmp)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["conformance", "realonly", "--repo-root", tmp])
        self.assertEqual(rc, 2)
        self.assertIn("--real", err.getvalue())

        self.assertEqual(
            cli.resolve_lab("realonly", real=True, repo_root=tmp),
            ("realpkg.conformance_lab:RealOnlyLab", real_dir),
        )
        with self.assertRaisesRegex(ValueError, "no real conformance class"):
            cli.resolve_lab("toyonly", real=True, repo_root=tmp)

    def test_conformance_prepends_livelab_core_for_explicit_module_spec_with_launch_args(self):
        """Regression test for [W2M-P1-1]: when an external entrypoint invokes
        `cli.main(["conformance", "ext_lab:ExtLab", "--real", "--", ...])` with only the
        engine's package dir in `PYTHONPATH` (not `core/livelab`), `resolve_lab` returns
        `extra_path=None`, so `_cmd_conformance` must still prepend `core/livelab` to
        `PYTHONPATH` before spawning `python3 -m conformance`."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ext_lab.py").write_text(
                "import json, os, sys\n"
                f"sys.path.insert(0, {LIVELAB!r})\n"
                "from kernel.toy_lab import ToyLab\n"
                "class ExtLab(ToyLab):\n"
                "    @staticmethod\n"
                "    def launch_environment(argv):\n"
                "        return {'EXT_LAUNCH_ARGS': json.dumps(argv)}\n"
                "    def __init__(self, receipts=None):\n"
                "        assert json.loads(os.environ['EXT_LAUNCH_ARGS']) == ['--flag', '1']\n"
                "        super().__init__(receipts=receipts)\n"
                "if __name__ == '__main__':\n"
                "    from rsibox import lab_cli as cli\n"
                "    raise SystemExit(cli.main([\n"
                "        'conformance', 'ext_lab:ExtLab', '--real', '--only', 'c0_repeatable', '--', '--flag', '1'\n"
                "    ]))\n",
                encoding="utf-8",
            )
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = tmp
            env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "ext_lab"],
                cwd=tmp, capture_output=True, text=True, env=env, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1/1 as expected", result.stdout)

    def test_conformance_rejects_launch_args_for_toy_lab(self):
        """Input: `rsibox conformance toy --only c0_repeatable -- --extra 1`.
        Expected: exits 2 cleanly without running conformance because toy labs are not `--real`."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["conformance", "toy", "--only", "c0_repeatable", "--", "--extra", "1"])
        self.assertEqual(rc, 2)
        self.assertIn("does not accept launch arguments", err.getvalue())

    def test_conformance_strips_at_most_one_leading_delimiter(self):
        """Input: `rsibox conformance <real-lab> --real` with (a) `--model M --tp 2`,
        (b) `-- --model M --tp 2`, (c) `--root /r --nproc 2 -- --only train`,
        (d) `-- --root /r --nproc 2 -- --only train`, and (e) bare trailing `--` on `toy`.
        Expected: `_cmd_conformance` strips at most one leading `--`, keeping any inner `--`
        delimiter intact for adapter `launch_environment(argv)` implementations."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "delim_lab.py").write_text(
                "import json\n"
                "from kernel.toy_lab import ToyLab\n"
                "class DelimLab(ToyLab):\n"
                "    seen = []\n"
                "    @staticmethod\n"
                "    def launch_environment(argv):\n"
                "        DelimLab.seen.append(list(argv))\n"
                "        return {'CAPTURED_ARGV': json.dumps(argv)}\n",
                encoding="utf-8",
            )
            cases = [
                (
                    ["conformance", "delim_lab:DelimLab", "--real", "--path", tmp,
                     "--model", "M", "--tp", "2"],
                    ["--model", "M", "--tp", "2"],
                ),
                (
                    ["conformance", "delim_lab:DelimLab", "--real", "--path", tmp,
                     "--", "--model", "M", "--tp", "2"],
                    ["--model", "M", "--tp", "2"],
                ),
                (
                    ["conformance", "delim_lab:DelimLab", "--real", "--path", tmp,
                     "--root", "/r", "--nproc", "2", "--", "--only", "train"],
                    ["--root", "/r", "--nproc", "2", "--", "--only", "train"],
                ),
                (
                    ["conformance", "delim_lab:DelimLab", "--real", "--path", tmp,
                     "--", "--root", "/r", "--nproc", "2", "--", "--only", "train"],
                    ["--root", "/r", "--nproc", "2", "--", "--only", "train"],
                ),
            ]
            from unittest import mock

            captured_envs = []

            def fake_run(cmd, env=None, **_kw):
                captured_envs.append(json.loads(env["CAPTURED_ARGV"]))
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch.object(cli.subprocess, "run", side_effect=fake_run):
                for argv, expected in cases:
                    with self.subTest(argv=argv):
                        rc = cli.main(argv)
                        self.assertEqual(rc, 0)
            self.assertEqual(captured_envs, [exp for _, exp in cases])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_empty = cli.main(["conformance", "toy", "--only", "c0_repeatable", "--"])
        self.assertEqual(rc_empty, 0, buf.getvalue())
        self.assertIn("1/1 as expected", buf.getvalue())

    def test_doctor_checks_core_health_all_repo_engines_and_human_table(self):
        """Input: `rsibox doctor` (both `--json` and human table) against the real repository.
        Expected: core kernel & 11 conformance cases healthy, every engine under `repos/`
        discovered and importable with zero eager GPU/torch imports, exit 0."""
        buf_json = io.StringIO()
        with contextlib.redirect_stdout(buf_json):
            rc_json = cli.main(["doctor", "--json"])
        self.assertEqual(rc_json, 0, buf_json.getvalue())
        doc = json.loads(buf_json.getvalue())
        self.assertTrue(doc["ok"], doc)
        self.assertTrue(doc["core"]["ok"])
        self.assertEqual(doc["core"]["conformance"]["case_count"], 11)
        repos_dir = os.path.join(doc["repo_root"], "repos")
        expected_engines = sorted(
            d for d in os.listdir(repos_dir)
            if not d.startswith(".") and os.path.isdir(os.path.join(repos_dir, d))
        )
        self.assertGreaterEqual(len(expected_engines), 6)
        self.assertEqual(sorted(doc["engines"]), expected_engines)
        self.assertEqual(doc["summary"]["engines_ok"], len(expected_engines))
        self.assertEqual(doc["summary"]["gpu_import_violations"], 0)
        by_name = {item["name"]: item for item in doc["labs"]}
        self.assertEqual(by_name["miles"]["real_spec"], "adapter.conformance_lab:MilesConformanceLab")
        self.assertTrue(by_name["miles"]["cli_launch"])
        self.assertTrue(by_name["sglang"]["cli_launch"])

        buf_txt = io.StringIO()
        with contextlib.redirect_stdout(buf_txt):
            rc_txt = cli.main(["doctor"])
        self.assertEqual(rc_txt, 0, buf_txt.getvalue())
        text = buf_txt.getvalue()
        self.assertIn("core:", text)
        self.assertIn("conformance 11/11", text)
        self.assertIn("cli_launch", text)
        for eng in expected_engines:
            self.assertIn(eng, text)

    def test_doctor_flags_eager_gpu_import_missing_methods_and_unregistered_engine(self):
        """Input: synthetic repo with (a) a clean engine, (b) an engine whose lab eagerly
        imports a GPU runtime (`torch`), (c) an engine lab missing required `LabUnderTest`
        methods, and (d) an engine directory without `livelab/`.
        Expected: `doctor --json` exits 1, reports `ok=False`, names each failing engine,
        and records the GPU import violation."""
        tmp = tempfile.mkdtemp(prefix="rsibox-doctor-neg-")
        self.addCleanup(shutil.rmtree, tmp, True)
        os.makedirs(os.path.join(tmp, "core", "livelab"))
        valid_body = (
            "class GoodLab:\n"
            "    def install(self, p): pass\n"
            "    def check(self, w=None): pass\n"
            "    def reset(self): pass\n"
            "    def uninstall(self, tid=None): pass\n"
            "    def status(self): pass\n"
            "    def patches(self): pass\n"
            "    def fresh_check(self, p, w=None): pass\n"
            "    def inject(self, f, **k): pass\n"
            "    def close(self): pass\n"
        )
        # (a) clean engine
        good_pkg = Path(tmp, "repos", "good_eng", "livelab", "adapter")
        good_pkg.mkdir(parents=True)
        (good_pkg / "__init__.py").write_text("")
        (good_pkg / "toy_lab.py").write_text(valid_body)
        # (b) eager torch import (simulated via stub torch.py on livelab_dir)
        gpu_dir = Path(tmp, "repos", "eager_gpu", "livelab")
        gpu_pkg = gpu_dir / "adapter"
        gpu_pkg.mkdir(parents=True)
        (gpu_dir / "torch.py").write_text("X = 1\n")
        (gpu_pkg / "__init__.py").write_text("")
        (gpu_pkg / "lab.py").write_text("import torch\n" + valid_body.replace("GoodLab", "EagerLab"))
        # (c) incomplete LabUnderTest API
        bad_pkg = Path(tmp, "repos", "incomplete_eng", "livelab", "adapter")
        bad_pkg.mkdir(parents=True)
        (bad_pkg / "__init__.py").write_text("")
        (bad_pkg / "toy_lab.py").write_text("class IncompleteLab:\n    def status(self): pass\n")
        # (d) engine folder without livelab/
        Path(tmp, "repos", "orphan_eng").mkdir(parents=True)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["doctor", "--repo-root", tmp, "--json"])
        self.assertEqual(rc, 1)
        doc = json.loads(buf.getvalue())
        self.assertFalse(doc["ok"])
        by_name = {item["name"]: item for item in doc["labs"]}
        self.assertTrue(by_name["good_eng"]["ok"])
        self.assertFalse(by_name["eager_gpu"]["ok"])
        self.assertEqual(by_name["eager_gpu"]["gpu_imports"], ["torch"])
        self.assertFalse(by_name["incomplete_eng"]["ok"])
        self.assertIn("install", by_name["incomplete_eng"]["toy"]["missing_methods"])
        self.assertFalse(by_name["orphan_eng"]["ok"])
        self.assertIn("missing livelab/ directory", by_name["orphan_eng"]["errors"])
        self.assertEqual(doc["summary"]["gpu_import_violations"], 1)

    def test_conformance_toy_alias_and_mutual_exclusion_with_real(self):
        """Input: `rsibox conformance toy --toy --only c0_repeatable` vs `--toy --real`.
        Expected: `--toy` runs the toy lab (exit 0); `--toy --real` exits 2 via argparse."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["conformance", "toy", "--toy", "--only", "c0_repeatable"])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("1/1 as expected", buf.getvalue())

        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cli.main(["conformance", "toy", "--toy", "--real"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("not allowed with argument --toy", err.getvalue())

    def test_livelab_json_schema_validation_and_doctor_cli_launch_reporting(self):
        """Input: synthetic repo with valid `real: null` manifest, valid real labs with and
        without `launch_environment`, and 4 invalid `livelab.json` manifests (bad JSON, list,
        unknown key, invalid spec). Expected: `real: null` is healthy with `cli_launch=None`
        and `--real` refuses cleanly; `cli_launch` reports True/False accurately; invalid
        manifests fail `doctor` and `conformance` without falling back to AST discovery."""
        tmp = tempfile.mkdtemp(prefix="rsibox-manifest-schema-")
        self.addCleanup(shutil.rmtree, tmp, True)
        os.makedirs(os.path.join(tmp, "core", "livelab"))
        valid_body = (
            "class GoodLab:\n"
            "    def install(self, p): pass\n"
            "    def check(self, w=None): pass\n"
            "    def reset(self): pass\n"
            "    def uninstall(self, tid=None): pass\n"
            "    def status(self): pass\n"
            "    def patches(self): pass\n"
            "    def fresh_check(self, p, w=None): pass\n"
            "    def inject(self, f, **k): pass\n"
            "    def close(self): pass\n"
        )
        launch_body = valid_body + "    @staticmethod\n    def launch_environment(argv): return {}\n"

        def make_pkg(eng_name, mod_name, body, manifest_text=None):
            ldir = Path(tmp, "repos", eng_name, "livelab")
            pkg = ldir / "adapter"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / f"{mod_name}.py").write_text(body)
            if manifest_text is not None:
                (ldir / "livelab.json").write_text(manifest_text)

        make_pkg("toy_null_real", "toy_lab", valid_body,
                 json.dumps({"toy": "adapter.toy_lab:GoodLab", "real": None}))
        make_pkg("real_cli", "lab", launch_body,
                 json.dumps({"toy": None, "real": "adapter.lab:GoodLab"}))
        make_pkg("real_nocli", "lab", valid_body,
                 json.dumps({"toy": None, "real": "adapter.lab:GoodLab"}))
        make_pkg("bad_json", "toy_lab", valid_body, "{not valid json")
        make_pkg("bad_type", "toy_lab", valid_body, '["adapter.toy_lab:GoodLab"]')
        make_pkg("bad_key", "toy_lab", valid_body,
                 json.dumps({"toyy": "adapter.toy_lab:GoodLab"}))
        make_pkg("bad_spec", "toy_lab", valid_body,
                 json.dumps({"toy": "adapter.toy_lab:GoodLab", "real": "no_colon"}))

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc_real = cli.main(["conformance", "toy_null_real", "--real", "--repo-root", tmp])
        self.assertEqual(rc_real, 2)
        self.assertIn("no real conformance class", err.getvalue())

        for broken in ("bad_json", "bad_type", "bad_key", "bad_spec"):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc_bad = cli.main(["conformance", broken, "--repo-root", tmp])
            self.assertEqual(rc_bad, 2)
            self.assertIn("invalid livelab.json", err.getvalue())

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_doc = cli.main(["doctor", "--repo-root", tmp, "--json"])
        self.assertEqual(rc_doc, 1)
        by_name = {item["name"]: item for item in json.loads(buf.getvalue())["labs"]}
        self.assertTrue(by_name["toy_null_real"]["ok"])
        self.assertIsNone(by_name["toy_null_real"]["cli_launch"])
        self.assertTrue(by_name["real_cli"]["ok"])
        self.assertTrue(by_name["real_cli"]["cli_launch"])
        self.assertTrue(by_name["real_nocli"]["ok"])
        self.assertFalse(by_name["real_nocli"]["cli_launch"])
        for broken in ("bad_json", "bad_type", "bad_key", "bad_spec"):
            self.assertFalse(by_name[broken]["ok"])
            self.assertTrue(any("invalid livelab.json" in e for e in by_name[broken]["errors"]),
                            by_name[broken])

    def test_conformance_subprocess_enables_unbuffered_streaming(self):
        """Input: `rsibox conformance <real-lab> --real -- --model M`.
        Expected: `_cmd_conformance` flushes stdout/stderr, invokes `[sys.executable, '-u',
        '-m', 'conformance', ...]`, and sets `PYTHONUNBUFFERED='1'` via `setdefault`."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "stream_lab.py").write_text(
                "from kernel.toy_lab import ToyLab\n"
                "class StreamLab(ToyLab):\n"
                "    @staticmethod\n"
                "    def launch_environment(argv):\n"
                "        return {'STREAM_SEEN': ' '.join(argv)}\n",
                encoding="utf-8",
            )
            calls = []
            flushed = []

            class _TrackStream(io.StringIO):
                def __init__(self, tag):
                    super().__init__()
                    self._tag = tag

                def flush(self):
                    flushed.append(self._tag)
                    super().flush()

            def fake_run(cmd, env=None, **_kw):
                calls.append((list(cmd), dict(env or {})))
                return subprocess.CompletedProcess(cmd, 0)

            clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
            with (
                mock.patch.dict(os.environ, clean_env, clear=True),
                mock.patch.object(cli.subprocess, "run", side_effect=fake_run),
                contextlib.redirect_stdout(_TrackStream("out")),
                contextlib.redirect_stderr(_TrackStream("err")),
            ):
                rc = cli.main([
                    "conformance", "stream_lab:StreamLab", "--real", "--path", tmp,
                    "--only", "c0_repeatable", "--", "--model", "M",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)
            cmd, env = calls[0]
            self.assertEqual(cmd[:4], [sys.executable, "-u", "-m", "conformance"])
            self.assertIn("--lab", cmd)
            self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")
            self.assertEqual(env.get("STREAM_SEEN"), "--model M")
            self.assertIn("out", flushed)
            self.assertIn("err", flushed)

    def test_conformance_rejects_unknown_only_case_invalid_spec_and_bad_ctl_json(self):
        """Input: unknown `--only` case name (and valid whitespace-separated `--only`),
        malformed `module:Class` target, and invalid JSON in `rsibox ctl`.
        Expected: crisp exit-2 errors before booting any lab; valid spaced `--only` runs."""
        from conformance import __main__ as conf_main

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["conformance", "toy", "--only", "c0_repeatable,c2_typo"])
        self.assertEqual(rc, 2)
        self.assertIn("c2_typo", err.getvalue())
        self.assertIn("valid:", err.getvalue())

        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            rc2 = conf_main.main(["--lab", "nonexistent_mod:NoLab", "--only", "bogus_case"])
        self.assertEqual(rc2, 2)
        self.assertIn("bogus_case", err2.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc_ws = cli.main(["conformance", "toy", "--only", "c0_repeatable, c1_noop"])
        self.assertEqual(rc_ws, 0, out.getvalue())
        self.assertIn("2/2 as expected", out.getvalue())

        err_spec = io.StringIO()
        with contextlib.redirect_stderr(err_spec):
            rc_spec = cli.main(["conformance", "bad:"])
        self.assertEqual(rc_spec, 2)
        self.assertIn("invalid lab spec", err_spec.getvalue())

        for bad_ctl in (
            ["ctl", "--sock", "/tmp/unused.sock", "--payload", "{bad"],
            ["ctl", "--sock", "/tmp/unused.sock", "status", "--json-arg", "steps=notjson"],
        ):
            err_ctl = io.StringIO()
            with contextlib.redirect_stderr(err_ctl):
                rc_ctl = cli.main(bad_ctl)
            self.assertEqual(rc_ctl, 2)
            self.assertIn("invalid JSON", err_ctl.getvalue())

    def test_real_engines_support_real_help_without_gpu_imports(self):
        """Input: `rsibox conformance <engine> --real -- --help` for all 7 real GPU engines.
        Expected: exits 0, prints engine-specific help text, and leaves GPU runtime libraries
        (`torch`, etc.) unimported."""
        repo_root = os.path.dirname(os.path.dirname(LIVELAB))
        probe = (
            "import sys\n"
            "from rsibox import lab_cli as cli\n"
            "engine, root, gpu_csv = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "before = set(sys.modules)\n"
            "try:\n"
            "    rc = cli.main(['conformance', engine, '--real', '--repo-root', root, '--', '--help'])\n"
            "except SystemExit as exc:\n"
            "    rc = exc.code\n"
            "loaded = sorted({m.split('.')[0] for m in (set(sys.modules) - before) if m.split('.')[0] in set(gpu_csv.split(','))})\n"
            "assert not loaded, f'eager GPU import(s) for {engine}: {loaded}'\n"
            "raise SystemExit(rc)\n"
        )
        env = {**os.environ, "PYTHONPATH": LIVELAB}
        for engine in ("vllm", "fastvideo", "megatron", "verl", "sglang", "miles", "transformers"):
            with self.subTest(engine=engine):
                proc = subprocess.run(
                    [sys.executable, "-c", probe, engine, repo_root, ",".join(cli._GPU_RUNTIME_LIBS)],
                    capture_output=True, text=True, env=env, timeout=30,
                )
                self.assertEqual(proc.returncode, 0, f"{engine}: stdout={proc.stdout} stderr={proc.stderr}")
                self.assertIn(f"rsibox conformance {engine} --real", proc.stdout)

    def test_megatron_launch_environment_strips_leading_separator_and_maps_opts(self):
        """Input: `MegatronLab.launch_environment` with leading `--`, `--steps`, `--check-timeout`,
        and trailing `-- --foo`. Expected: strips single leading `--`, sets `RSIBOX_CHECK_STEPS`
        and `RSIBOX_CHECK_TIMEOUT`, and forwards `--foo` to `pretrain_gpt.py`."""
        megatron_livelab = os.path.join(
            os.path.dirname(os.path.dirname(LIVELAB)), "repos", "megatron", "livelab"
        )
        sys.path.insert(0, megatron_livelab)
        try:
            from rsibox_megatron.conformance_lab import MegatronLab

            env = MegatronLab.launch_environment([
                "--", "--root", "/tmp/mg", "--nproc", "4", "--steps", "3",
                "--check-timeout", "180", "--", "--foo",
            ])
        finally:
            if megatron_livelab in sys.path:
                sys.path.remove(megatron_livelab)
        self.assertEqual(env["RSIBOX_MG_ROOT"], os.path.abspath("/tmp/mg"))
        self.assertEqual(env["RSIBOX_CHECK_STEPS"], "3")
        self.assertEqual(env["RSIBOX_CHECK_TIMEOUT"], "180.0")
        self.assertIn("--nproc-per-node 4", env["RSIBOX_MG_CMD"])
        self.assertTrue(env["RSIBOX_MG_CMD"].endswith("--foo"))


if __name__ == "__main__":
    unittest.main()
