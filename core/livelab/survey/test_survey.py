"""Golden cases for the survey on the fixture tree (fixture/upstream, a toy engine of 13
functions, with fixture/lab as its lab). Each case names the wrong answer it guards against.

Run from core/livelab:  python3 -m unittest discover -s survey -t . -v
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from survey import history as H
from survey import rules as R
from survey import scan as S
from survey.__main__ import main

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture")
SOURCE = os.path.join(HERE, "upstream")


def survey(lab=HERE):
    rules = R.load(os.path.join(HERE, "rules.json"))
    out, records = S.report(rules, SOURCE, S.Files(lab), None, "2026-01-01")
    return out, {(r["path"].split("/", 1)[1], r["qualname"]): r for r in records}


class ClassifyTest(unittest.TestCase):
    def test_a_constructor_is_a_rebuild_never_a_covered_swap(self):
        """Pain point: a constructor edit swaps cleanly and changes nothing, because the objects
        were built at start (kernel: constructor_edit). Counting `__init__` as a swap would
        report it live-editable on a lab that only swaps. Fails without the constructor rule:
        the function would fall to its path rule (GRAPH) or to the default (OP, covered)."""
        out, by = survey()
        r = by["model/layers.py", "Norm.__init__"]
        self.assertEqual((r["group"], r["install"], r["rule"], r["firm"]),
                         ("STATE", "rebuild", "constructor", True))
        self.assertTrue(out["lab"]["capabilities"]["swap"])     # the lab does swap ...
        self.assertFalse(r["covered"])                          # ... and that does not cover it
        self.assertEqual(out["functions"]["by_install"]["rebuild"], {"total": 3, "covered": 0})

    def test_a_triton_kernel_needs_its_own_rebuild(self):
        """Pain point: a swap of a JIT kernel replaces Python the launches never run. The
        kernel is NATIVE, covered only by a declared jit_rebuild capability (none here);
        another DSL's kernel goes to a respawn, which this lab has."""
        _, by = survey()
        k, d = by["model/layers.py", "kernel"], by["model/layers.py", "dsl_kernel"]
        self.assertEqual((k["group"], k["install"], k["capability"], k["covered"], k["firm"]),
                         ("NATIVE", "swap_recapture", "jit_rebuild", False, True))
        self.assertEqual((d["group"], d["install"], d["covered"]), ("NATIVE", "respawn", True))

    def test_what_is_firm_and_what_is_a_guess(self):
        """Pain point: a share that hides how much of it is guessed. `scale` matched no rule
        (default: OP); `Norm.forward` is GRAPH only because its folder is declared; `fused`
        is GRAPH by its decorator. Only the last is firm."""
        out, by = survey()
        self.assertEqual([(by[k]["group"], by[k]["basis"], by[k]["firm"]) for k in (
            ("ops.py", "scale"), ("model/layers.py", "Norm.forward"), ("model/layers.py", "fused"))],
            [("OP", "default", False), ("GRAPH", "declared", False), ("GRAPH", "syntax", True)])
        self.assertEqual((out["functions"]["firm"], out["functions"]["uncertain"]), (4, 9))

    def test_launch_flags_are_restart_by_path_and_by_body(self):
        """`cli.py` is declared restart-only (its module-level lines too); `tools.make_parser`
        is found by its `add_argument` call, a name-based guess."""
        _, by = survey()
        self.assertEqual({q: by["cli.py", q]["install"] for q in ("parser", "serve", "<module>")},
                         {"parser": "restart", "serve": "restart", "<module>": "restart"})
        m = by["tools.py", "make_parser"]
        self.assertEqual((m["group"], m["rule"], m["firm"], m["covered"]),
                         ("restart", "launch_flags", False, False))

    def test_counts_add_up_and_nothing_is_dropped(self):
        """Every function and every code line of a parsed file lands in exactly one install
        class; the file that does not parse and the native file are reported apart. A nested
        function counts with `add` (13 functions, not 14)."""
        out, by = survey()
        for unit, total in (("functions", 13), ("lines", 47)):
            s = out[unit]
            self.assertEqual(s["total"], total)
            self.assertEqual(sum(c["total"] for c in s["by_install"].values()), total)
            self.assertEqual(s["live_covered"] + s["respawn_covered"] + s["restart"] + s["not_covered"], total)
        self.assertEqual(by["ops.py", "add"]["lines"], 4)
        self.assertEqual(by["ops.py", "<module>"]["lines"], 2)    # docstring and SCALE
        self.assertEqual(out["unparsed"], {"files": 1, "lines": 1})
        self.assertEqual(out["native"]["by_ext"], {".cu": {"files": 1, "lines": 4}})
        self.assertEqual(sorted(out["modules"]), [
            "toyengine/__init__.py", "toyengine/cli.py", "toyengine/csrc", "toyengine/model",
            "toyengine/ops.py", "toyengine/tools.py"])


class CoverTest(unittest.TestCase):
    def test_coverage_comes_from_the_labs_files_not_from_the_rules(self):
        """Pain point: a rules file that claims a path the lab does not have. Without
        `adapter/host.py` nothing is live; with `adapter/recapture.py` added, the two GRAPH
        functions of the folder it reaches become covered, and only those."""
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab")
            shutil.copytree(os.path.join(HERE, "lab"), lab)
            os.remove(os.path.join(lab, "adapter", "host.py"))
            out, _ = survey(tmp)
            self.assertEqual((out["lab"]["capabilities"]["swap"], out["functions"]["live_covered"]),
                             (False, 0))
            for name in ("host.py", "recapture.py"):
                with open(os.path.join(lab, "adapter", name), "w") as f:
                    f.write("StepGapCoordinator\n")
            out, by = survey(tmp)
            self.assertEqual(out["functions"]["live_covered"], 4)
            self.assertEqual(sorted(q for (_, q), r in by.items() if r["install"] == "swap_recapture"
                                    and r["covered"]), ["Norm.forward", "fused"])

    def test_a_file_the_lab_does_not_reach_is_not_covered(self):
        """The toy lab's swap does not reach `tools.py` (as a lab that has no participant in
        the API server): its OP function stays uncovered, 2 of 3 swaps are covered."""
        out, by = survey()
        self.assertFalse(by["tools.py", "broken_helper"]["covered"])
        self.assertEqual(out["functions"]["by_install"]["swap"], {"total": 3, "covered": 2})


class HistoryTest(unittest.TestCase):
    def test_coverage_steps_at_the_commit_that_adds_the_lab_file(self):
        """A real git history of three lab commits: nothing, the swap host, the recapture.
        Live coverage reads 0, 2 and 4 functions, and `changed` marks the two steps. What it
        does not show: that an edit installs. It shows which commit added the file."""
        with tempfile.TemporaryDirectory() as repo:
            def git(*a):
                subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t",
                                "-c", "commit.gpgsign=false", *a], check=True, capture_output=True)
            git("init", "-q", "-b", "main")
            for msg, name, text in (("lab notes", "README.md", "notes\n"),
                                    ("host", "adapter/host.py", "StepGapCoordinator\n"),
                                    ("docs", "README.md", "more notes\n"),
                                    ("recapture", "adapter/recapture.py", "\n")):
                path = os.path.join(repo, "lab", name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
                git("add", "-A")
                git("commit", "-q", "-m", msg)
            rules = R.load(os.path.join(HERE, "rules.json"))
            out = H.series(rules, SOURCE, repo, "main", "lab", "2026-01-01")
        pts = out["points"]
        self.assertEqual([p["subject"] for p in pts], ["lab notes", "host", "docs", "recapture"])
        self.assertEqual([p["functions"]["live_covered"] for p in pts], [0, 2, 2, 4])
        self.assertEqual([p["changed"] for p in pts], [True, True, False, True])
        self.assertEqual([p["lines"]["by_install_covered"]["swap_recapture"] for p in pts], [0, 0, 0, 5])

    def test_the_command_line_writes_the_report_and_one_line_per_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, fns = os.path.join(tmp, "o.json"), os.path.join(tmp, "f.jsonl")
            self.assertEqual(main(["scan", "--rules", os.path.join(HERE, "rules.json"), "--source",
                                   SOURCE, "--repo", HERE, "--date", "2026-01-01", "--out", out,
                                   "--functions", fns]), 0)
            with open(out) as f:
                d = json.load(f)
            with open(fns) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual((d["schema"], d["ruleset_version"], d["engine_rules_version"], d["scan_date"]),
                         ("rsibox-survey/1", "1.1", "toy-1", "2026-01-01"))
        self.assertEqual(sum(r["kind"] == "function" for r in rows), d["functions"]["total"])
        self.assertNotIn(tmp, json.dumps(d))


class ReinitAndSubsystemTest(unittest.TestCase):
    def test_pure_field_constructor_is_reinit_while_imperative_constructor_is_rebuild(self):
        """Pain point: PR #156 re-initializes constructors whose edit only touches pure fresh
        fields (`self.cache = {}`), while imperative constructors (`self.weight = [1.0] * dim`),
        `__new__`, and constructors that read the field again still block without full rebuild."""
        with tempfile.TemporaryDirectory() as tmp:
            upstream = os.path.join(tmp, "upstream")
            lab = os.path.join(tmp, "lab")
            shutil.copytree(SOURCE, upstream)
            shutil.copytree(os.path.join(HERE, "lab"), lab)
            with open(os.path.join(upstream, "toyengine", "ops.py"), "a") as handle:
                handle.write(
                    "\n\nclass Scheduler:\n"
                    "    def __init__(self, limit):\n"
                    "        self.limit = limit\n"
                    "        self.pending = set()\n"
                    "\n\nclass Imperative:\n"
                    "    def __init__(self, items):\n"
                    "        self.seen = set()\n"
                    "        self.count = len(self.seen)\n"
                )
            rules = R.load(os.path.join(HERE, "rules.json"))
            rules["capabilities"]["constructor_reinit"] = {
                "evidence": [[{"path": "lab/adapter/reinit.py"}]],
                "unreached": ["toyengine/tools.py"],
            }
            out_before, by_before = S.report(rules, upstream, S.Files(tmp), None, "2026-01-01")
            index_before = {(r["path"].split("/", 1)[1], r["qualname"]): r for r in by_before}
            sched = index_before["ops.py", "Scheduler.__init__"]
            imp = index_before["ops.py", "Imperative.__init__"]
            norm = index_before["model/layers.py", "Norm.__init__"]
            self.assertEqual((sched["group"], sched["install"], sched["capability"],
                              sched["rule"], sched["firm"], sched["covered"]),
                             ("STATE", "rebuild", "constructor_reinit",
                              "constructor_reinit", True, False))
            self.assertEqual((imp["rule"], imp["capability"], imp["covered"]),
                             ("constructor", "rebuild", False))
            self.assertEqual((norm["rule"], norm["capability"], norm["covered"]),
                             ("constructor", "rebuild", False))

            with open(os.path.join(lab, "adapter", "reinit.py"), "w") as handle:
                handle.write("reinit\n")
            out_after, by_after = S.report(rules, upstream, S.Files(tmp), None, "2026-01-01")
            index_after = {(r["path"].split("/", 1)[1], r["qualname"]): r for r in by_after}
            self.assertTrue(out_after["lab"]["capabilities"]["constructor_reinit"])
            self.assertTrue(index_after["ops.py", "Scheduler.__init__"]["covered"])
            self.assertFalse(index_after["ops.py", "Imperative.__init__"]["covered"])
            self.assertFalse(index_after["model/layers.py", "Norm.__init__"]["covered"])
            self.assertEqual(out_after["functions"]["by_install"]["rebuild"],
                             {"total": 5, "covered": 1})

    def test_two_level_subsystems_and_capability_breakdown_sum_to_totals(self):
        """Pain point: top-level modules (`vllm/v1`, `sglang/srt`) hide which subsystem holds
        uncovered functions, and capability simulators drift if capability buckets do not sum
        back to exact whole-tree totals."""
        out, _ = survey()
        subsystems = out["by_subsystem"]
        self.assertIn("toyengine/model", subsystems)
        self.assertIn("toyengine/csrc", subsystems)
        for unit in ("functions", "lines"):
            whole = out[unit]
            self.assertEqual(
                sum(sub[unit]["total"] for sub in subsystems.values()),
                whole["total"],
            )
            self.assertEqual(
                sum(bucket["total"] for bucket in whole["by_cap"].values()) + whole["restart"],
                whole["total"],
            )
        self.assertEqual(
            sum(sub["native"]["lines"] for sub in subsystems.values()),
            out["native"]["lines"],
        )


if __name__ == "__main__":
    unittest.main()
