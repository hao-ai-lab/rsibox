"""Golden unit tests for rstack deterministic gate scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import claim_board  # noqa: E402
import facts_sync  # noqa: E402
import gpu_receipt_close  # noqa: E402
import merge_train  # noqa: E402
import pr_gate  # noqa: E402
import pre_push  # noqa: E402
import preflight  # noqa: E402
import trajectory_audit  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return res.stdout.strip()


def _init_repo(root: Path) -> Path:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "config", "user.name", "CI Bot")
    (root / "README.md").write_text("base\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "chore(repo): initial commit")
    _git(root, "branch", "base-ref")
    return root


def _valid_pr_body() -> str:
    return "\n".join(
        [
            "## What and why",
            "Test change.",
            "",
            "## Evidence",
            "- unittest ran cleanly.",
            "",
            "## Loop record",
            "- **Engine**: core",
            "- **Upstream PRs**: none",
            "- **Function groups**: gate",
            "- **Rule added**: none",
            "- **Overheads measured**: none",
            "- **Receipts**: none",
            "- **Not verified**: GPU runtime",
            "- **Ideal full GPU test**: 2x B200 integration run",
            "- **Actual tests run**: local unittest suite",
            "- **Why unrun GPU paths were skipped**: CPU-only gate tests",
        ]
    )


class RstackGatesTest(unittest.TestCase):
    def test_claim_board_lifecycle_and_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            self.assertEqual(
                claim_board.main(
                    ["--repo", str(repo), "acquire", "--issue", "290", "--owner", "agent-a", "--ttl", "100"]
                ),
                0,
            )
            # Second owner blocked on active lease.
            self.assertEqual(
                claim_board.main(
                    ["--repo", str(repo), "acquire", "--issue", "290", "--owner", "agent-b"]
                ),
                1,
            )
            # Same owner refreshes cleanly.
            self.assertEqual(
                claim_board.main(
                    ["--repo", str(repo), "acquire", "--issue", "290", "--owner", "agent-a", "--ttl", "0"]
                ),
                0,
            )
            # Expired lease can be reclaimed with --force-stale.
            self.assertEqual(
                claim_board.main(
                    [
                        "--repo",
                        str(repo),
                        "acquire",
                        "--issue",
                        "290",
                        "--owner",
                        "agent-b",
                        "--force-stale",
                        "--ttl",
                        "0",
                    ]
                ),
                0,
            )
            # Foreign release blocked; matching release succeeds.
            self.assertEqual(
                claim_board.main(["--repo", str(repo), "release", "--issue", "290", "--owner", "agent-a"]),
                1,
            )
            self.assertEqual(
                claim_board.main(["--repo", str(repo), "release", "--issue", "290", "--owner", "agent-b"]),
                0,
            )
            # Expire then gc check.
            self.assertEqual(
                claim_board.main(
                    ["--repo", str(repo), "acquire", "--issue", "291", "--owner", "agent-c", "--ttl", "0"]
                ),
                0,
            )
            self.assertEqual(claim_board.main(["--repo", str(repo), "status"]), 0)
            self.assertEqual(claim_board.main(["--repo", str(repo), "gc"]), 0)

    def test_preflight_loc_and_cross_engine_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            sglang_dir = repo / "repos" / "sglang"
            sglang_dir.mkdir(parents=True)
            (sglang_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")
            (sglang_dir / "test_mod.py").write_text(
                "\n".join(["a = 1"] * 30) + "\n", encoding="utf-8"
            )
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "feat(sglang): add mod and test")

            self.assertEqual(
                preflight.main(["--repo", str(repo), "--base", "base-ref", "--max-loc", "10"]),
                0,
            )
            self.assertEqual(
                preflight.main(["--repo", str(repo), "--base", "base-ref", "--max-loc", "0"]),
                1,
            )

            # Inject cross-engine import.
            (sglang_dir / "mod.py").write_text("from repos.vllm import helper\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "feat(sglang): bad cross import")
            self.assertEqual(
                preflight.main(["--repo", str(repo), "--base", "base-ref"]),
                1,
            )

    def test_pre_push_blocks_ai_trailer_leak_and_bad_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "core" / "livelab").mkdir(parents=True)
            (repo / "core" / "livelab" / "stub.py").write_text("VALUE = 1\n", encoding="utf-8")
            scripts_dir = repo / ".github" / "scripts"
            scripts_dir.mkdir(parents=True)
            leak_sh = scripts_dir / "leak-check.sh"
            leak_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(leak_sh, 0o755)
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "feat(core): add stub")

            body_file = repo / "pr_body.md"
            body_file.write_text(_valid_pr_body(), encoding="utf-8")

            self.assertEqual(
                pre_push.main(
                    [
                        "--repo",
                        str(repo),
                        "--base",
                        "base-ref",
                        "--title",
                        "feat(core): add stub",
                        "--body-file",
                        str(body_file),
                        "--skip-tests",
                    ]
                ),
                0,
            )
            # Bad title blocked.
            self.assertEqual(
                pre_push.main(
                    [
                        "--repo",
                        str(repo),
                        "--base",
                        "base-ref",
                        "--title",
                        "bad title",
                        "--body-file",
                        str(body_file),
                        "--skip-tests",
                    ]
                ),
                1,
            )
            # Missing GPU bullet blocked when core/livelab changed.
            bad_body = repo / "bad_body.md"
            bad_body.write_text("## What and why\nhi\n## Evidence\nok\n## Loop record\n", encoding="utf-8")
            self.assertEqual(
                pre_push.main(
                    [
                        "--repo",
                        str(repo),
                        "--base",
                        "base-ref",
                        "--title",
                        "feat(core): add stub",
                        "--body-file",
                        str(bad_body),
                        "--skip-tests",
                    ]
                ),
                1,
            )
            # Mocked leak-check failure blocks push.
            leak_sh.write_text("#!/usr/bin/env bash\necho leak >&2\nexit 1\n", encoding="utf-8")
            self.assertEqual(
                pre_push.main(
                    [
                        "--repo",
                        str(repo),
                        "--base",
                        "base-ref",
                        "--title",
                        "feat(core): add stub",
                        "--body-file",
                        str(body_file),
                        "--skip-tests",
                    ]
                ),
                1,
            )
            # AI attribution trailer in commit blocks push.
            leak_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (repo / "README.md").write_text("update\n", encoding="utf-8")
            _git(repo, "add", "README.md")
            trailer = "Co-authored" + "-by: Cla" + "ude <x@example.com>"
            _git(repo, "commit", "-m", f"docs(core): note\n\n{trailer}")
            self.assertEqual(
                pre_push.main(
                    [
                        "--repo",
                        str(repo),
                        "--base",
                        "base-ref",
                        "--title",
                        "feat(core): add stub",
                        "--body-file",
                        str(body_file),
                        "--skip-tests",
                    ]
                ),
                1,
            )

    def test_pr_gate_rollup_handles_cached_in_progress_and_failures(self) -> None:
        green_pr = {
            "mergeStateStatus": "CLEAN",
            "statusCheckRollup": [
                {"name": "formal", "status": "IN_PROGRESS", "conclusion": "SUCCESS"},
                {"name": "cpu", "status": "COMPLETED", "conclusion": "SUCCESS"},
            ],
        }
        state, counts = pr_gate.evaluate_rollup(green_pr)
        self.assertEqual(state, "green")
        self.assertEqual(len(counts["passed"]), 2)

        failed_pr = {
            "mergeStateStatus": "BLOCKED",
            "statusCheckRollup": [
                {"name": "ruff", "status": "COMPLETED", "conclusion": "FAILURE"},
            ],
        }
        state_fail, _ = pr_gate.evaluate_rollup(failed_pr)
        self.assertEqual(state_fail, "failed")

        pending_pr = {
            "mergeStateStatus": "BLOCKED",
            "statusCheckRollup": [
                {"name": "cpu", "status": "IN_PROGRESS", "conclusion": ""},
            ],
        }
        state_pend, _ = pr_gate.evaluate_rollup(pending_pr)
        self.assertEqual(state_pend, "pending")

        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            body = repo / "body.md"
            body.write_text(_valid_pr_body(), encoding="utf-8")
            self.assertEqual(
                pr_gate.main(
                    [
                        "--repo",
                        str(repo),
                        "validate-body",
                        "--body-file",
                        str(body),
                        "--require-gpu-fields",
                    ]
                ),
                0,
            )

    def test_trajectory_audit_detects_skip_mismatch_and_missing_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "transcript.jsonl"
            records = [
                {"call_id": "c1", "CommandLine": "python3 -m unittest"},
                {
                    "call_id": "c1",
                    "output": "Ran 12 tests in 0.1s\n\nOK (skipped=2)\nCommand exited with code 0\nabc1234",
                },
            ]
            tpath.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            claim = Path(tmp) / "claim.md"
            claim.write_text("Verified 12 tests, 0 skipped.", encoding="utf-8")

            # Mismatch on claim asserting 0 skipped.
            self.assertEqual(
                trajectory_audit.main(
                    ["--transcript", str(tpath), "--require-sha", "abc1234", "--claim-file", str(claim)]
                ),
                1,
            )
            # Honest claim passes.
            honest = Path(tmp) / "honest.md"
            honest.write_text("Ran 12 tests with 2 skipped.", encoding="utf-8")
            self.assertEqual(
                trajectory_audit.main(
                    ["--transcript", str(tpath), "--require-sha", "abc1234", "--claim-file", str(honest)]
                ),
                0,
            )
            # Missing SHA fails.
            self.assertEqual(
                trajectory_audit.main(
                    ["--transcript", str(tpath), "--require-sha", "deadbeef", "--claim-file", str(honest)]
                ),
                1,
            )

    def test_plugin_bundle_integrity(self) -> None:
        plugin_root = SCRIPTS_DIR.parent
        manifest_path = plugin_root / "plugin.json"
        self.assertTrue(manifest_path.is_file(), "plugin.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        allowed_keys = {
            "name",
            "displayName",
            "version",
            "description",
            "author",
            "homepage",
            "repository",
            "logo",
            "suggestedPrompts",
            "disabled",
        }
        self.assertTrue(set(manifest).issubset(allowed_keys))
        self.assertEqual(manifest.get("name"), "rstack")
        self.assertLessEqual(len(manifest.get("suggestedPrompts", [])), 3)

        rules_text = (plugin_root / "rules" / "AGENTS.md").read_text(encoding="utf-8")
        self.assertFalse(rules_text.lstrip().startswith("---"), "rules/AGENTS.md must not have frontmatter")

        expected_agents = {
            "rstack-worker.md",
            "rstack-gate1-auditor.md",
            "rstack-gate2-adversary.md",
            "rstack-site-syncer.md",
        }
        actual_agents = {p.name for p in (plugin_root / "agents").glob("*.md")}
        self.assertEqual(actual_agents, expected_agents)
        for agent_file in expected_agents:
            atext = (plugin_root / "agents" / agent_file).read_text(encoding="utf-8")
            self.assertTrue(atext.startswith("---\n"), f"{agent_file} missing YAML frontmatter")
            self.assertIn(f"name: {Path(agent_file).stem}", atext)
            self.assertIn("description:", atext)
            self.assertIn("subagent: true", atext)
            self.assertIn("mainAgent: false", atext)
            self.assertIn("tools:", atext)

        expected_skills = {
            "rsibox-mode",
            "rsibox-claim",
            "rsibox-build",
            "rsibox-precheck",
            "rsibox-gpu-receipt",
            "rsibox-audit",
            "rsibox-triage",
            "rsibox-merge-train",
            "rsibox-facts-sync",
        }
        for skill in expected_skills:
            skill_md = plugin_root / "skills" / skill / "SKILL.md"
            self.assertTrue(skill_md.is_file(), f"missing {skill_md}")
            text = skill_md.read_text(encoding="utf-8")
            self.assertIn(f"name: {skill}", text)
            self.assertIn("description:", text)

        workflows_dir = plugin_root / "skills" / "rsibox-mode" / "workflows"
        expected_wfs = {
            "01-claim-and-build.md",
            "02-gpu-receipt-and-witness.md",
            "03-adversarial-review-triangle.md",
            "04-dag-merge-and-site-sync.md",
        }
        self.assertEqual({p.name for p in workflows_dir.glob("*.md")}, expected_wfs)

    def test_merge_train_dag_patch_id_and_worktree_gc(self) -> None:
        prs = [
            {"number": 12, "isDraft": True, "baseRefName": "feat-10", "body": "Depends on #11"},
            {"number": 11, "isDraft": False, "baseRefName": "main", "body": "Depends on #10"},
            {"number": 10, "isDraft": False, "baseRefName": "main", "body": "Base PR"},
        ]
        ok, rep = merge_train.plan_dag(prs)
        self.assertTrue(ok)
        self.assertEqual(rep["order"], [10, 11, 12])
        self.assertEqual(rep["ready"], [10])
        self.assertEqual(rep["blocked"], {"11": [10], "12": [11]})

        cyclic = [
            {"number": 1, "isDraft": False, "baseRefName": "main", "body": "Depends on #2"},
            {"number": 2, "isDraft": False, "baseRefName": "main", "body": "Depends on #1"},
        ]
        ok_cyc, rep_cyc = merge_train.plan_dag(cyclic)
        self.assertFalse(ok_cyc)
        self.assertEqual(rep_cyc["cycle"], [1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            prs_file = repo / "prs.json"
            prs_file.write_text(json.dumps(prs), encoding="utf-8")
            self.assertEqual(
                merge_train.main(["--repo", str(repo), "plan", "--prs-json", str(prs_file)]),
                0,
            )
            (repo / "f.py").write_text("x = 1\n", encoding="utf-8")
            _git(repo, "add", "f.py")
            _git(repo, "commit", "-m", "feat(core): patch")
            self.assertEqual(
                merge_train.main(["--repo", str(repo), "patch-id", "--base", "base-ref", "--head", "HEAD"]),
                0,
            )
            self.assertEqual(
                merge_train.main(["--repo", str(repo), "worktree-gc", "--base", "base-ref"]),
                0,
            )

    def test_gpu_receipt_close_validation_and_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            good = repo / "good.json"
            good.write_text(
                json.dumps(
                    {
                        "witness_nonzero": True,
                        "replay_verified": True,
                        "shapes": ["bs1_seq128", "bs8_seq1024"],
                        "cold_s": 12.4,
                        "warm_s": 0.42,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(gpu_receipt_close.main(["validate", "--receipt", str(good)]), 0)

            bad = repo / "bad.json"
            bad.write_text(
                json.dumps(
                    {
                        "witness_nonzero": False,
                        "replay_verified": False,
                        "shapes": ["only_one"],
                        "cold_s": -1,
                        "host_ip": ".".join([str(10), str(20), str(30), str(40)]),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(gpu_receipt_close.main(["validate", "--receipt", str(bad)]), 1)
            errs = gpu_receipt_close.validate_receipt(
                {
                    "witness_nonzero": True,
                    "replay_verified": True,
                    "shapes": ["a", "b"],
                    "cold_s": 1.0,
                    "warm_s": 0.1,
                },
                raw_text="path=" + chr(47) + "r" + "aid" + chr(47) + "x",
            )
            self.assertTrue(any("path leaked" in e for e in errs))
            self.assertEqual(
                gpu_receipt_close.main(
                    ["--repo", str(repo), "backlog", "--title", "[T1] GPU receipt", "--pr", "295", "--dry-run"]
                ),
                0,
            )

    def test_facts_sync_passes_and_catches_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "scripts").mkdir()
            (repo / "site" / "facts").mkdir(parents=True)
            (repo / "scripts" / "verify_facts.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
            (repo / "scripts" / "build_site.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
            matrix = repo / "site" / "facts" / "delivery-matrix.json"
            matrix.write_text(json.dumps({"engines": ["vllm"]}), encoding="utf-8")
            self.assertEqual(facts_sync.main(["--repo", str(repo), "--skip-tests"]), 0)

            matrix.write_text("{broken", encoding="utf-8")
            self.assertEqual(facts_sync.main(["--repo", str(repo), "--skip-tests"]), 1)


if __name__ == "__main__":
    unittest.main()
