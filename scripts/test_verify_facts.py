"""Golden unit tests for `scripts/verify_facts.py` (pure Python stdlib, read-only).

Run from the repository root:
    python3 -m unittest discover -s scripts -p 'test_*.py' -v
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_facts as VF  # noqa: E402


def _build_clean_repo(root: Path) -> None:
    """Populate a minimal, 100% consistent synthetic repo under `root`."""
    for eng in ("alpha", "beta"):
        pkg = root / "repos" / eng / "livelab" / "adapter"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "conformance_lab.py").write_text(f"class {eng.title()}Lab:\n    pass\n", encoding="utf-8")

    facts = root / "site" / "facts"
    facts.mkdir(parents=True)
    (facts / "engines.json").write_text(json.dumps({
        "shared_kernel": {"what": "kernel"},
        "engines": [
            {"engine": "Alpha", "folder": "repos/alpha", "mechanisms": []},
            {"engine": "Beta", "folder": "repos/beta", "mechanisms": []},
        ],
        "caveats": [],
    }), encoding="utf-8")
    (facts / "conversion.json").write_text(json.dumps({
        "meta": {"whole_repo": {}},
        "engines": [
            {"engine": "Alpha", "folder": "repos/alpha", "recorded_facts": []},
            {"engine": "Beta", "folder": "repos/beta", "recorded_facts": []},
        ],
    }), encoding="utf-8")
    (facts / "coverage.json").write_text(json.dumps({
        "measured": [
            {
                "id": "conformance",
                "engine": "all",
                "what": "Protocol conformance c0-c10",
                "unit": "cases",
                "results": {
                    "toy lab (CPU)": "11/11",
                    "Alpha (2 GPUs)": "11/11",
                    "Beta (2 GPUs)": "11/11",
                },
                "source": "core/livelab/conformance/README.md",
            }
        ],
        "not_covered_stated_by_repo": {
            "Alpha": ["none"],
            "Beta": ["none"],
            "source": "repos/README.md",
        },
    }), encoding="utf-8")
    (facts / "prs.json").write_text(json.dumps({
        "meta": {"counts": {"vllm_named_with_result": 1}},
        "engines": [
            {
                "engine": "Alpha",
                "aggregate": {"what": "replay"},
                "prs": [{"pr": 101, "upstream_repo": "org/alpha", "installed_live": True}],
            }
        ],
    }), encoding="utf-8")
    (facts / "case-copy.json").write_text(json.dumps({
        "blocks": [{"key": "b1", "label": "What"}, {"key": "b2", "label": "Why"}],
        "cases": [
            {"pr": 101, "b1": "change", "b2": "fast"},
            {"pr": "agent-opt-1", "b1": "opt", "b2": "fast"},
        ],
    }), encoding="utf-8")
    (facts / "case-timing.json").write_text(json.dumps({
        "cases": [
            {
                "engine": "Alpha",
                "id": "101",
                "source": "upstream_pr",
                "variant": "",
                "restart": {"total": 10.0, "outcome": "restarted", "phases": [["Init", 8.0], ["Check", 2.0]]},
                "fastrsi": {"total": 2.2, "outcome": "installed live", "phases": [["Install", 0.2], ["Check", 2.0]]},
                "improved": {"total": 1.9, "outcome": "installed live"},
                "effect": [{"method": "paired", "base_step_s": 2.0, "edit_step_s": 1.8, "ratio": 0.9}],
            },
            {
                "engine": "Beta",
                "id": "agent-opt-1",
                "source": "agent_edit",
                "variant": "",
                "restart": {"total": 12.0, "outcome": "restarted", "phases": [["Init", 10.0], ["Check", 2.0]]},
                "fastrsi": {"total": 2.5, "outcome": "installed live", "phases": [["Install", 0.5], ["Check", 2.0]]},
                "improved": None,
                "effect": [],
            },
        ]
    }), encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic\n\n"
        "## What is in the repo today\n\n"
        "| Engine | CPU | GPU | Gaps |\n"
        "|---|---|---|---|\n"
        "| Toy lab | 11/11 | n/a | |\n"
        "| Alpha | tests | 11/11 | |\n"
        "| Beta | tests | 11/11 | |\n\n"
        "## Repository layout\n\n"
        "```\n"
        "repos/<engine>/      alpha, beta\n"
        "```\n",
        encoding="utf-8",
    )


class VerifyFactsTests(unittest.TestCase):
    def test_clean_synthetic_repo_passes_strict(self):
        """Input: fully consistent synthetic repo. Expected: 0 errors, 0 drifts, exit 0 even with `--strict`."""
        with tempfile.TemporaryDirectory() as tmp:
            _build_clean_repo(Path(tmp))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = VF.main(["--repo-root", tmp, "--strict", "--json"])
            self.assertEqual(rc, 0, buf.getvalue())
            rep = json.loads(buf.getvalue())
            self.assertTrue(rep["ok"])
            self.assertTrue(rep["schema_ok"])
            self.assertEqual(rep["summary"]["errors"], 0)
            self.assertEqual(rep["summary"]["drifts"], 0)

    def test_schema_arithmetic_and_join_errors_exit_1(self):
        """Input: synthetic repo mutated with missing engine, count mismatch, phase sum drift,
        effect ratio drift, and orphaned case IDs. Expected: exit 1 and every error code reported."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_clean_repo(root)
            # Add an engine folder not listed in engines.json or conversion.json
            (root / "repos" / "gamma").mkdir()
            # Break prs.json meta.counts (vLLM engine with wrong count)
            prs_p = root / "site" / "facts" / "prs.json"
            prs_d = json.loads(prs_p.read_text(encoding="utf-8"))
            prs_d["meta"]["counts"] = {"megatron_replayed_with_result": 99}
            prs_d["engines"].append({
                "engine": "Megatron-LM",
                "aggregate": {},
                "prs": [{"pr": 777, "upstream_repo": "org/meg", "installed_live": False}],
            })
            prs_p.write_text(json.dumps(prs_d), encoding="utf-8")
            # Break case-timing phase sum, effect ratio, and add an orphan case id
            ct_p = root / "site" / "facts" / "case-timing.json"
            ct_d = json.loads(ct_p.read_text(encoding="utf-8"))
            ct_d["cases"][0]["restart"]["total"] = 99.0
            ct_d["cases"][0]["effect"][0]["ratio"] = 0.5
            ct_d["cases"].append({
                "engine": "Alpha",
                "id": "orphan-999",
                "source": "agent_edit",
                "variant": "",
                "restart": {"total": 1.0, "outcome": "restarted", "phases": [["Init", 1.0]]},
                "fastrsi": {"total": 1.0, "outcome": "installed live", "phases": [["Install", 1.0]]},
                "improved": None,
                "effect": [],
            })
            ct_p.write_text(json.dumps(ct_d), encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = VF.main(["--repo-root", tmp, "--json"])
            self.assertEqual(rc, 1)
            rep = json.loads(buf.getvalue())
            codes = {f["code"] for f in rep["findings"] if f["severity"] == "error"}
            for expected in (
                "missing_engine_in_facts",
                "missing_engine_in_conversion",
                "unknown_engine_name",
                "prs_count_mismatch",
                "prs_timing_id_mismatch",
                "timing_phase_sum_mismatch",
                "effect_ratio_mismatch",
                "case_copy_timing_id_mismatch",
            ):
                self.assertIn(expected, codes, f"missing {expected} in {codes}")

    def test_split_brain_drifts_flagged_and_strict_exits_2(self):
        """Input: schema-valid synthetic repo where `Beta` is omitted from README/coverage and
        `Alpha` is claimed to have no conformance adapter despite `AlphaLab` existing.
        Expected: exit 0 without `--strict`, exit 2 with `--strict`, and all 5 drift codes reported."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_clean_repo(root)
            cov_p = root / "site" / "facts" / "coverage.json"
            cov_d = json.loads(cov_p.read_text(encoding="utf-8"))
            del cov_d["not_covered_stated_by_repo"]["Beta"]
            cov_d["measured"][0]["results"] = {"toy lab (CPU)": "11/11", "Alpha": None}
            cov_p.write_text(json.dumps(cov_d), encoding="utf-8")
            (root / "README.md").write_text(
                "# Synthetic\n\n"
                "## What is in the repo today\n\n"
                "| Engine | CPU | GPU | Gaps |\n"
                "|---|---|---|---|\n"
                "| Toy lab | 11/11 | n/a | |\n"
                "| Alpha | tests | runs | no conformance adapter yet |\n\n"
                "## Repository layout\n\n"
                "```\n"
                "repos/<engine>/      alpha\n"
                "```\n",
                encoding="utf-8",
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_default = VF.main(["--repo-root", tmp, "--json"])
            self.assertEqual(rc_default, 0)
            rep = json.loads(buf.getvalue())
            self.assertTrue(rep["schema_ok"])
            self.assertFalse(rep["ok"])
            drift_codes = {f["code"] for f in rep["findings"] if f["severity"] == "drift"}
            self.assertEqual(drift_codes, {
                "coverage_not_covered_missing_engine",
                "coverage_conformance_missing_engine",
                "coverage_conformance_null_with_adapter",
                "readme_table_missing_engine",
                "readme_stale_conformance_claim",
                "readme_layout_missing_engine",
            })

            buf_strict = io.StringIO()
            with contextlib.redirect_stdout(buf_strict):
                rc_strict = VF.main(["--repo-root", tmp, "--strict"])
            self.assertEqual(rc_strict, 2)

    def test_real_repo_has_zero_schema_errors_and_surfaces_known_drifts(self):
        """Input: actual repository root (read-only).
        Expected: 0 schema/math/join errors (`schema_ok=True`, exit 0 default), while `--strict`
        exits 2 and reports the known README/coverage split-brain drifts without editing any file."""
        rep = VF.verify_repo(REPO_ROOT)
        self.assertTrue(rep["schema_ok"], rep["findings"])
        self.assertEqual(rep["summary"]["errors"], 0)
        self.assertGreaterEqual(len(rep["engines"]), 6)
        self.assertGreater(rep["summary"]["drifts"], 0)
        drift_codes = {f["code"] for f in rep["findings"] if f["severity"] == "drift"}
        self.assertIn("readme_table_missing_engine", drift_codes)
        self.assertIn("coverage_conformance_missing_engine", drift_codes)


if __name__ == "__main__":
    unittest.main()
