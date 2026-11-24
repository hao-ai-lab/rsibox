"""Real end-to-end subprocess tests for `rsibox.promote` and `rsibox.doctor`.

No mocking of subprocess execution: every promotion test runs real child Python
processes across alternating `(base, candidate)` / `(candidate, base)` pairs and
multiple input shapes, then verifies signed receipts and process-group cleanup.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rsibox import doctor, proc as proc_mod, promote

LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(LIVELAB))

_EVAL_SCRIPT = """
import json, os, subprocess, sys, time

mode = sys.argv[1]
pid_log = sys.argv[2]
shapes = json.loads(os.environ["RSIBOX_PROMOTE_SHAPES"])
arm = os.environ["RSIBOX_PROMOTE_ARM"]
pair_idx = int(os.environ["RSIBOX_PROMOTE_PAIR"])

with open(pid_log, "a", encoding="utf-8") as fh:
    fh.write(f"{arm}:{os.getpid()}\\n")

if mode == "timeout_grandchild":
    gc_file = sys.argv[3]
    gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with open(gc_file, "a", encoding="utf-8") as fh:
        fh.write(f"{gc.pid}\\n")
    time.sleep(30)

results = []
for idx, shape in enumerate(shapes):
    n = int(shape["len"]) if isinstance(shape, dict) else len(str(shape))
    out = [round((i + 1) * 0.25, 6) for i in range(n)]
    ok = True
    recompiles = 1 if idx == 0 else 0
    metric = 0.040 + 0.001 * n

    if mode == "fast_exact":
        metric = 0.020 + 0.0005 * n
    elif mode == "fast_1p2x":
        metric = (0.040 + 0.001 * n) / 1.2
    elif mode == "higher_better_cand":
        metric = (0.040 + 0.001 * n) * 2.0
    elif mode == "fast_within_tol":
        metric = 0.020 + 0.0005 * n
        out = [round(v + 1e-5, 8) for v in out]
    elif mode == "parity_bad":
        metric = 0.020 + 0.0005 * n
        if idx == 1:
            out[-1] = out[-1] + 0.5
    elif mode == "fail_shape2":
        metric = 0.020 + 0.0005 * n
        if idx == 1:
            ok = False
    elif mode == "recompile_shape2":
        metric = 0.020 + 0.0005 * n
        if idx == 1:
            recompiles = 2
    elif mode == "slower":
        metric = 0.080 + 0.002 * n
    elif mode == "odd_pair_zero":
        metric = 0.0 if (pair_idx % 2 == 1 and idx == 1) else (0.020 + 0.0005 * n)
    elif mode == "odd_pair_nan":
        metric = float("nan") if (pair_idx % 2 == 1 and idx == 1) else (0.020 + 0.0005 * n)
    elif mode == "odd_pair_inf":
        metric = float("inf") if (pair_idx % 2 == 1 and idx == 1) else (0.020 + 0.0005 * n)
    elif mode == "neg_cancelling_metric":
        metric = -0.010 if idx == 0 else 0.025
    elif mode == "bad_recompiles_neg":
        metric = 0.020 + 0.0005 * n
        recompiles = -1
    elif mode == "bad_recompiles_bool":
        metric = 0.020 + 0.0005 * n
        recompiles = True
    elif mode == "bad_recompiles_str":
        metric = 0.020 + 0.0005 * n
        recompiles = "two"
    elif mode == "bad_ok_str":
        metric = 0.020 + 0.0005 * n
        ok = "false"

    entry = {
        "shape": shape.get("name", f"s{idx}") if isinstance(shape, dict) else str(shape),
        "ok": ok,
        "output": out,
        "metric": metric,
        "recompiles": recompiles,
    }
    if mode == "missing_recompiles":
        entry.pop("recompiles")
    if mode == "missing_output":
        entry.pop("output")
    results.append(entry)

if mode in ("odd_pair_nan", "odd_pair_inf"):
    print("RSIBOX_PROMOTE_RESULT:" + json.dumps({"ok": True, "shapes": results}, allow_nan=True))
else:
    print("RSIBOX_PROMOTE_RESULT:" + json.dumps({"ok": True, "shapes": results}))
"""


class PromoteGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rsibox-promote-test-")
        self.addCleanup(self.tmp.cleanup)
        self.script = Path(self.tmp.name) / "eval_worker.py"
        self.script.write_text(_EVAL_SCRIPT, encoding="utf-8")
        self.pid_log = Path(self.tmp.name) / "pids.log"
        self.shapes = [{"name": "short", "len": 4}, {"name": "long", "len": 8}]

    def _cmd(self, mode: str, *extra: str) -> list[str]:
        return [sys.executable, str(self.script), mode, str(self.pid_log), *extra]

    def test_faster_bit_exact_candidate_promotes_with_receipt_binding_and_fresh_processes(self):
        """Proves faster bit-exact candidate promotes across 2 shapes in fresh alternating PIDs,
        records cmd digests + candidate_id, and supports both SHA-256 and HMAC-SHA256 verification."""
        receipt_file = Path(self.tmp.name) / "receipt.json"
        rec = promote.run_promotion(
            self._cmd("base"), self._cmd("fast_exact"), self.shapes,
            pairs=2, candidate_id="cand-v1", hmac_key="test-key", receipt_path=receipt_file,
        )
        self.assertEqual(rec["verdict"], "promote", rec)
        self.assertEqual(rec["reasons"], [])
        self.assertEqual([p["order"] for p in rec["pairs"]], [["base", "candidate"], ["candidate", "base"]])
        self.assertEqual(rec["summary"]["wins"], 2)
        self.assertEqual(rec["artifacts"]["candidate_id"], "cand-v1")
        self.assertEqual(len(rec["artifacts"]["base_cmd_sha256"]), 64)
        self.assertEqual(len(rec["artifacts"]["candidate_cmd_sha256"]), 64)
        self.assertEqual(rec["signature_mode"], "hmac-sha256")

        entries = [ln.split(":") for ln in self.pid_log.read_text(encoding="utf-8").splitlines() if ln]
        self.assertEqual([e[0] for e in entries], ["base", "candidate", "candidate", "base"])
        self.assertEqual(len({e[1] for e in entries}), 4)

        self.assertTrue(promote.verify_promotion_receipt(receipt_file, hmac_key="test-key")["ok"])
        self.assertFalse(promote.verify_promotion_receipt(receipt_file, hmac_key="wrong-key")["ok"])

    def test_cheats_and_odd_pair_invalid_metrics_are_rejected_fail_closed(self):
        """Proves shape #2 failures, extra recompiles, parity errors, slower runs,
        odd-pair 0/NaN/Inf metrics, and cancelling negative metrics are all rejected."""
        cases = [
            ("fail_shape2", "failed shapes ['long']"),
            ("recompile_shape2", "excessive recompiles"),
            ("parity_bad", "numerical parity failed"),
            ("slower", "insufficient speedup"),
            ("odd_pair_zero", "finite positive metric"),
            ("odd_pair_nan", "non-standard JSON constant"),
            ("odd_pair_inf", "non-standard JSON constant"),
            ("neg_cancelling_metric", "finite positive metric"),
            ("bad_recompiles_neg", "'recompiles' must be int >= 0"),
            ("bad_recompiles_bool", "'recompiles' must be int >= 0"),
            ("bad_recompiles_str", "'recompiles' must be int >= 0"),
            ("missing_recompiles", "'recompiles' must be int >= 0"),
            ("missing_output", "missing 'output' field"),
            ("bad_ok_str", "'ok' must be a boolean"),
        ]
        for mode, expected_substr in cases:
            with self.subTest(mode=mode):
                rec = promote.run_promotion(self._cmd("base"), self._cmd(mode), self.shapes, pairs=2)
                self.assertEqual(rec["verdict"], "reject", rec)
                self.assertTrue(any(expected_substr in r for r in rec["reasons"]), f"expected {expected_substr!r} in {rec['reasons']}")
                self.assertTrue(promote.verify_promotion_receipt(rec)["ok"])

        # Opt-in recompile override and tolerance parity + exact parity reason check.
        allowed = promote.run_promotion(self._cmd("base"), self._cmd("recompile_shape2"), self.shapes, pairs=2, allow_extra_recompiles=True)
        self.assertEqual(allowed["verdict"], "promote", allowed)

        exact_rec = promote.run_promotion(self._cmd("base"), self._cmd("fast_within_tol"), self.shapes, pairs=2)
        self.assertEqual(exact_rec["verdict"], "reject")
        self.assertTrue(any("numerical parity" in r for r in exact_rec["reasons"]), exact_rec["reasons"])
        tol_rec = promote.run_promotion(self._cmd("base"), self._cmd("fast_within_tol"), self.shapes, pairs=2, atol=1e-4)
        self.assertEqual(tol_rec["verdict"], "promote", tol_rec)

        # Higher-is-better and min_speedup > 1 checks.
        hb_rec = promote.run_promotion(self._cmd("base"), self._cmd("higher_better_cand"), self.shapes, pairs=2, higher_is_better=True)
        self.assertEqual(hb_rec["verdict"], "promote", hb_rec)
        self.assertGreater(hb_rec["summary"]["median_speedup"], 1.9)

        slow_thresh = promote.run_promotion(self._cmd("base"), self._cmd("fast_1p2x"), self.shapes, pairs=2, min_speedup=1.5)
        self.assertEqual(slow_thresh["verdict"], "reject")
        self.assertTrue(any("insufficient speedup" in r for r in slow_thresh["reasons"]))

    def test_timeout_kills_and_reaps_grandchild_process_group(self):
        """Proves timeout kills the entire process group so grandchild workers never leak."""
        gc_file = Path(self.tmp.name) / "grandchild.pid"
        rec = promote.run_promotion(
            self._cmd("base"), self._cmd("timeout_grandchild", str(gc_file)), self.shapes,
            pairs=2, timeout_s=0.5,
        )
        self.assertEqual(rec["verdict"], "reject")
        self.assertTrue(any("TimeoutExpired" in r for r in rec["reasons"]), rec["reasons"])
        self.assertTrue(gc_file.is_file(), "grandchild pid file was not written before timeout")
        gc_pids = [int(ln) for ln in gc_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(gc_pids), 2)
        for gc_pid in gc_pids:
            self.assertFalse(proc_mod.pid_alive(gc_pid), f"grandchild pid {gc_pid} survived timeout!")

    def test_policy_loopholes_refused_and_single_pass_token_expansion(self):
        """Proves single-shape, --min-shapes 1, odd/single pairs, bad thresholds refused fail-closed,
        and single-pass argv substitution does not corrupt literal '{arm}' inside shapes."""
        with self.assertRaisesRegex(ValueError, "at least 2 distinct input shapes"):
            promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), [self.shapes[0]])
        with self.assertRaisesRegex(ValueError, "at least 2 distinct input shapes"):
            promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), [self.shapes[0], dict(self.shapes[0])])
        with self.assertRaisesRegex(ValueError, "min_shapes must be >= 2"):
            promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), [self.shapes[0]], min_shapes=1)
        with self.assertRaisesRegex(ValueError, "even integer >= 2"):
            promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), self.shapes, pairs=1)
        with self.assertRaisesRegex(ValueError, "even integer >= 2"):
            promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), self.shapes, pairs=3)
        for bad_kw in ({"min_speedup": 0.0}, {"min_speedup": float("nan")}, {"timeout_s": -1.0}, {"atol": -0.1}):
            with self.subTest(bad_kw=bad_kw), self.assertRaises(ValueError):
                promote.run_promotion(self._cmd("base"), self._cmd("fast_exact"), self.shapes, **bad_kw)

        # CLI refuses --min-shapes 1 without --allow-single-shape with exit code 2.
        env = {**os.environ, "PYTHONPATH": LIVELAB}
        cli_ref = subprocess.run(
            [sys.executable, "-m", "rsibox", "promote", "--base-cmd", json.dumps(self._cmd("base")),
             "--candidate-cmd", json.dumps(self._cmd("fast_exact")), "--min-shapes", "1",
             "--shape", json.dumps(self.shapes[0])],
            capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(cli_ref.returncode, 2, cli_ref.stderr)

        # Explicit overrides succeed.
        ov = promote.run_promotion(
            self._cmd("base"), self._cmd("fast_exact"), [self.shapes[0]],
            pairs=1, allow_single_pair=True, allow_single_shape=True,
        )
        self.assertEqual(ov["verdict"], "promote", ov)

        # Single-pass substitution preserves literal "{arm}" inside shape names.
        tricky_shapes = [{"name": "literal-{arm}-s1", "len": 4}, {"name": "literal-{pair}-s2", "len": 8}]
        tricky = promote.run_promotion(
            [*self._cmd("base"), "{shapes_json}"], [*self._cmd("fast_exact"), "{shapes_json}"],
            tricky_shapes, pairs=2,
        )
        self.assertEqual(tricky["verdict"], "promote", tricky)
        self.assertEqual(tricky["pairs"][0]["candidate"]["shapes"][0]["shape"], "literal-{arm}-s1")

    def test_tampering_duplicate_keys_and_inconsistent_verdict_fail_verification(self):
        """Proves SHA-256 verification catches field tampering, duplicate keys, NaN, and inconsistent verdicts."""
        receipt_file = Path(self.tmp.name) / "signed.json"
        rec = promote.run_promotion(self._cmd("base"), self._cmd("slower"), self.shapes, pairs=2, receipt_path=receipt_file)
        self.assertEqual(rec["verdict"], "reject")
        self.assertTrue(promote.verify_promotion_receipt(receipt_file)["ok"])

        for label, mutate in (
            ("verdict", lambda d: d.__setitem__("verdict", "promote")),
            ("reasons", lambda d: d.__setitem__("reasons", [])),
            ("summary_speedup", lambda d: d["summary"].__setitem__("median_speedup", 9.99)),
            ("shapes", lambda d: d["shapes"].append({"name": "fake", "len": 16})),
            ("env_fingerprint", lambda d: d["env_fingerprint"].__setitem__("cpu_count", 999)),
            ("extra_field", lambda d: d.__setitem__("injected", True)),
        ):
            with self.subTest(mutation=label):
                tampered = json.loads(receipt_file.read_text(encoding="utf-8"))
                mutate(tampered)
                t_path = Path(self.tmp.name) / f"tampered_{label}.json"
                t_path.write_text(json.dumps(tampered, indent=1), encoding="utf-8")
                self.assertFalse(promote.verify_promotion_receipt(t_path)["ok"])

        # Duplicate JSON key rejected at load time.
        dup_path = Path(self.tmp.name) / "dup.json"
        dup_path.write_text('{"verdict": "reject", "verdict": "promote"}', encoding="utf-8")
        self.assertFalse(promote.verify_promotion_receipt(dup_path)["ok"])

        # Unserializable / NaN in-memory receipt handled cleanly without raising.
        bad_mem = dict(rec)
        bad_mem["summary"] = {**rec["summary"], "median_speedup": float("nan")}
        self.assertFalse(promote.verify_promotion_receipt(bad_mem)["ok"])

        # Re-signed inconsistent verdict ("promote" with non-empty reasons) is still rejected.
        inconsistent = promote._sign_receipt({**rec, "verdict": "promote"})
        inc_res = promote.verify_promotion_receipt(inconsistent)
        self.assertFalse(inc_res["ok"])
        self.assertTrue(any("inconsistent" in e for e in inc_res["errors"]))


class DoctorEnginePreflightTests(unittest.TestCase):
    def test_doctor_engine_toy_cli_succeeds_cleanly_on_cpu_and_strict_fails_visibly(self):
        """Proves `rsibox doctor --engine toy --json` exits 0 on CPU without nvidia-smi,
        unknown `--engine` exits 2, and `--strict` exits 1 with visible diagnostic output."""
        env = {**os.environ, "PYTHONPATH": LIVELAB, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run(
            [sys.executable, "-m", "rsibox", "doctor", "--engine", "toy", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["core_ok"])
        self.assertEqual(report["engine"], "toy")
        by_name = {item["name"]: item for item in report["findings"]}
        self.assertEqual(by_name["engine:toy:adapter"]["status"], "ok")
        self.assertEqual(by_name["nvidia_driver"]["status"], "ok")
        self.assertIn("skipped", by_name["nvidia_driver"]["summary"])

        # Unknown engine exits 2 cleanly.
        unk = subprocess.run(
            [sys.executable, "-m", "rsibox", "doctor", "--engine", "no_such_engine_xyz"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(unk.returncode, 2, unk.stderr)
        self.assertIn("unknown engine", unk.stderr)

        # Bare `rsibox doctor --strict` on CPU host with missing optional tools exits 1 and prints warnings.
        strict_proc = subprocess.run(
            [sys.executable, "-m", "rsibox", "doctor", "--strict"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=30,
        )
        if strict_proc.returncode == 1:
            self.assertIn("preflight issues (--strict):", strict_proc.stdout)

    def test_rlimit_nofile_low_and_healthy_thresholds_with_synthetic_probe_input(self):
        """Proves RLIMIT_NOFILE warns below 65536 with concrete remediation and passes >= 65536."""
        low = doctor.check_rlimit_nofile((1024, 4096))
        self.assertEqual(low["status"], "warn")
        self.assertIn("ulimit -n 65536", low["remediation"])
        self.assertEqual(doctor.check_rlimit_nofile((65536, 1048576))["status"], "ok")
        self.assertEqual(doctor.check_rlimit_nofile((-1, -1))["status"], "ok")

    def test_preflight_dynamic_engines_malformed_driver_and_strict_mode(self):
        """Proves dynamic engine discovery (no hardcoded engine table), malformed driver safety,
        early invalid-engine rejection before probing, and strict_ok semantics."""
        which_calls = []

        def track_which(binary: str):
            which_calls.append(binary)
            return None

        with self.assertRaisesRegex(ValueError, "unknown engine"):
            doctor.run_preflight(engine="not_real_engine", repo_root=REPO_ROOT, which_fn=track_which)
        self.assertEqual(which_calls, [], "must reject invalid engine before running tool probes")

        # Malformed driver version warns cleanly without raising ValueError.
        bad_drv = doctor.run_preflight(
            engine="vllm", repo_root=REPO_ROOT, python_version=(3, 12, 0),
            nofile_limits=(4096, 8192), driver_version="not-a-version",
            which_fn=lambda _b: None, find_spec_fn=lambda mod: mod in doctor._CORE_STDLIB,
        )
        by_bad = {item["name"]: item for item in bad_drv["findings"]}
        self.assertTrue(bad_drv["ok"])
        self.assertFalse(bad_drv["strict_ok"])
        self.assertEqual(by_bad["nvidia_driver"]["status"], "warn")
        self.assertEqual(by_bad["rlimit_nofile"]["status"], "warn")
        self.assertEqual(by_bad["engine:vllm:adapter"]["status"], "ok")
        self.assertEqual(by_bad["engine:vllm:module"]["status"], "warn")

        # Driver < 615 warns about 615+ CRIU; driver >= 615 passes strict_ok.
        drv_570 = doctor.run_preflight(
            engine="vllm", repo_root=REPO_ROOT, python_version=(3, 12, 0),
            nofile_limits=(65536, 65536), driver_version="570.124.06",
            which_fn=lambda _b: "/usr/bin/tool", find_spec_fn=lambda _m: True,
        )
        self.assertFalse(drv_570["strict_ok"])
        self.assertIn("615+", {f["name"]: f for f in drv_570["findings"]}["nvidia_driver"]["remediation"])

        drv_615 = doctor.run_preflight(
            engine="vllm", repo_root=REPO_ROOT, python_version=(3, 12, 0),
            nofile_limits=(131072, 1048576), driver_version="615.10",
            which_fn=lambda _b: "/usr/bin/tool", find_spec_fn=lambda _m: True,
        )
        self.assertTrue(drv_615["ok"] and drv_615["strict_ok"], drv_615)

        # Engine "all" discovers every engine folder under repos/ dynamically.
        all_rep = doctor.run_preflight(
            engine="all", repo_root=REPO_ROOT, probe_driver=False,
            nofile_limits=(65536, 65536), which_fn=lambda _b: "/usr/bin/tool", find_spec_fn=lambda _m: True,
        )
        self.assertIn("toy", all_rep["engines_checked"])
        self.assertIn("vllm", all_rep["engines_checked"])
        self.assertGreaterEqual(len(all_rep["engines_checked"]), 7)


if __name__ == "__main__":
    unittest.main()
