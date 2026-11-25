"""Golden end-to-end tests for `rsibox verify-trace` CLI command."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from kernel import toy_lab
from rsibox import PoolSlot, SandboxPool, cli

LIVELAB = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class VerifyTraceCliTests(unittest.TestCase):
    def _write_real_pool_trace(self, trace_path: str, *, leases: int = 3) -> tuple[int, str]:
        lab = toy_lab.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        pool = SandboxPool([PoolSlot("s0", lab)], trace_path=trace_path)
        self.addCleanup(pool.close)
        for idx in range(leases):
            res = pool.run_isolated(f"agent-{idx}", lambda plab, _l: plab.check(), timeout_s=5.0)
            self.assertTrue(res["ok"], res)
        head = pool.metrics()["trace_head"]
        return int(head[0]), str(head[1])

    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_verify_trace_cli_valid_expected_head_truncation_tamper_and_batch(self):
        """Golden case: real SandboxPool trace passes human & --json verification, catches
        tail truncation via --expected-count/--expected-hash, catches mid-file tampering with
        line numbers, handles empty/missing files cleanly, and aggregates multi-file batches."""
        with tempfile.TemporaryDirectory(prefix="rsibox-vt-cli-") as tmp:
            good_a = os.path.join(tmp, "good_a.jsonl")
            good_b = os.path.join(tmp, "good_b.jsonl")
            empty_f = os.path.join(tmp, "empty.jsonl")
            missing_f = os.path.join(tmp, "missing.jsonl")
            tampered_f = os.path.join(tmp, "tampered.jsonl")
            truncated_f = os.path.join(tmp, "truncated.jsonl")

            count_a, hash_a = self._write_real_pool_trace(good_a, leases=3)
            count_b, _hash_b = self._write_real_pool_trace(good_b, leases=2)
            with open(empty_f, "w", encoding="utf-8"):
                pass

            # 1. Valid single file (human + --json + matching expected head)
            rc, out, err = self._run_cli(["verify-trace", good_a])
            self.assertEqual(rc, 0, err)
            self.assertIn("OK", out)
            self.assertIn("3 records", out)
            self.assertIn(hash_a[:12], out)

            rc, out, _ = self._run_cli([
                "verify-trace", good_a,
                "--expected-count", str(count_a),
                "--expected-hash", hash_a,
                "--json",
            ])
            self.assertEqual(rc, 0, out)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual((payload["checked"], payload["passed"], payload["failed"], payload["total_records"]), (1, 1, 0, 3))
            self.assertEqual(payload["results"][0]["head"], [count_a, hash_a])

            # 2. Empty file passes with genesis head (and with --expected-count 0 --expected-hash genesis)
            rc, out, _ = self._run_cli(["verify-trace", empty_f])
            self.assertEqual(rc, 0)
            self.assertIn("0:genesis", out)
            rc, out, _ = self._run_cli(["verify-trace", empty_f, "--expected-count", "0", "--expected-hash", "genesis", "--json"])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(out)["ok"])
            rc, out, _ = self._run_cli(["verify-trace", empty_f, "--expected-count", "1", "--expected-hash", hash_a, "--json"])
            self.assertEqual(rc, 1)
            self.assertFalse(json.loads(out)["ok"])
            # Uppercase 64-hex digest is normalized to lowercase cleanly
            rc, out, _ = self._run_cli([
                "verify-trace", good_a,
                "--expected-count", str(count_a),
                "--expected-hash", hash_a.upper(),
                "--json",
            ])
            self.assertEqual(rc, 0, out)
            self.assertTrue(json.loads(out)["ok"])

            # 3. Tail truncation caught via --expected-count/--expected-hash
            with open(good_a, encoding="utf-8") as handle:
                raw_lines = handle.readlines()
            with open(truncated_f, "w", encoding="utf-8") as handle:
                handle.writelines(raw_lines[:-1])
            rc, out, _ = self._run_cli([
                "verify-trace", truncated_f,
                "--expected-count", str(count_a),
                "--expected-hash", hash_a,
                "--json",
            ])
            self.assertEqual(rc, 1)
            trunc_payload = json.loads(out)
            self.assertFalse(trunc_payload["ok"])
            self.assertIn("head mismatch", trunc_payload["results"][0]["error"])

            # 4. Mid-file tampering caught with exact line number
            tampered_lines = list(raw_lines)
            rec2 = json.loads(tampered_lines[1])
            rec2["agent_id"] = "evil-agent"
            tampered_lines[1] = json.dumps(rec2, sort_keys=True) + "\n"
            with open(tampered_f, "w", encoding="utf-8") as handle:
                handle.writelines(tampered_lines)
            rc, out, _ = self._run_cli(["verify-trace", tampered_f])
            self.assertEqual(rc, 1)
            self.assertIn("FAIL", out)
            self.assertIn("line 2:", out)
            self.assertIn("digest mismatch", out)

            # 5. Missing file, directory path, and invalid UTF-8 binary file fail cleanly with exit 1 (no traceback)
            rc, out, _ = self._run_cli(["verify-trace", missing_f, "--json"])
            self.assertEqual(rc, 1)
            miss_payload = json.loads(out)
            self.assertFalse(miss_payload["ok"])
            self.assertIn("FileNotFoundError", miss_payload["results"][0]["error"])

            rc, out, _ = self._run_cli(["verify-trace", tmp, "--json"])
            self.assertEqual(rc, 1)
            dir_payload = json.loads(out)
            self.assertFalse(dir_payload["ok"])
            self.assertIsNone(dir_payload["results"][0]["head"])
            self.assertIsNone(dir_payload["results"][0]["line"])
            self.assertIn("IsADirectoryError", dir_payload["results"][0]["error"])

            bad_utf8 = os.path.join(tmp, "bad_utf8.jsonl")
            with open(bad_utf8, "wb") as handle:
                handle.write(b"\xff\xfe\x00\x80\n")
            rc, out, _ = self._run_cli(["verify-trace", bad_utf8, "--json"])
            self.assertEqual(rc, 1)
            utf8_payload = json.loads(out)
            self.assertFalse(utf8_payload["ok"])
            self.assertIsNone(utf8_payload["results"][0]["head"])
            self.assertIn("UnicodeDecodeError", utf8_payload["results"][0]["error"])

            # 6. Multi-file batch (including duplicate path arguments) checks every entry cleanly
            rc, out, _ = self._run_cli(["verify-trace", good_a, good_a, "--json"])
            self.assertEqual(rc, 0)
            dup_payload = json.loads(out)
            self.assertTrue(dup_payload["ok"])
            self.assertEqual((dup_payload["checked"], dup_payload["total_records"]), (2, count_a * 2))

            rc, out, _ = self._run_cli(["verify-trace", good_a, tampered_f, good_b])
            self.assertEqual(rc, 1)
            self.assertIn("Summary: 2/3 traces OK (6 total records)", out)
            rc, out, _ = self._run_cli(["verify-trace", good_a, good_b, "--json"])
            self.assertEqual(rc, 0)
            batch_ok = json.loads(out)
            self.assertTrue(batch_ok["ok"])
            self.assertEqual((batch_ok["checked"], batch_ok["passed"], batch_ok["failed"], batch_ok["total_records"]), (2, 2, 0, count_a + count_b))

            # 7. Real subprocess invocation `python3 -m rsibox verify-trace`
            env = {**os.environ, "PYTHONPATH": LIVELAB}
            proc = subprocess.run(
                [sys.executable, "-m", "rsibox", "verify-trace", good_a, good_b, "--json"],
                capture_output=True, text=True, env=env, timeout=15, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            sub_payload = json.loads(proc.stdout)
            self.assertTrue(sub_payload["ok"])
            self.assertEqual(sub_payload["total_records"], 5)

    def test_verify_trace_cli_argument_validation(self):
        """Invalid combinations of --expected-count / --expected-hash exit with argparse code 2, and --help works."""
        valid_hex = "a" * 64
        bad_invocations = [
            ["verify-trace", "a.jsonl", "--expected-count", "3"],
            ["verify-trace", "a.jsonl", "--expected-hash", valid_hex],
            ["verify-trace", "a.jsonl", "b.jsonl", "--expected-count", "3", "--expected-hash", valid_hex],
            ["verify-trace", "a.jsonl", "--expected-count", "-1", "--expected-hash", valid_hex],
            ["verify-trace", "a.jsonl", "--expected-count", "1", "--expected-hash", "   "],
            ["verify-trace", "a.jsonl", "--expected-count", "1", "--expected-hash", "not-hex!"],
            ["verify-trace", "a.jsonl", "--expected-count", "1", "--expected-hash", "abcd"],
            ["verify-trace", "a.jsonl", "--expected-count", "0", "--expected-hash", valid_hex],
            ["verify-trace", "a.jsonl", "--expected-count", "1", "--expected-hash", "genesis"],
        ]
        for argv in bad_invocations:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(argv)
            self.assertEqual(ctx.exception.code, 2, argv)
            self.assertIn("rsibox verify-trace", err.getvalue(), argv)

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["verify-trace", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--expected-count", out.getvalue())
        self.assertIn("--expected-hash", out.getvalue())


if __name__ == "__main__":
    unittest.main()
