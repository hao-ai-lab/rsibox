#!/usr/bin/env python3
"""Golden cases for .github/scripts/leak-check.sh beyond raw substring matching.

Run: python3 .github/review/test_leak_check.py -v
"""
import base64
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leak-check.sh"


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class LeakCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _git(self.repo, "init", "--quiet", "-b", "main")
        _git(self.repo, "config", "user.email", "ci@example.invalid")
        _git(self.repo, "config", "user.name", "ci")
        (self.repo / "ok.txt").write_text("clean baseline\n")
        _git(self.repo, "add", "ok.txt")
        _git(self.repo, "commit", "--quiet", "-m", "base")
        self.base = _git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        self.tmp.cleanup()

    def _run_on_commit(self, rel_path, content):
        target = self.repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        _git(self.repo, "add", rel_path)
        _git(self.repo, "commit", "--quiet", "-m", "candidate")
        head = _git(self.repo, "rev-parse", "HEAD")
        return subprocess.run(
            ["bash", str(SCRIPT), self.base, head],
            cwd=self.repo, capture_output=True, text=True,
        )

    def test_clean_commit_passes(self):
        r = self._run_on_commit(
            "notes.md",
            "Runs on 127.0.0.1:8000 with task-conformance-runner-v2 and disk-cache-2026-11-21-bench.\n",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_catches_added_line_starting_with_double_plus(self):
        """[190-1]: an added line beginning with '++ ' emits '+++ ' in git diff -U0 and must
        still be scanned (for both single-line and multi-line hunks) with its file preserved."""
        slash = chr(47)
        raid_path = f"{slash}ra" + "id" + f"{slash}alice{slash}ckpt"
        gh_tok = ("gh" + "p_") + ("A" * 24)
        for idx, (rel, body) in enumerate([
            ("single_plus.txt", f"++ {raid_path} {gh_tok}\n"),
            ("multi_plus.txt", f"header\n++ {raid_path}\ntrailer\n"),
        ]):
            _git(self.repo, "checkout", "--quiet", self.base)
            _git(self.repo, "checkout", "--quiet", "-B", f"plus-{idx}")
            r = self._run_on_commit(rel, body)
            self.assertNotEqual(r.returncode, 0, f"failed to catch {rel}: {r.stdout}")
            self.assertIn("LEAK?", r.stdout, rel)
            self.assertIn(f"{rel}:", r.stdout, rel)

    def test_token_prefix_requires_boundary_and_case(self):
        """[190-2]: benign identifiers ending in 'sk-' or uppercase 'SK-' must pass, while real
        anchored tokens still fail."""
        benign = "See task-conformance-runner-v2, disk-cache-2026-11-21-bench, and SK-UPPERCASE-IDENTIFIER-2026.\n"
        r_ok = self._run_on_commit("benign.md", benign)
        self.assertEqual(r_ok.returncode, 0, r_ok.stdout + r_ok.stderr)

        _git(self.repo, "checkout", "--quiet", self.base)
        _git(self.repo, "checkout", "--quiet", "-B", "real-sk")
        sk_tok = ("s" + "k-") + ("a1b2c3d4e5" * 3)
        r_bad = self._run_on_commit("real.txt", f'api_key = "{sk_tok}"\n')
        self.assertNotEqual(r_bad.returncode, 0, r_bad.stdout)
        self.assertIn("LEAK? token or credential", r_bad.stdout)

    def test_catches_obfuscated_encoded_and_split_leaks(self):
        # Build synthetic test secrets dynamically so this test file itself stays leak-clean.
        slash = chr(47)
        raid_path = f"{slash}ra" + "id" + f"{slash}alice{slash}ckpt"
        gh_tok = ("gh" + "p_") + ("A" * 24)
        pem = "".join(["-----", "BEGIN", " RSA ", "PRIVATE KEY-----"])
        ipv6 = ":".join(["fd12", "3456", "789a", "", "1"])
        jwt = ".".join(["eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", "signaturepart12345"])
        b64_raid = base64.b64encode(raid_path.encode()).decode()
        urlsafe_raid = base64.urlsafe_b64encode((f"aa?{raid_path}???").encode()).decode()
        self.assertTrue("_" in urlsafe_raid or "-" in urlsafe_raid, urlsafe_raid)

        cases = [
            ("json_escaped.txt", raid_path.replace("/", "\\/") + "\n"),
            ("url_encoded.txt", raid_path.replace("/", "%2F") + "\n"),
            ("same_line_concat.py", f'p = "{slash}ra" + "id{slash}alice"\n'),
            ("multiline_concat.py", f't = "gh" + \\\n"p_{"B" * 24}"\n'),
            ("base64_path.txt", b64_raid + "\n"),
            ("split_base64.py", f'p = "{b64_raid[:8]}" + "{b64_raid[8:]}"\n'),
            ("urlsafe_base64.txt", urlsafe_raid + "\n"),
            ("hex_token.txt", gh_tok.encode().hex() + "\n"),
            ("hex_0x_token.txt", "0x" + gh_tok.encode().hex() + "\n"),
            ("pem_b64.txt", base64.b64encode(pem.encode()).decode() + "\n"),
            ("ipv6_ula.txt", f"host = {ipv6}\n"),
            ("jwt.txt", f"tok = {jwt}\n"),
            ("creds/prod.pem", "not empty\n"),
            ("caf\u00e9/prod.pem", "not empty\n"),
        ]
        for idx, (rel, body) in enumerate(cases):
            _git(self.repo, "checkout", "--quiet", self.base)
            _git(self.repo, "checkout", "--quiet", "-B", f"case-{idx}")
            r = self._run_on_commit(rel, body)
            self.assertNotEqual(r.returncode, 0, f"failed to catch {rel}: {r.stdout}")
            self.assertIn("LEAK?", r.stdout, rel)


if __name__ == "__main__":
    unittest.main()
