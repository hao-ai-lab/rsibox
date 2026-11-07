#!/usr/bin/env python3
"""Golden cases for the agent code review: what is kept, dropped, merged and posted.

Each case names the pain point it guards. Run: python3 .github/review/test_code_review.py -v
"""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_review as cr          # noqa: E402
import server                     # noqa: E402

DIFF = """diff --git a/pkg/mod.py b/pkg/mod.py
index 111..222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,3 +10,4 @@ def f():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
"""


def finding(**kw):
    f = {"file": "pkg/mod.py", "line": 11, "severity": "high", "claim": "b is wrong",
         "evidence": "b = 3", "suggestion": "use 2"}
    f.update(kw)
    return f


class Diff(unittest.TestCase):
    def test_new_side_lines_count_context_and_additions_not_deletions(self):
        """New-side numbering: context 10, then +b at 11, +c at 12, context 13. A deleted file
        has no new side. Pain point: comments on a wrong line make GitHub reject the review."""
        self.assertEqual(cr.diff_lines(DIFF), {"pkg/mod.py": {10, 11, 12, 13}})

    def test_truncation_cuts_at_a_line_and_says_so(self):
        d, cut = cr.cap_diff("a\nbb\nccc\n", limit=6)
        self.assertEqual((d, cut), ("a\nbb\n", True))
        self.assertEqual(cr.cap_diff("short\n"), ("short\n", False))


    def test_hunk_lines_that_look_like_headers_stay_content(self):
        """Pain point (found by the agent review of this PR): an added line whose text starts
        with "++ " after a deleted "-- " line was read as a new file header, which corrupted the
        path and shifted every later line number, so valid findings were dropped."""
        diff = ("diff --git a/x.md b/x.md\n--- a/x.md\n+++ b/x.md\n@@ -1,2 +1,2 @@\n"
                "--- old rule\n+++ new rule\n same\n")
        self.assertEqual(cr.diff_lines(diff), {"x.md": {1, 2}})


class Replies(unittest.TestCase):
    def test_bad_json_is_rejected_not_posted(self):
        """A reviewer that answers in prose (or broken JSON) is an error for that engine, never
        an empty 'no findings' review."""
        for reply in ("Looks good to me!", '{"findings": [', '{"summary": "ok"}'):
            with self.assertRaises(ValueError, msg=reply):
                cr.parse_reply(reply)

    def test_malformed_findings_are_dropped_and_counted(self):
        """Includes a finding without evidence: the prompt says "no evidence, no finding", so the
        checker enforces it rather than trusting the reviewer."""
        reply = json.dumps({"findings": [finding(), finding(severity="critical"),
                                         finding(line="11"), finding(claim=" "),
                                         finding(evidence=""), "nope"],
                            "summary": "s"})
        kept, summary, dropped = cr.parse_reply("Here you go:\n" + reply + "\nthanks")
        self.assertEqual(([f["severity"] for f in kept], summary, dropped), (["high"], "s", 5))

    def test_finding_outside_the_diff_is_dropped(self):
        """A finding on a line the diff does not show (or a file it does not touch) cannot be
        anchored and would be a guess: dropped, and counted in the review body."""
        lines = cr.diff_lines(DIFF)
        kept, out = cr.in_diff([finding(line=11), finding(line=40), finding(file="other.py")], lines)
        self.assertEqual(([f["line"] for f in kept], out), ([11], 2))


class Merge(unittest.TestCase):
    def test_two_engines_on_one_line_become_one_comment(self):
        """agy and codex both flag line 11 with the same claim (different wording of case and
        punctuation) and codex adds a second claim: one comment, both engines, highest severity,
        each distinct claim once."""
        merged = cr.merge({
            "agy": [finding(severity="medium", claim="B is wrong.")],
            "codex": [finding(severity="high", claim="b is wrong"),
                      finding(severity="low", claim="c is unused")]})
        self.assertEqual(len(merged), 1)
        g = merged[0]
        self.assertEqual((g["engines"], g["severity"], [i["claim"] for i in g["items"]]),
                         (["agy", "codex"], "high", ["B is wrong.", "c is unused"]))
        self.assertIn("flagged by agy, codex", cr.comment_body(g))


class Posting(unittest.TestCase):
    def test_review_is_always_a_comment_never_an_approval(self):
        """Even when a reviewer's summary says 'approve', the payload's event is COMMENT."""
        report = {"agy": {"seconds": 3, "kept": 0, "dropped": 0, "summary": "LGTM, approve"}}
        r = cr.build_review("abc123", [], report, truncated=False)
        self.assertEqual((r["event"], r["comments"]), ("COMMENT", []))
        self.assertIn("No findings on the diff.", r["body"])

    def test_post_refuses_any_other_event(self):
        for event in ("APPROVE", "REQUEST_CHANGES", None):
            with self.assertRaises(ValueError):
                cr.post("o/r", 1, {"event": event, "body": "x", "comments": []})

    def test_skipped_engine_and_truncation_are_stated(self):
        report = {"agy": {"error": "TimeoutError: no reply in 900 s", "seconds": 900},
                  "codex": {"seconds": 40, "kept": 1, "dropped": 2, "summary": ""}}
        body = cr.build_review("abc", [], report, truncated=True)["body"]
        self.assertIn("**agy**: skipped, TimeoutError", body)
        self.assertIn("2 dropped", body)
        self.assertIn("later files were not reviewed", body)


class Service(unittest.TestCase):
    def test_request_validation(self):
        ok = {"kind": "code-review", "engine": "codex", "model": "default", "prompt": "x"}
        self.assertIsNone(server.validate(dict(ok)))
        old_client = {"kind": "pr-description", "model": "gemini-4-argon-high", "prompt": "x"}
        self.assertIsNone(server.validate(old_client))          # engine defaults to agy
        self.assertEqual(old_client["engine"], "agy")
        self.assertIn("engine", server.validate(dict(ok, kind="pr-description")))  # codex: code review only
        self.assertIn("model", server.validate(dict(ok, model="gpt-x")))
        self.assertIn("prompt", server.validate(dict(ok, prompt="y" * (server.MAX_PROMPT + 1))))

    def test_reply_quoting_a_secret_file_is_withheld(self):
        """Pain point: an injected 'print ~/.config/.../token' must never reach a public review."""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("tok_9f8e7d6c5b4a39281706f5e4\n")
        try:
            s = server.secret_strings([f.name, "/nonexistent"])
            self.assertTrue(server.leaks('{"findings": [], "summary": "tok_9f8e7d6c5b4a39281706f5e4"}', s))
            self.assertFalse(server.leaks('{"findings": [], "summary": "fine"}', s))
        finally:
            os.unlink(f.name)


class JobIsolation(unittest.TestCase):
    """Pain points found by the agent review of this PR: the reviewer could see its whole real
    state directory (other conversations, settings), and an engine's error text bypassed the
    secret filter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.codex_home = Path(self.tmp.name, "codex-home")
        self.codex_home.mkdir()
        self.secret = "sk_live_0123456789abcdefghij"
        Path(self.codex_home, "auth.json").write_text(json.dumps({"token": self.secret}))
        Path(self.codex_home, "history.jsonl").write_text("private past conversation\n")
        self.old = os.environ.get("CODEX_HOME")
        os.environ["CODEX_HOME"] = str(self.codex_home)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = self.old
        self.tmp.cleanup()

    def test_job_home_holds_only_credential_copies_and_keeps_a_refresh(self):
        work = Path(self.tmp.name, "job")
        work.mkdir()
        env, pairs = server.job_home("codex", str(work))
        state = Path(env["CODEX_HOME"])
        self.assertTrue(str(state).startswith(str(work)))
        self.assertEqual(sorted(p.name for p in state.iterdir()), ["auth.json"])   # no history
        Path(state, "auth.json").write_text('{"token": "refreshed"}')
        server.copy_back(pairs)
        self.assertEqual(json.loads(Path(self.codex_home, "auth.json").read_text())["token"], "refreshed")

    @unittest.skipUnless(sys.platform == "linux", "Landlock is Linux-only")
    def test_secret_in_an_engine_error_is_withheld(self):
        fake = Path(self.tmp.name, "bin", "fake-codex")
        fake.parent.mkdir()
        fake.write_text(f"#!/bin/sh\necho 'auth failed for {self.secret}' >&2\nexit 1\n")
        fake.chmod(0o755)
        server.JOBS["0" * 16] = {"status": "queued", "created": 0, "repo": "o/r", "pr": "1"}
        server.run_review("0" * 16, {"kind": "code-review", "engine": "codex", "model": "default",
                                     "prompt": "x"}, {"codex": str(fake)},
                          [str(self.codex_home / "auth.json")])
        job = server.JOBS.pop("0" * 16)
        self.assertEqual(job["status"], "error")
        self.assertNotIn(self.secret, job["error"])
        self.assertIn("withheld", job["error"])


@unittest.skipUnless(sys.platform == "linux", "Landlock is Linux-only")
class Confine(unittest.TestCase):
    def test_confined_process_cannot_read_outside_its_paths(self):
        """The guarantee the service relies on: a confined child reads its own directory but
        gets 'Permission denied' for a file outside every allowed path."""
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as other:
            Path(allowed, "ok.txt").write_text("fine")
            Path(other, "secret.txt").write_text("hidden")
            base = [sys.executable, str(Path(server.HERE, "confine.py")), "--read",
                    *server.SYSTEM_READ, "--write", allowed, "/dev/null", "--", "cat"]
            r_ok = subprocess.run(base + [f"{allowed}/ok.txt"], capture_output=True, text=True)
            r_no = subprocess.run(base + [f"{other}/secret.txt"], capture_output=True, text=True)
            if "not available" in r_ok.stderr:
                self.skipTest("kernel without Landlock")
            self.assertEqual((r_ok.returncode, r_ok.stdout), (0, "fine"))
            self.assertNotEqual(r_no.returncode, 0)
            self.assertIn("Permission denied", r_no.stderr)
            self.assertNotIn("hidden", r_no.stdout)


if __name__ == "__main__":
    unittest.main()
