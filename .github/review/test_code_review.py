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
@@ -10,4 +10,5 @@ def f():
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


class Replies(unittest.TestCase):
    def test_bad_json_is_rejected_not_posted(self):
        """A reviewer that answers in prose (or broken JSON) is an error for that engine, never
        an empty 'no findings' review."""
        for reply in ("Looks good to me!", '{"findings": [', '{"summary": "ok"}'):
            with self.assertRaises(ValueError, msg=reply):
                cr.parse_reply(reply)

    def test_malformed_findings_are_dropped_and_counted(self):
        reply = json.dumps({"findings": [finding(), finding(severity="critical"),
                                         finding(line="11"), finding(claim=" "), "nope"],
                            "summary": "s"})
        kept, summary, dropped = cr.parse_reply("Here you go:\n" + reply + "\nthanks")
        self.assertEqual(([f["severity"] for f in kept], summary, dropped), (["high"], "s", 4))

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
