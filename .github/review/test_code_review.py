#!/usr/bin/env python3
"""Golden cases for the agent code review: what is kept, dropped, merged and posted.

Each case names the pain point it guards. Run: python3 .github/review/test_code_review.py -v
"""
import base64, json, os, socket, subprocess, sys, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

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
        """Pain point: an added line whose text starts with "++ " after a deleted "-- " line,
        read as a new file header, corrupts the path and shifts every later line number, so
        valid findings are dropped."""
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

    def test_prohibited_model_is_not_allowed(self):
        """AGENTS.md and docs/process/developer_workflow.md prohibit Gemini 3.8."""
        req = {"kind": "pr-description", "engine": "agy", "model": "gemini-3.8-flash-high", "prompt": "x"}
        self.assertIn("model", server.validate(req))
        allowed = [model for models in server.MODELS.values() for model in models]
        self.assertFalse([model for model in allowed if "3.8" in model], allowed)

    def test_short_or_empty_token_file_refuses_to_start(self):
        """Pain point: an empty token file made `Authorization: Bearer ` (empty token) valid."""
        with tempfile.TemporaryDirectory() as folder:
            token_file = Path(folder, "token")
            for text in ("", "\n", "short\n"):
                token_file.write_text(text)
                with self.assertRaises(SystemExit):
                    server.load_token(str(token_file))
            token_file.write_text("t" * 32 + "\n")
            self.assertEqual(server.load_token(str(token_file)), "t" * 32)

    def test_bad_content_length_is_a_clean_4xx(self):
        """Pain point: Content-Length -1 made the handler read the body to EOF (no size cap),
        and Content-Length abc killed the handler thread with ValueError (no response)."""
        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        srv.token, srv.bins, srv.secret_files = "t" * 32, {}, []
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            for length, code in (("-1", 400), ("abc", 400), ("", 400),
                                 (str(2 * server.MAX_PROMPT + 1), 413)):
                with socket.create_connection(srv.server_address, timeout=5) as conn:
                    conn.sendall((f"POST /v1/review HTTP/1.1\r\nHost: x\r\n"
                                  f"Authorization: Bearer {'t' * 32}\r\n"
                                  f"Content-Length: {length}\r\n\r\n{{}}").encode())
                    # A server reading to EOF would block here until the 5 s socket timeout.
                    status = conn.recv(200).decode().split("\r\n")[0]
                self.assertIn(f" {code} ", status, length)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_reply_quoting_a_secret_file_is_withheld(self):
        """Pain point: an injected 'print ~/.config/.../token' must never reach a public review,
        even when obfuscated via spacing, reversal, escaping, partial slices, base64, or hex."""
        tok = "tok_9f8e7d6c5b4a39281706f5e4"
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(tok + "\n")
        try:
            s = server.secret_strings([f.name, "/nonexistent"])
            self.assertTrue(server.leaks(f'{{"findings": [], "summary": "{tok}"}}', s))
            self.assertTrue(server.leaks(" ".join(tok), s))
            self.assertTrue(server.leaks(tok[::-1], s))
            self.assertTrue(server.leaks(base64.b64encode(tok.encode()).decode(), s))
            self.assertTrue(server.leaks(base64.b64encode(tok.encode()).decode().rstrip("="), s))
            self.assertTrue(server.leaks(tok.encode().hex(), s))
            self.assertTrue(server.leaks("\\u0074" + tok[1:], s))
            self.assertTrue(server.leaks("%74" + tok[1:], s))
            self.assertTrue(server.leaks(tok[:16], s))
            pem_header = "".join(["-----", "BEGIN", " RSA ", "PRIVATE KEY-----"])
            self.assertTrue(server.leaks(pem_header, set()))
            self.assertTrue(server.leaks(base64.b64encode(pem_header.encode()).decode(), set()))
            self.assertFalse(server.leaks('{"findings": [], "summary": "fine"}', s))
        finally:
            os.unlink(f.name)

    def test_leaks_catches_whole_file_base64_and_substring_encodings(self):
        """[190-3]: whole-file base64 (arbitrary mod-3 offsets, line-wrapped, prose-prefixed),
        URL-safe base64, and base64/hex of secret substrings must all be caught."""
        tok = "sk_live_0123456789abcdefghij"
        s = {tok}
        # Test all three mod-3 alignments of tok inside a JSON payload.
        for pad in ("", "x", "xy"):
            auth_json = json.dumps({"pad": pad, "auth_mode": "chatgpt", "tokens": {"access_token": tok}})
            b64_full = base64.b64encode(auth_json.encode()).decode()
            wrapped = "\n".join(b64_full[i:i + 76] for i in range(0, len(b64_full), 76))
            self.assertTrue(server.leaks(b64_full, s), f"raw b64 pad={pad!r}")
            self.assertTrue(server.leaks(f"dump {wrapped}", s), f"wrapped prose b64 pad={pad!r}")
            b64_url = base64.urlsafe_b64encode(auth_json.encode()).decode().rstrip("=")
            self.assertTrue(server.leaks(b64_url, s), f"urlsafe b64 pad={pad!r}")
            self.assertTrue(server.leaks(auth_json.encode().hex(), s), f"whole-file hex pad={pad!r}")
        self.assertTrue(server.leaks(base64.b64encode(tok[1:].encode()).decode(), s))
        self.assertTrue(server.leaks(base64.b64encode(tok[:-1].encode()).decode().rstrip("="), s))
        self.assertTrue(server.leaks(tok[1:].encode().hex(), s))
        self.assertTrue(server.leaks(tok[1:].encode().hex().upper(), s))
        self.assertTrue(server.leaks(base64.b64encode(" ".join(tok).encode()).decode(), s))
        self.assertTrue(server.leaks(base64.b64encode(tok[::-1].encode()).decode(), s))
        self.assertFalse(server.leaks(base64.b64encode(b'{"findings": [], "summary": "clean"}').decode(), s))


class JobIsolation(unittest.TestCase):
    """Pain points: a reviewer that sees its whole real state directory (other conversations,
    settings), and an engine's error text that bypasses the secret filter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.codex_home = Path(self.tmp.name, "codex-home")
        self.codex_home.mkdir()
        self.secret = "sk_live_0123456789abcdefghij"
        Path(self.codex_home, "auth.json").write_text(json.dumps(
            {"auth_mode": "chatgpt", "tokens": {"account_id": "acct-1", "access_token": self.secret}}))
        Path(self.codex_home, "history.jsonl").write_text("private past conversation\n")
        self.old = os.environ.get("CODEX_HOME")
        os.environ["CODEX_HOME"] = str(self.codex_home)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = self.old
        self.tmp.cleanup()

    def auth(self, account, access):
        return json.dumps({"auth_mode": "chatgpt", "tokens": {"account_id": account,
                                                             "access_token": access}})

    def job(self):
        work = Path(tempfile.mkdtemp(dir=self.tmp.name))
        env, pairs = server.job_home("codex", str(work))
        return Path(env["CODEX_HOME"], "auth.json"), pairs

    def test_job_home_holds_only_credential_copies(self):
        copy, _ = self.job()
        self.assertTrue(str(copy).startswith(self.tmp.name))
        self.assertEqual(sorted(p.name for p in copy.parent.iterdir()), ["auth.json"])  # no history

    def test_copy_back_keeps_only_a_refresh_of_the_same_account(self):
        """Pain point: a prompt-injected reviewer can write its credential copy. Copying that
        back unchecked would install garbage, or another account's credentials, as the host's
        login."""
        real = self.codex_home / "auth.json"
        cases = [(self.auth("acct-1", "new-token"), True),       # genuine refresh: kept
                 (self.auth("attacker", "x"), False),            # another account: refused
                 ("not json", False)]                            # garbage: refused
        for planted, kept in cases:
            real.write_text(self.auth("acct-1", "old-token"))
            copy, pairs = self.job()
            copy.write_text(planted)
            server.copy_back("codex", pairs)
            self.assertEqual(real.read_text() == planted, kept, planted)

    def test_copy_back_never_overwrites_a_newer_host_credential(self):
        """Pain point: two jobs refresh concurrently; the later copy-back must not clobber the
        credential another job (or the user) already put in place."""
        real = self.codex_home / "auth.json"
        real.write_text(self.auth("acct-1", "old"))
        copy, pairs = self.job()
        real.write_text(self.auth("acct-1", "newer-from-elsewhere"))
        copy.write_text(self.auth("acct-1", "stale-refresh"))
        server.copy_back("codex", pairs)
        self.assertIn("newer-from-elsewhere", real.read_text())

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
        if "Landlock is not available" in job["error"]:
            self.skipTest("kernel without Landlock")
        self.assertIn("withheld", job["error"])

    def test_command_restricts_binary_folder_run_and_shared_shm(self):
        """M-M20: confine grants the binary file (and any shebang interpreter file) rather than
        recursively granting its parent folder, excludes /run recursively, and uses --private-shm
        instead of sharing host /dev/shm."""
        bin_dir = Path(self.tmp.name, "tools")
        bin_dir.mkdir()
        interp = Path(self.tmp.name, "interp", "custom-py")
        interp.parent.mkdir()
        interp.write_text("#!/bin/sh\n")
        exe = bin_dir / "fake-codex"
        exe.write_text(f"#!{interp}\necho ok\n")
        work = Path(self.tmp.name, "job")
        work.mkdir()
        with mock.patch.dict(os.environ, {"RSIBOX_REVIEW_EXTRA_READ": ""}):
            argv, _ = server.command("codex", "hi", "default", str(work), {"codex": str(exe)})
        r_idx = argv.index("--read")
        w_idx = argv.index("--write")
        sep_idx = argv.index("--")
        read_paths = argv[r_idx + 1:w_idx]
        write_paths = argv[w_idx + 1:sep_idx]
        self.assertIn("--private-shm", argv[:r_idx])
        self.assertNotIn("/run", server.SYSTEM_READ)
        self.assertNotIn("/run", read_paths)
        self.assertNotIn(str(bin_dir), read_paths)
        self.assertNotIn(str(interp.parent), read_paths)
        self.assertIn(os.path.realpath(exe), read_paths)
        self.assertIn(os.path.realpath(interp), read_paths)
        self.assertNotIn("/dev/shm", write_paths)

    def test_private_shm_control_flow_never_grants_global_dev_shm(self):
        """M-M20: when setup_private_shm fails (e.g. restricted userns), --private-shm must
        NOT pass /dev/shm to confine(); --require-private-shm fails closed."""
        confine_mod = __import__("confine")
        seen = []
        with mock.patch.object(confine_mod, "setup_private_shm", return_value=None), \
                mock.patch.object(confine_mod, "confine", lambda r, w: seen.append((r, w))), \
                mock.patch.object(confine_mod.os, "execvp", lambda *_a: None):
            confine_mod.main(["--private-shm", "--read", "/usr", "--write", "/tmp/job", "--", "true"])
            self.assertEqual(seen, [(["/usr"], ["/tmp/job"])])
            with self.assertRaises(OSError):
                confine_mod.main(["--require-private-shm", "--read", "/usr", "--write", "/tmp/job", "--", "true"])
        seen.clear()
        with mock.patch.object(confine_mod, "setup_private_shm", return_value="/dev/shm"), \
                mock.patch.object(confine_mod, "confine", lambda r, w: seen.append((r, w))), \
                mock.patch.object(confine_mod.os, "execvp", lambda *_a: None):
            confine_mod.main(["--private-shm", "--read", "/usr", "--write", "/tmp/job", "--", "true"])
            self.assertEqual(seen, [(["/usr"], ["/tmp/job", "/dev/shm"])])

    def test_setup_private_shm_falls_back_on_enospc_and_eusers(self):
        """[190-4]: hosts with user.max_user_namespaces=0 return ENOSPC (or EUSERS) from
        unshare(CLONE_NEWUSER); setup_private_shm must return None rather than crashing."""
        import errno
        confine_mod = __import__("confine")
        with tempfile.TemporaryDirectory() as shm_dir:
            fake_libc = mock.MagicMock()
            fake_libc.unshare.return_value = -1
            fake_libc.mount.return_value = 0
            for err in (errno.ENOSPC, getattr(errno, "EUSERS", errno.ENOSPC)):
                with mock.patch.object(confine_mod, "libc", fake_libc), \
                        mock.patch.object(confine_mod.ctypes, "get_errno", return_value=err):
                    self.assertIsNone(confine_mod.setup_private_shm(shm_dir))
            with mock.patch.object(confine_mod, "libc", fake_libc), \
                    mock.patch.object(confine_mod.ctypes, "get_errno", return_value=errno.ENOMEM):
                with self.assertRaises(OSError):
                    confine_mod.setup_private_shm(shm_dir)


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

    def test_confined_binary_cannot_read_sibling_in_its_folder_or_run(self):
        """M-M20: granting the executable file lets it run while denying sibling files and
        subdirectories in the executable's directory, as well as /run."""
        with tempfile.TemporaryDirectory() as folder:
            bin_dir = Path(folder, "bin")
            sub = bin_dir / "nested"
            sub.mkdir(parents=True)
            exe = bin_dir / "tool.sh"
            exe.write_text("#!/bin/sh\necho ran\n")
            exe.chmod(0o755)
            (bin_dir / "sibling.txt").write_text("sibling-secret")
            (sub / "deep.txt").write_text("deep-secret")
            work = Path(folder, "work")
            work.mkdir()
            read_paths = server.system_read_paths() + server.binary_read_paths(str(exe))
            base = [sys.executable, str(Path(server.HERE, "confine.py")),
                    "--private-shm", "--read", *read_paths,
                    "--write", str(work), "/dev/null", "--"]
            r_exe = subprocess.run(base + [str(exe)], capture_output=True, text=True)
            if "not available" in r_exe.stderr:
                self.skipTest("kernel without Landlock")
            self.assertEqual((r_exe.returncode, r_exe.stdout.strip()), (0, "ran"))
            for blocked in (str(bin_dir / "sibling.txt"), str(sub / "deep.txt"), "/run"):
                r_blk = subprocess.run(base + ["ls" if blocked == "/run" else "cat", blocked],
                                       capture_output=True, text=True)
                self.assertNotEqual(r_blk.returncode, 0, blocked)
                self.assertIn("Permission denied", r_blk.stderr, blocked)

    def test_private_shm_never_exposes_host_dev_shm(self):
        """M-M20: with --private-shm, a confined process sees either its own empty /dev/shm
        namespace (when unprivileged user namespaces work) or Permission denied (when disabled),
        and never sees files created on the host /dev/shm."""
        if not os.path.isdir("/dev/shm"):
            self.skipTest("/dev/shm not present")
        with tempfile.NamedTemporaryFile(dir="/dev/shm", prefix="rsibox-host-", delete=False) as f:
            f.write(b"host-shm-secret")
            host_file = f.name
        try:
            with tempfile.TemporaryDirectory() as work:
                probe = ("import os, sys; p = sys.argv[1]; "
                         "print('HOST_LEAK' if os.path.exists(p) else 'ISOLATED')")
                base = [sys.executable, str(Path(server.HERE, "confine.py")),
                        "--private-shm", "--read", *server.system_read_paths(),
                        "--write", work, "/dev/null", "--",
                        sys.executable, "-c", probe, host_file]
                r = subprocess.run(base, capture_output=True, text=True)
                if "not available" in r.stderr:
                    self.skipTest("kernel without Landlock")
                self.assertNotIn("HOST_LEAK", r.stdout)
        finally:
            os.unlink(host_file)


if __name__ == "__main__":
    unittest.main()
