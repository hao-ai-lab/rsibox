"""Tests for `site/build.py` verifying `--check` does not create or modify `site/public/`."""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_PY = os.path.join(REPO_ROOT, "site", "build.py")


def _load_site_build():
    spec = importlib.util.spec_from_file_location("site_build", BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SiteBuildTests(unittest.TestCase):
    def test_check_flag_validates_without_creating_pub(self):
        site_build = _load_site_build()
        with tempfile.TemporaryDirectory() as tmp:
            pub_dir = os.path.join(tmp, "public")
            buf = io.StringIO()
            with (
                mock.patch.object(site_build, "PUB", pub_dir),
                contextlib.redirect_stdout(buf),
            ):
                rc = site_build.main(["--check"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(pub_dir))
            self.assertIn("checked index.html ", buf.getvalue())

    def test_write_mode_writes_ascii_index_html(self):
        site_build = _load_site_build()
        with tempfile.TemporaryDirectory() as tmp:
            pub_dir = os.path.join(tmp, "public")
            audit_dir = os.path.join(pub_dir, "audit")
            os.makedirs(audit_dir, exist_ok=True)
            with open(os.path.join(audit_dir, "stale.txt"), "w", encoding="utf-8") as fh:
                fh.write("stale")

            buf = io.StringIO()
            with (
                mock.patch.object(site_build, "PUB", pub_dir),
                contextlib.redirect_stdout(buf),
            ):
                rc = site_build.main([])

            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(audit_dir))
            out_html = os.path.join(pub_dir, "index.html")
            self.assertTrue(os.path.isfile(out_html))
            with open(out_html, encoding="utf-8") as fh:
                html = fh.read()
            self.assertTrue(html.isascii())
            self.assertIn("<title>rsibox</title>", html)

    def test_subprocess_check_leaves_site_public_untouched(self):
        real_pub = os.path.join(REPO_ROOT, "site", "public")
        existed_before = os.path.exists(real_pub)
        before_mtime = os.path.getmtime(real_pub) if existed_before else None

        proc = subprocess.run(
            [sys.executable, BUILD_PY, "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("checked index.html ", proc.stdout)
        if not existed_before:
            self.assertFalse(os.path.exists(real_pub))
        else:
            self.assertEqual(os.path.getmtime(real_pub), before_mtime)


if __name__ == "__main__":
    unittest.main()
