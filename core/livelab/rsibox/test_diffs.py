"""Tests for `rsibox.diffs` against real `git diff` patch files and edge-case headers.

Run from the repo root (pure CPU, read-only on `repos/*/patches`):
    python3 -W error::ResourceWarning -m unittest discover -s core/livelab/rsibox -t core/livelab -p "test_diffs.py" -v
"""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from rsibox import diffs

REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_PATCH_DIR = REPO_ROOT / "repos" / "megatron" / "patches"
FASTVIDEO_PATCH_DIR = REPO_ROOT / "repos" / "fastvideo" / "patches" / "h3-hillclimb"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _synthesize_base_and_expected(diff_text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build minimal synthetic `{module: base}` and `{module: expected}` texts matching every hunk."""
    file_hunks: dict[str, list[tuple[int, list[str], list[str], bool, bool]]] = {}
    cur_path: str | None = None
    lines = diff_text.splitlines(keepends=True)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("+++ "):
            raw = line[4:].split("\t", 1)[0].strip()
            path = raw[2:] if raw.startswith("b/") else raw
            cur_path = None if path == "/dev/null" else path
            if cur_path is not None:
                file_hunks.setdefault(cur_path, [])
            idx += 1
            continue
        m = _HUNK.match(line)
        if m and cur_path is not None:
            old_start = int(m.group(1))
            old_rem = int(m.group(2)) if m.group(2) is not None else 1
            new_rem = int(m.group(4)) if m.group(4) is not None else 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            old_no_nl = new_no_nl = False
            last_tag: str | None = None
            idx += 1
            while idx < len(lines) and (old_rem > 0 or new_rem > 0):
                hl = lines[idx]
                if hl.startswith("\\"):
                    if last_tag in (" ", "-"):
                        old_no_nl = True
                    if last_tag in (" ", "+"):
                        new_no_nl = True
                elif hl.startswith(" "):
                    old_lines.append(hl[1:])
                    new_lines.append(hl[1:])
                    old_rem -= 1
                    new_rem -= 1
                    last_tag = " "
                elif hl.startswith("-"):
                    old_lines.append(hl[1:])
                    old_rem -= 1
                    last_tag = "-"
                elif hl.startswith("+"):
                    new_lines.append(hl[1:])
                    new_rem -= 1
                    last_tag = "+"
                elif hl in ("\n", "\r\n", ""):
                    old_lines.append(hl)
                    new_lines.append(hl)
                    old_rem -= 1
                    new_rem -= 1
                    last_tag = " "
                else:
                    break
                idx += 1
            if idx < len(lines) and lines[idx].startswith("\\"):
                if last_tag in (" ", "-"):
                    old_no_nl = True
                if last_tag in (" ", "+"):
                    new_no_nl = True
                idx += 1
            file_hunks[cur_path].append((old_start, old_lines, new_lines, old_no_nl, new_no_nl))
            continue
        idx += 1

    base: dict[str, str] = {}
    expected: dict[str, str] = {}
    for rel_path, hunks in file_hunks.items():
        mod = diffs.module_of(rel_path)
        old_buf: list[str] = []
        new_buf: list[str] = []
        for old_start, old_lines, new_lines, old_no_nl, new_no_nl in hunks:
            while len(old_buf) < old_start - 1:
                filler = f"# filler line {len(old_buf) + 1}\n"
                old_buf.append(filler)
                new_buf.append(filler)
            old_buf.extend(old_lines)
            new_buf.extend(new_lines)
            if old_no_nl and old_buf:
                old_buf[-1] = old_buf[-1].rstrip("\r\n")
            if new_no_nl and new_buf:
                new_buf[-1] = new_buf[-1].rstrip("\r\n")
        base[mod] = "".join(old_buf)
        expected[mod] = "".join(new_buf)
    return base, expected


class RealGitDiffCorpusTests(unittest.TestCase):
    """Read-only coverage of real `git diff` patch files across Megatron and FastVideo."""

    def test_real_megatron_and_fastvideo_git_diff_patches(self):
        megatron_patches = sorted(MEGATRON_PATCH_DIR.glob("*.patch"))
        fastvideo_patches = sorted(FASTVIDEO_PATCH_DIR.glob("*.patch"))
        self.assertGreaterEqual(len(megatron_patches), 4, megatron_patches)
        self.assertGreaterEqual(len(fastvideo_patches), 20, fastvideo_patches)

        corpus = [
            ("megatron.", megatron_patches),
            ("fastvideo.", fastvideo_patches),
        ]
        saw_multi_file = False
        saw_multi_hunk = False
        for mod_prefix, patch_files in corpus:
            for patch_path in patch_files:
                with self.subTest(patch=patch_path.relative_to(REPO_ROOT).as_posix()):
                    diff_text = patch_path.read_text(encoding="utf-8")
                    rel_paths = diffs.paths(diff_text)
                    self.assertTrue(rel_paths, f"no target paths parsed for {patch_path.name}")
                    self.assertEqual(len(rel_paths), len(set(rel_paths)))
                    for rel in rel_paths:
                        self.assertTrue(rel.endswith(".py"), rel)
                        self.assertFalse(rel.startswith(("a/", "b/")), rel)

                    modules = [diffs.module_of(rel) for rel in rel_paths]
                    for mod in modules:
                        self.assertTrue(mod.startswith(mod_prefix), mod)

                    if len(modules) > 1:
                        saw_multi_file = True
                    if sum(1 for line in diff_text.splitlines() if _HUNK.match(line)) > 1:
                        saw_multi_hunk = True

                    base, expected = _synthesize_base_and_expected(diff_text)
                    self.assertEqual(list(base.keys()), modules)
                    applied = diffs.apply(diff_text, base.get)
                    self.assertEqual(applied, expected)

                    # Zero-fuzz context check: corrupted base for the first module must refuse.
                    broken = {**base, modules[0]: "# mismatched base source\n"}
                    with self.assertRaisesRegex(ValueError, "does not apply"):
                        diffs.apply(diff_text, broken.get)

        self.assertTrue(saw_multi_file, "expected at least one multi-file real patch")
        self.assertTrue(saw_multi_hunk, "expected at least one multi-hunk real patch")


class DiffHeaderEdgeCaseTests(unittest.TestCase):
    """Edge cases in real `git diff` / `diff -u` headers and hunk contents."""

    def test_git_diff_with_index_mode_and_multi_hunk_multi_file(self):
        patch = (
            "diff --git a/pkg/alpha.py b/pkg/alpha.py\n"
            "old mode 100644\n"
            "new mode 100755\n"
            "index 1111111..2222222\n"
            "--- a/pkg/alpha.py\t2026-11-19 00:00:00 +0000\n"
            "+++ b/pkg/alpha.py\t2026-11-19 00:00:01 +0000\n"
            "@@ -1,3 +1,3 @@\n"
            " A = 1\n"
            "-B = 2\n"
            "+B = 20\n"
            " C = 3\n"
            "@@ -6,2 +6,3 @@\n"
            " F = 6\n"
            " G = 7\n"
            "+H = 8\n"
            "diff --git a/pkg/sub/__init__.py b/pkg/sub/__init__.py\n"
            "new file mode 100644\n"
            "index 0000000..e69de29\n"
        )
        self.assertEqual(diffs.paths(patch), ["pkg/alpha.py", "pkg/sub/__init__.py"])
        alpha_base = "A = 1\nB = 2\nC = 3\nD = 4\nE = 5\nF = 6\nG = 7\n"
        out = diffs.apply(patch, lambda m: alpha_base if m == "pkg.alpha" else None)
        self.assertEqual(
            out,
            {
                "pkg.alpha": "A = 1\nB = 20\nC = 3\nD = 4\nE = 5\nF = 6\nG = 7\nH = 8\n",
                "pkg.sub": "",
            },
        )

    def test_hunk_bodies_with_diff_like_markers_do_not_fool_header_parser(self):
        base = (
            '"""Docstring:\n'
            "-- old note\n"
            '"""\n'
            'TAG = "old"\n'
        )
        patch = (
            "diff --git a/pkg/mod.py b/pkg/mod.py\n"
            "index 1234567..89abcdef 100644\n"
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -1,4 +1,5 @@\n"
            ' """Docstring:\n'
            "--- old note\n"
            "+++ new note\n"
            "+++ /dev/null\n"
            ' """\n'
            '-TAG = "old"\n'
            '+TAG = "+++ /dev/null"\n'
        )
        self.assertEqual(diffs.paths(patch), ["pkg/mod.py"])
        out = diffs.apply(patch, lambda m: base if m == "pkg.mod" else None)
        self.assertEqual(
            out,
            {
                "pkg.mod": (
                    '"""Docstring:\n'
                    "++ new note\n"
                    "++ /dev/null\n"
                    '"""\n'
                    'TAG = "+++ /dev/null"\n'
                )
            },
        )

    def test_empty_delete_rename_binary_mode_non_python_and_traversal_refused(self):
        empty_del = (
            "diff --git a/pkg/gone.py b/pkg/gone.py\n"
            "deleted file mode 100644\n"
            "index e69de29..0000000\n"
        )
        self.assertEqual(diffs.paths(empty_del), [])
        with self.assertRaisesRegex(ValueError, "deletes a module"):
            diffs.apply(empty_del, lambda _m: "")

        rename_patch = (
            "diff --git a/pkg/old.py b/pkg/new.py\n"
            "similarity index 100%\n"
            "rename from pkg/old.py\n"
            "rename to pkg/new.py\n"
        )
        with self.assertRaisesRegex(ValueError, "deletes a module"):
            diffs.apply(rename_patch, lambda _m: "X = 1\n")

        non_py_mode = (
            "diff --git a/scripts/run.sh b/scripts/run.sh\n"
            "old mode 100644\n"
            "new mode 100755\n"
        )
        with self.assertRaisesRegex(ValueError, "not modules"):
            diffs.apply(non_py_mode, lambda _m: None)

        non_py_bin = (
            "diff --git a/assets/blob.bin b/assets/blob.bin\n"
            "index 1111111..2222222 100644\n"
            "Binary files a/assets/blob.bin and b/assets/blob.bin differ\n"
        )
        with self.assertRaisesRegex(ValueError, "not modules"):
            diffs.apply(non_py_bin, lambda _m: None)

        traversal = (
            "--- /dev/null\n"
            "+++ b/../escape.py\n"
            "@@ -0,0 +1 @@\n"
            "+X = 1\n"
        )
        with self.assertRaisesRegex(ValueError, "unsafe diff path"):
            diffs.apply(traversal, lambda _m: None)


if __name__ == "__main__":
    unittest.main()
