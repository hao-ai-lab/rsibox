#!/usr/bin/env python3
"""Deterministic lint gate for public RSIBox markdown documentation."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']')
DUP_PAREN_RE = re.compile(r"\b([a-zA-Z0-9_-]{3,})\s+\(\1\)", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`]+`")


def _iter_non_fenced_lines(text: str) -> list[tuple[int, str]]:
    """Return 1-based (lineno, line) pairs outside fenced code blocks."""
    rows: list[tuple[int, str]] = []
    fence_marker = ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)[0]
            if not fence_marker:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = ""
            continue
        if not fence_marker:
            rows.append((lineno, line))
    return rows


def _slugify_heading(raw_heading: str) -> str:
    """Convert a Markdown heading into its GitHub-style anchor slug."""
    cleaned = re.sub(r"`([^`]*)`", r"\1", raw_heading)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned).strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return re.sub(r"\s", "-", cleaned)


def _heading_slugs(md_path: Path, cache: dict[Path, set[str]]) -> set[str]:
    """Collect deduplicated GitHub heading slugs for a markdown file."""
    resolved = md_path.resolve()
    if resolved in cache:
        return cache[resolved]
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for _, line in _iter_non_fenced_lines(resolved.read_text(encoding="utf-8")):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base_slug = _slugify_heading(match.group(1))
        if not base_slug:
            continue
        count = seen.get(base_slug, -1) + 1
        seen[base_slug] = count
        slugs.add(base_slug if count == 0 else f"{base_slug}-{count}")
    cache[resolved] = slugs
    return slugs


def _target_docs(repo: Path) -> list[Path]:
    """Return all top-level and core markdown documentation files to lint."""
    candidates = [
        repo / "README.md",
        repo / "CONTRIBUTING.md",
        repo / "core" / "livelab" / "README.md",
        repo / "core" / "livelab" / "conformance" / "README.md",
    ]
    docs_dir = repo / "docs"
    if docs_dir.is_dir():
        candidates.extend(sorted(docs_dir.glob("*.md")))
    return [path for path in candidates if path.is_file()]


def lint_docs(repo: Path) -> tuple[int, list[dict[str, object]]]:
    """Check docs for broken relative links, invalid anchors, and duplicate artifacts."""
    files = _target_docs(repo)
    slug_cache: dict[Path, set[str]] = {}
    violations: list[dict[str, object]] = []
    for doc_path in files:
        rel_doc = doc_path.relative_to(repo).as_posix()
        for lineno, line in _iter_non_fenced_lines(doc_path.read_text(encoding="utf-8")):
            stripped_line = INLINE_CODE_RE.sub("", line)
            for dup_match in DUP_PAREN_RE.finditer(stripped_line):
                violations.append({"file": rel_doc, "line": lineno, "kind": "duplicate_artifact", "detail": dup_match.group(0)})
            targets = [m.group(1) for m in MD_LINK_RE.finditer(stripped_line)] + [m.group(1) for m in HREF_RE.finditer(stripped_line)]
            for raw_target in targets:
                target = raw_target.strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "tel:", "data:")):
                    continue
                rel_part, _, anchor = target.partition("#")
                dest = (doc_path.parent / rel_part).resolve() if rel_part else doc_path.resolve()
                if rel_part and not dest.exists():
                    violations.append({"file": rel_doc, "line": lineno, "kind": "broken_link", "detail": target})
                elif anchor and dest.is_file() and dest.suffix.lower() == ".md":
                    if anchor not in _heading_slugs(dest, slug_cache):
                        violations.append({"file": rel_doc, "line": lineno, "kind": "broken_anchor", "detail": target})
    return len(files), violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint public RSIBox markdown docs")
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = find_repo_root(args.repo)
    files_checked, violations = lint_docs(repo)
    return emit(not violations, "docs_lint", files_checked=files_checked, violations=violations)


if __name__ == "__main__":
    raise SystemExit(main())
