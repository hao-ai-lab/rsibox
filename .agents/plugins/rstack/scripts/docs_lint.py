#!/usr/bin/env python3
"""Deterministic lint gate for public RSIBox markdown documentation."""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
MD_LINK_RE = re.compile(
    r"!?\[((?:[^\[\]]|\[[^\]]*\])*)\]\((?:<([^>\n]+)>|((?:[^()\s]+|\([^()\s]*\))+))(?:\s+[\"'][^\"']*[\"'])?\)"
)
HTML_TARGET_RE = re.compile(r'\b(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
SCHEME_RE = re.compile(r"^(?:https?|mailto|tel|file|customization|data):", re.IGNORECASE)
DUP_PAREN_RE = re.compile(r"\b([a-zA-Z0-9_-]{3,})\s+\(\1\)", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`]+`")


def _iter_non_fenced_lines(text: str) -> list[tuple[int, str]]:
    """Return 1-based (lineno, line) pairs outside fenced code blocks."""
    rows: list[tuple[int, str]] = []
    fence_char = ""
    fence_len = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if match:
            marker, rest = match.group(1), match.group(2)
            if not fence_char:
                if (marker[0] == "`" and "`" in rest) or (marker[0] == "~" and "~" * len(marker) in rest):
                    rows.append((lineno, line))
                    continue
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len and not rest.strip():
                fence_char = ""
                fence_len = 0
            continue
        if not fence_char:
            rows.append((lineno, line))
    return rows


def _strip_emphasis_outside_code(text: str) -> str:
    """Strip Markdown code backticks and emphasis delimiters outside code spans."""
    parts = re.split(r"(`[^`]*`)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            out.append(part[1:-1])
        else:
            cleaned = part.replace("*", "")
            cleaned = re.sub(r"(?<!\w)_+|_+(?!\w)", "", cleaned)
            out.append(cleaned)
    return "".join(out)


def _slugify_heading(raw_heading: str) -> str:
    """Convert a Markdown heading into its GitHub-style anchor slug."""
    cleaned = html.unescape(raw_heading)
    cleaned = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = _strip_emphasis_outside_code(cleaned).replace("`", "").strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return re.sub(r"\s", "-", cleaned)


def _heading_slugs(md_path: Path, cache: dict[Path, set[str]]) -> set[str]:
    """Collect deduplicated GitHub heading slugs for a markdown file."""
    resolved = md_path.resolve()
    if resolved in cache:
        return cache[resolved]
    occurrences: dict[str, int] = {}
    used: set[str] = set()
    for _, line in _iter_non_fenced_lines(resolved.read_text(encoding="utf-8")):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base_slug = _slugify_heading(match.group(1))
        if not base_slug:
            continue
        count = occurrences.get(base_slug, 0)
        candidate = base_slug if count == 0 else f"{base_slug}-{count}"
        while candidate in used:
            count += 1
            candidate = f"{base_slug}-{count}"
        occurrences[base_slug] = count + 1
        used.add(candidate)
    cache[resolved] = used
    return used


def _extract_md_targets(text: str) -> list[str]:
    """Extract markdown link/image targets including nested linked images."""
    targets: list[str] = []
    for match in MD_LINK_RE.finditer(text):
        label = match.group(1)
        target = match.group(2) or match.group(3)
        if target:
            targets.append(target)
        if label and "[" in label:
            targets.extend(_extract_md_targets(label))
    return targets


def _target_docs(repo: Path) -> list[Path]:
    """Return all top-level, docs, and core/livelab markdown files to lint."""
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in (repo / "README.md", repo / "CONTRIBUTING.md"):
        if path.is_file() and path not in seen:
            seen.add(path)
            candidates.append(path)
    docs_dir = repo / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.rglob("*.md")):
            if path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)
    livelab_dir = repo / "core" / "livelab"
    if livelab_dir.is_dir():
        for path in sorted(livelab_dir.rglob("README.md")):
            if path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


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
            targets = _extract_md_targets(stripped_line) + [m.group(1) for m in HTML_TARGET_RE.finditer(stripped_line)]
            for raw_target in targets:
                target = raw_target.strip("<>")
                if SCHEME_RE.match(target):
                    continue
                rel_part, _, anchor = target.partition("#")
                if rel_part.startswith("/"):
                    dest = (repo / rel_part.lstrip("/")).resolve()
                elif rel_part:
                    dest = (doc_path.parent / rel_part).resolve()
                else:
                    dest = doc_path.resolve()
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
