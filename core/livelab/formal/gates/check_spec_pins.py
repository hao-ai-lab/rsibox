#!/usr/bin/env python3
"""Verify or refresh SHA-256 pins for all RSIBox TLA+ specification files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

FORMAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PINS = FORMAL_DIR / "SPEC_PINS.sha256"
SPEC_SUBDIRS = ("core/livelab/formal", "core/livelab/kernel/formal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_specs(repo_root: Path) -> set[str]:
    r = repo_root.resolve()
    found = {p.resolve().relative_to(r).as_posix() for s in SPEC_SUBDIRS if (repo_root / s).is_dir()
             for p in (repo_root / s).rglob("*.tla")}
    return found or {p.resolve().relative_to(r).as_posix() for p in repo_root.glob("*.tla")}


def verify_pins(pins_path: Path, repo_root: Path) -> list[str]:
    if not pins_path.is_file():
        return [f"missing pins file: {pins_path}"]
    resolved_root, errors, seen = repo_root.resolve(), [], set()
    for lineno, raw in enumerate(pins_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"{pins_path}:{lineno}: malformed pin line")
            continue
        digest, rel = parts
        target = (repo_root / rel).resolve()
        if Path(rel).is_absolute() or not target.is_relative_to(resolved_root):
            errors.append(f"{pins_path}:{lineno}: path outside repo: {rel}")
            continue
        seen.add(target.relative_to(resolved_root).as_posix())
        if not target.is_file():
            errors.append(f"missing pinned spec: {rel}")
        elif _sha256(target) != digest:
            errors.append(f"sha256 mismatch for {rel}")
    errors.extend(f"unpinned spec on disk: {m}" for m in sorted(_discover_specs(repo_root) - seen))
    return errors if seen else errors + [f"no pin entries in {pins_path}"]


def update_pins(pins_path: Path, repo_root: Path, specs: list[str] | None = None) -> None:
    targets = specs or sorted(_discover_specs(repo_root))
    pins_path.write_text("".join(f"{_sha256(repo_root / rel)}  {rel}\n" for rel in targets), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--repo", type=Path, default=FORMAL_DIR.parents[2])
    parser.add_argument("--update", action="store_true", help="refresh pin hashes")
    args = parser.parse_args(argv)
    if args.update:
        update_pins(args.pins, args.repo)
        return 0
    errors = verify_pins(args.pins, args.repo)
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr) if errors else print(
        f"OK: verified spec pins in {args.pins}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
