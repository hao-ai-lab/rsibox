#!/usr/bin/env python3
"""Audit Antigravity transcript.jsonl against claimed test and commit outcomes."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit

EXIT_RE = re.compile(r"Command exited with code (\d+)")
RAN_RE = re.compile(r"Ran (\d+) tests?")
SKIP_RE = re.compile(r"skipped=(\d+)")
SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
ZERO_SKIP_CLAIM_RE = re.compile(r"\b(?:0\s+skipped|no\s+skipped|all\s+passed)\b", re.IGNORECASE)


def audit_transcript(
    transcript_path: Path, require_sha: str | None = None, claim_text: str = ""
) -> tuple[bool, dict[str, Any]]:
    """Extract executed commands, exit codes, skips, and SHAs from transcript.jsonl."""
    commands: list[dict[str, Any]] = []
    pending_calls: dict[str, str] = {}
    observed_shas: set[str] = set()
    total_skips = 0

    with transcript_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            entry = json.loads(raw)
            cmd_line = entry.get("CommandLine") or entry.get("commandLine")
            call_id = entry.get("call_id") or entry.get("id")
            if isinstance(cmd_line, str) and call_id:
                pending_calls[str(call_id)] = cmd_line
            out = entry.get("output") or entry.get("Output") or json.dumps(entry)
            if isinstance(out, str):
                exit_m, ran_m, skip_m = EXIT_RE.search(out), RAN_RE.search(out), SKIP_RE.search(out)
                if skip_m:
                    total_skips += int(skip_m.group(1))
                observed_shas.update(SHA_RE.findall(out))
                if exit_m or cmd_line:
                    commands.append(
                        {
                            "command": cmd_line or pending_calls.get(str(call_id), ""),
                            "exit_code": int(exit_m.group(1)) if exit_m else entry.get("exit_code"),
                            "tests_ran": int(ran_m.group(1)) if ran_m else None,
                            "skipped": int(skip_m.group(1)) if skip_m else 0,
                        }
                    )

    mismatches: list[str] = []
    if require_sha and not any(s.startswith(require_sha[:7]) for s in observed_shas):
        mismatches.append(f"required SHA {require_sha} not observed in transcript")
    if total_skips > 0 and ZERO_SKIP_CLAIM_RE.search(claim_text):
        mismatches.append(f"claim states zero skips/all passed, but transcript recorded {total_skips} skips")
    if not commands:
        mismatches.append("no command executions found in transcript")

    failed = [c for c in commands if c["exit_code"] not in (None, 0)]
    return not mismatches, {
        "commands_count": len(commands),
        "failed_count": len(failed),
        "total_skips": total_skips,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox Gate 1 transcript trajectory auditor")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--require-sha", default=None)
    parser.add_argument("--claim-file", type=Path, default=None)
    args = parser.parse_args(argv)
    claim_text = args.claim_file.read_text(encoding="utf-8") if args.claim_file else ""
    ok, details = audit_transcript(args.transcript, require_sha=args.require_sha, claim_text=claim_text)
    return emit(ok, "trajectory_audit", **details)


if __name__ == "__main__":
    raise SystemExit(main())
