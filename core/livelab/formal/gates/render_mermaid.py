#!/usr/bin/env python3
"""Render a TLC trace.json into a Mermaid stateDiagram-v2 transition graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def render_mermaid(trace_data: dict) -> str:
    states = trace_data.get("states") or []
    if not states:
        raise ValueError("trace contains no states")
    first = states[0].get("values") or {}
    summary = ", ".join(f"{k}={first[k]}" for k in sorted(first)[:4]) or "Init"
    lines = ["stateDiagram-v2", f"    [*] --> S1: {summary}"]
    for pos in range(1, len(states)):
        prev_vals, cur = states[pos - 1].get("values") or {}, states[pos]
        cur_vals = cur.get("values") or {}
        m = re.search(r"<([A-Za-z0-9_]+)", str(cur.get("action") or ""))
        action = re.sub(r"[^A-Za-z0-9_.-]+", "_", m.group(1) if m else str(cur.get("action") or f"step_{pos + 1}")).strip("_")
        changed = [f"{k}={cur_vals[k]}" for k in sorted(cur_vals) if prev_vals.get(k) != cur_vals[k]]
        detail = f"{action} ({', '.join(changed[:3])})" if changed else action
        lines.append(f"    S{pos} --> S{pos + 1}: {detail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="path to TLC trace.json")
    args = parser.parse_args(argv)
    sys.stdout.write(render_mermaid(json.loads(args.trace.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
