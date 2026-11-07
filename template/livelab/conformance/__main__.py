"""Run the conformance cases against a lab and print a pass table.

    python3 -m conformance --lab module:Class          # from template/livelab
    python3 -m conformance --lab module:Class --only c4_uninstall,c7_chain
    python3 -m conformance --lab module:Class --receipts receipts.jsonl --json

`--lab` takes `module:Class` (a LabUnderTest; its constructor takes `receipts`). Each engine's
lab lives in `repos/<engine>/livelab/`; LABS maps short names to them once they exist.
`--receipts` keeps the txn / proof / undo receipts the suite's transactions write.

The run's own verdict is printed, not written as a receipt: the receipt proposal
(repos/vllm/livelab/kernel/receipt.py) has no record kind for a conformance run, and adding one extends the
receipt writer, which needs its human owner first (AGENTS.md, Trust).
"""
import argparse
import importlib
import json
import sys

from . import cases

LABS = {}   # short name -> "module:Class"


def load(spec):
    mod, _, cls = LABS.get(spec, spec).partition(":")
    return getattr(importlib.import_module(mod), cls)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m conformance")
    ap.add_argument("--lab", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--receipts", default=None)
    ap.add_argument("--json", action="store_true", help="print the results as JSON too")
    a = ap.parse_args(argv)
    Lab = load(a.lab)
    names = [n for n in cases.CASES if not a.only or n in a.only.split(",")]
    rows = [cases.run(n, lambda: Lab(receipts=a.receipts)) for n in names]
    w = max(len(n) for n in names)
    print(f"{'case':<{w}}  result  seconds  evidence")
    for r in rows:
        first = r["error"] or (r["evidence"][0] if r["evidence"] else "")
        print(f"{r['case']:<{w}}  {r['result']:<6}  {r['seconds']:>7.3f}  {first}")
        for line in r["evidence"][1:]:
            print(f"{'':<{w}}  {'':<6}  {'':>7}  {line}")
        if r["xfail_reason"]:
            print(f"{'':<{w}}  {'':<6}  {'':>7}  xfail: {r['xfail_reason']}")
    bad = [r["case"] for r in rows if r["result"] in ("FAIL", "XPASS")]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} as expected"
          + (f"; failing: {', '.join(bad)}" if bad else ""))
    print("Not written as a receipt: the receipt proposal has no conformance record "
          "(needs the receipt writer's human owner).")
    if a.json:
        print(json.dumps({"lab": a.lab, "cases": rows}, indent=1, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
