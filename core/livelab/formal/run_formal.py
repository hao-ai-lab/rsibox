#!/usr/bin/env python3
"""Run bounded TLC checks across all 11 RSIBox TLA+ specs, mutants, and witnesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parent
KERNEL_FORMAL, REPO_ROOT = ROOT.parent / "kernel" / "formal", ROOT.parents[2]
VERSION = "1.7.4"
JAR_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
URL = f"https://github.com/tlaplus/tlaplus/releases/download/v{VERSION}/tla2tools.jar"

FORMAL_MODELS = [
    ("pool", ROOT / "pool" / "SandboxPool.tla", ROOT / "pool" / "SandboxPool.cfg", "NeverRemoved"),
    ("coordinator", ROOT / "coordinator" / "Coordinator.tla", ROOT / "coordinator" / "Coordinator.cfg", "NeverSuperseded"),
    ("swap", ROOT / "swap" / "SwapLifecycle.tla", ROOT / "swap" / "SwapLifecycle.cfg", "NeverRolledBack"),
    ("shadow", ROOT / "shadow" / "StandbyShadow.tla", ROOT / "shadow" / "StandbyShadow.cfg", "NeverClosed"),
    ("autoscaler", ROOT / "autoscaler" / "PoolAutoScaler.tla", ROOT / "autoscaler" / "PoolAutoScaler.cfg", "NeverScaleDown"),
    ("snapshot", ROOT / "snapshot" / "SnapshotLifecycle.tla", ROOT / "snapshot" / "SnapshotLifecycle.cfg", "NeverReady"),
    ("forksafe", ROOT / "forksafe" / "ForksafeLockOrder.tla", ROOT / "forksafe" / "ForksafeLockOrder.cfg", "NeverChildReady"),
]
SUBSYSTEM_MUTANTS = [
    ("snapshot", "WriteReadyBeforeQuiesce", "ReadySentinelRequiresQuiesce"),
    ("snapshot", "WriteResumeBeforeRestore", "ResumeReadyRequiresRestore"),
    ("forksafe", "PreloadBeforeGuards", "PreloadAfterGuards"),
    ("forksafe", "ForkWhileCudaInit", "ForkRequiresCleanDriver"),
]
KERNEL_MODELS: dict[str, tuple[Path, dict[str, str], list[str]]] = {
    "Transaction": (KERNEL_FORMAL / "Transaction.tla", {"Participants": '{"p1", "p2"}', "Bug": '"None"'},
                    ["TypeOK", "NoSplitCommit", "RollbackFreeze"]),
    "Participant": (KERNEL_FORMAL / "Participant.tla", {"MaxJournalDepth": "2", "Bug": '"None"'},
                    ["TypeOK", "PartialImpliesPoisoned", "NoCommitAfterDeadline", "CommittedHasJournal"]),
    "SequentialTx": (KERNEL_FORMAL / "SequentialTx.tla", {"TxnIds": '{"t1", "t2"}', "Bug": '"None"'},
                     ["TypeOK", "EmptyStackRestoresBase", "NoDuplicateInStack"]),
    "Incarnation": (KERNEL_FORMAL / "Incarnation.tla",
                    {"Participants": '{"p1"}', "TxnIds": "{1, 2}", "MaxIncarnation": "2", "Bug": '"None"'},
                    ["TypeOK", "SingleActiveIncarnation", "StaleReplyRejection"]),
}


def cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    formal_by_name = {name: (tla, cfg) for name, tla, cfg, _ in FORMAL_MODELS}
    for name, tla, cfg, witness_inv in FORMAL_MODELS:
        out.append({"name": name, "tla": tla, "cfg": cfg, "expected": None})
        out.append({"name": f"witness-{tla.stem}-{witness_inv}", "tla": tla, "cfg": cfg, "expected": witness_inv})
    for model_name, bug, inv in SUBSYSTEM_MUTANTS:
        tla, cfg = formal_by_name[model_name]
        out.append({"name": f"mutant-{bug}", "tla": tla, "cfg": cfg, "constants": {"Bug": json.dumps(bug)}, "expected": inv})

    def add_k(name: str, mod: str, overrides: dict[str, str] | None = None, expected: str | None = None) -> None:
        tla, consts, invs = KERNEL_MODELS[mod]
        out.append({"name": name, "tla": tla, "constants": {**consts, **(overrides or {})},
                    "invariants": list(invs), "expected": expected})

    for name, mod in [("transaction", "Transaction"), ("participant", "Participant"),
                      ("sequential-tx", "SequentialTx"), ("incarnation", "Incarnation")]:
        add_k(name, mod)
    for mod, bug, inv in [("Transaction", "SkipRollbackFreeze", "RollbackFreeze"), ("Transaction", "AllowSplitCommit", "NoSplitCommit"),
                          ("Participant", "AllowLateCommit", "NoCommitAfterDeadline"), ("SequentialTx", "AllowDuplicateTxn", "NoDuplicateInStack"),
                          ("Incarnation", "AcceptStaleReply", "StaleReplyRejection"), ("Incarnation", "KeepOldIncarnation", "SingleActiveIncarnation")]:
        add_k(f"mutant-{bug}", mod, {"Bug": json.dumps(bug)}, inv)
    for mod, inv in [("Transaction", "NeverCommitted"), ("Transaction", "NeverRolledBack"), ("Transaction", "NeverRecoveryRequired"),
                     ("Participant", "NeverCommitted"), ("Participant", "NeverPartialPoisoned"),
                     ("SequentialTx", "NeverTwoStacked"), ("SequentialTx", "NeverRecovery"),
                     ("Incarnation", "NeverAccepted"), ("Incarnation", "NeverRespawned")]:
        add_k(f"witness-{mod}-{inv}", mod, expected=inv)
    return out


def ensure_jar(path: Path, bootstrap: bool) -> Path:
    if not path.exists():
        if not bootstrap:
            raise RuntimeError("TLC jar missing; run with --bootstrap or provide --jar")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "rsibox-ci/1"}), timeout=60) as resp:
            data = resp.read()
        if hashlib.sha256(data).hexdigest() != JAR_SHA256:
            raise RuntimeError("downloaded TLC jar checksum mismatch")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != JAR_SHA256:
        raise RuntimeError(f"TLC jar checksum mismatch; expected pinned v{VERSION}")
    return path.resolve()


def export_trace(stdout: str, path: Path) -> None:
    parts = re.split(r"(?m)^State (\d+):\s*(.*)\n", stdout)
    if len(parts) < 4:
        raise RuntimeError("TLC reported a violation without a state trace")
    states = []
    for idx in range(1, len(parts), 3):
        values: dict[str, Any] = {}
        for name, lit in re.findall(r"(?m)^(?:/\\\s*)?(\w+)\s*=\s*(.*)$", parts[idx + 2]):
            values[name] = (lit == "TRUE") if lit in ("TRUE", "FALSE") else (
                int(lit) if re.fullmatch(r"\d+", lit) else (json.loads(lit) if re.fullmatch(r'"[^"\\]*"', lit) else lit.strip())
            )
        states.append({"index": int(parts[idx]), "action": parts[idx + 1].strip(), "values": values, "raw": parts[idx + 2].strip()})
    if len(states) < 2:
        raise RuntimeError("expected a nontrivial counterexample/witness")
    path.write_text(json.dumps({"states": states}, indent=2) + "\n", encoding="utf-8")


def _write_cfg(case: dict[str, Any], cfg_dest: Path) -> None:
    expected = case["expected"]
    if "cfg" in case:
        text = Path(case["cfg"]).read_text(encoding="utf-8")
        for k, v in case.get("constants", {}).items():
            text, n = re.subn(rf"(?m)^(\s*{re.escape(k)}\s*=\s*).*$", lambda m, val=v: m.group(1) + val, text)
            if n == 0:
                raise RuntimeError(f"constant {k} not found in {case['cfg']}")
        if expected and not re.search(rf"(?m)^INVARIANT\s+{re.escape(expected)}\s*$", text):
            text = text.rstrip() + f"\nINVARIANT {expected}\n"
        cfg_dest.write_text(text, encoding="utf-8")
        return
    invs = list(case["invariants"]) + ([expected] if expected and expected not in case["invariants"] else [])
    cfg = "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nCONSTANTS\n" + "".join(
        f"  {k} = {v}\n" for k, v in case["constants"].items()
    ) + "INVARIANTS\n" + "".join(f"  {inv}\n" for inv in invs)
    cfg_dest.write_text(cfg, encoding="utf-8")


def check(case: dict[str, Any], jar: Path, java: str, timeout: int, output: Path) -> dict[str, Any]:
    dest = output / case["name"]
    dest.mkdir(parents=True, exist_ok=True)
    tla_src = Path(case["tla"])
    module = tla_src.stem
    tla_dest, cfg_dest, trace = dest / f"{module}.tla", dest / f"{module}.cfg", dest / "trace.json"
    shutil.copyfile(tla_src, tla_dest)
    _write_cfg(case, cfg_dest)
    (dest / "result.json").unlink(missing_ok=True)
    trace.unlink(missing_ok=True)
    cmd = [java, "-Xmx1g", "-XX:+UseParallelGC", "-cp", str(jar), "tlc2.TLC",
           "-workers", "1", "-cleanup", "-metadir", str(dest / "states"), "-config", cfg_dest.name, module]
    try:
        res = subprocess.run(cmd, cwd=dest, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (dest / "tlc.log").write_bytes(exc.stdout or b"")
        raise RuntimeError(f"{case['name']}: timeout; verification is incomplete") from exc
    (dest / "tlc.log").write_text(res.stdout, encoding="utf-8")
    violation = re.search(r"Invariant (\w+) is violated", res.stdout)
    if violation:
        export_trace(res.stdout, trace)
    expected = case["expected"]
    ok = (res.returncode != 0 and violation is not None and violation[1] == expected and trace.exists()) if expected else (
        res.returncode == 0 and "Model checking completed. No error has been found." in res.stdout)
    if not ok:
        raise RuntimeError(f"{case['name']}: unexpected TLC result (exit {res.returncode})\n{res.stdout[-3000:]}")
    counts = re.search(r"([\d,]+) states generated, ([\d,]+) distinct states found", res.stdout)
    summary = {
        "name": case["name"], "module": module, "expected": expected,
        "outcome": "counterexample" if expected else "checked",
        "model_sha256": hashlib.sha256(tla_dest.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None,
        "tlc_version": VERSION, "distinct_states": int(counts[2].replace(",", "")) if counts else None,
    }
    (dest / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{case['name']}: {summary['outcome']} {expected or ''} ({summary['distinct_states']} states)", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="download checksum-pinned TLC jar if missing")
    parser.add_argument("--jar", type=Path, default=REPO_ROOT / ".tla-cache" / "tla2tools.jar")
    parser.add_argument("--java", default=os.environ.get("JAVA", "java"))
    parser.add_argument("--timeout", type=int, default=180, help="seconds per model")
    parser.add_argument("--case", action="append", dest="names", help="run only named case; repeatable")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".tla-cache" / "runs")
    args = parser.parse_args(argv)
    all_cases = cases()
    selected = [c for c in all_cases if args.names is None or c["name"] in args.names]
    if not selected or (args.names and set(args.names) - {c["name"] for c in selected}):
        parser.error("unknown case name")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    jar = ensure_jar(args.jar, args.bootstrap)
    summaries = [check(c, jar, args.java, args.timeout, output) for c in selected]
    (output / "report.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as err:
        print(err, file=sys.stderr)
        raise SystemExit(1)
