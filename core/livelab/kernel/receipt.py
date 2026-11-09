"""Receipt: linked records per transaction, written by one writer to an append-only JSON-lines file.

PROPOSAL, NEEDS A HUMAN OWNER. AGENTS.md ("Trust") says agents never write the code that
writes receipts. This file was written by an agent as a proposal for review; until a human
owns it, receipts it writes are drafts, not trust anchors.

This is the only code that writes receipts. Engine code, mechanisms and agents hand it a
record; they never write the file. Each record is one JSON object on one line; the writer
opens the file in append mode under an exclusive lock and never rewrites a line.

A transaction leaves up to three records that share its `id` (schema proposed for INTERFACE
v1; see core/livelab/INTERFACE.md):

- `txn` (from Coordinator.run): plan, each participant's prepare / validate / commit / abort
  replies, the outcome, in-transaction rollback, findings and decisions, edit_to_run_s,
  prepare_s (phase 1, steps go on) and gap_s (phase 2, the loop stalls); the post-commit
  `rebuild` (R1 ...) with recapture_s, and edit_to_commit_s. When the caller timed out, the
  record is written once `status(txn)` first sees the transaction finished.
- `proof` (from Coordinator.prove, after the workload ran): live witness counts per
  participant, the calls made during graph capture apart (`witness_capture`, and `ran`: "ran
  during capture N, ran live M"), their findings (zero_witness, running_frame), the caller's
  golden results, edit_to_finish_s.
- `undo` (from Coordinator.undo, whenever it happens): code undo per participant, then the
  data resets the plan declared (`undo_resets`: R2, R3, ...), the rebuild from the base code,
  status, undo_s.

Every record also has `schema` and `kind`; the writer adds `writer: {pid, written_at}`.
"""
import fcntl
import json
import os
import time

SCHEMA = "fastrsi-receipt/proposed-v1"
FIELDS = {
    "txn": ("schema", "kind", "id", "interface_version", "engine", "base_hash", "patch_hash",
            "plan", "actual_recipes", "participants", "commit_hashes", "findings", "decisions",
            "outcome", "timeline", "edit_to_run_s", "prepare_s", "gap_s", "baseline_restart_s",
            "rollback", "edit_to_commit_s", "rebuild", "recapture_s"),
    "proof": ("schema", "kind", "id", "witness", "findings", "golden", "timeline",
              "edit_to_finish_s", "witness_capture", "ran"),
    "undo": ("schema", "kind", "id", "participants", "resets", "status", "findings",
             "timeline", "undo_s", "rebuild", "recapture_s"),
}
OUTCOMES = ("committed", "aborted", "rolled_back")


def _invalid(record):
    kind = record.get("kind")
    if kind not in FIELDS:
        return f"unknown kind {kind!r}"
    missing = [f for f in FIELDS[kind] if f not in record]
    if missing:
        return f"{kind}: missing {missing}"
    if record["schema"] != SCHEMA:
        return f"schema {record['schema']!r}"
    if kind == "txn" and record["outcome"] not in OUTCOMES:
        return f"outcome {record['outcome']!r}"
    return None


class ReceiptWriter:
    def __init__(self, path):
        self.path = path

    def append(self, record):
        """Validate and append one record. Returns {"ok", "line"} or {"ok": False, "error"};
        never raises."""
        bad = _invalid(record)
        if bad:
            return {"ok": False, "error": f"invalid receipt: {bad}"}
        rec = {**record, "writer": {"pid": os.getpid(), "written_at": time.time()}}
        try:
            line = json.dumps(rec, sort_keys=True, default=str)
            with open(self.path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "path": self.path, "line": line}


def read(path, txn=None):
    """All records, or the linked records of one transaction, in file order."""
    with open(path) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    return recs if txn is None else [r for r in recs if r["id"] == txn]
