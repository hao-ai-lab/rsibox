"""Receipt: one record per transaction, written by one writer to an append-only JSON-lines file.

This is the only code that writes receipts. Engine code, mechanisms and agents hand it a
record; they never write the file. Each record is one JSON object on one line; the writer
opens the file in append mode under an exclusive lock and never rewrites a line.

Schema (`SCHEMA`, proposed for INTERFACE v1; see template/livelab/INTERFACE.md):

| field | type | meaning |
|---|---|---|
| schema | str | "fastrsi-receipt/proposed-v1" |
| id | str | transaction id |
| interface_version | str | live lab INTERFACE version the lab implements |
| engine | {name, version, commit} | the engine under change |
| base_hash, patch_hash | str | text_hash of the base and patched sources (and repair cells) |
| plan | {targets, predicted_resources, recipes, certainty, accept} | what the planner predicted |
| actual_recipes | [str] | recipes that ran (e.g. R0 swap, R10 repair) |
| participants | {name: {prepare, commit, abort, undo}} | each call's ok-dict |
| proofs | {participants: {name: {hashes, witness}}, hashes_agree: bool} | after commit |
| findings | [finding + participant] | every finding of every phase |
| decisions | {accepted: [check], blocking: [check]} | how the accept list was applied |
| outcome | "committed" / "aborted" / "rolled_back" | |
| timeline | {edit, prepare_start, vote, committed, finished} | unix seconds, measured |
| edit_to_run_s | float or null | timeline.committed - timeline.edit (null if not committed) |
| edit_to_finish_s | float | timeline.finished - timeline.edit |
| baseline_restart_s | {seconds, source} or null | the warm restart this replaces |
| undo | {status, participants} | status: "not_needed", "undone", "partial" |
| writer | {pid, written_at} | added by the writer |
"""
import fcntl
import json
import os
import time

SCHEMA = "fastrsi-receipt/proposed-v1"
FIELDS = ("schema", "id", "interface_version", "engine", "base_hash", "patch_hash", "plan",
          "actual_recipes", "participants", "proofs", "findings", "decisions", "outcome",
          "timeline", "edit_to_run_s", "edit_to_finish_s", "baseline_restart_s", "undo")
OUTCOMES = ("committed", "aborted", "rolled_back")


class ReceiptWriter:
    def __init__(self, path):
        self.path = path

    def append(self, record):
        """Validate and append one record. Returns {"ok", "line"} or {"ok": False, "error"};
        never raises."""
        missing = [f for f in FIELDS if f not in record]
        if missing or record.get("schema") != SCHEMA or record.get("outcome") not in OUTCOMES:
            return {"ok": False, "error": f"invalid receipt: missing {missing}, schema "
                                          f"{record.get('schema')!r}, outcome "
                                          f"{record.get('outcome')!r}"}
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


def read(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]
