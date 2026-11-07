"""LabUnderTest: what a live lab exposes so the conformance cases can drive it.

The cases call only these methods. The first five are the interface verbs
(INTERFACE-v1.md); the last four are a test harness each lab provides for conformance only
(named edits, a fresh process, fault injection). The same suite runs against the toy lab
on CPU now and against an engine adapter (vLLM) later.

Replies are plain dicts and never raise (a lab that raises fails the case).

    install(patch) -> {"ok", "id", "outcome", "codes", "mechanisms", "findings", "cost_s"}
        patch: {"files": {module: full patched text}, "accept": [check], "undo_resets": [..]}
        outcome: committed | aborted | rolled_back | refused | in-doubt
        codes: the checks that blocked it (empty when committed)
    check(workload=None) -> {"ok", "output", "proven", "witness", "findings", "cost_s"}
        output: JSON-comparable, bit-exact across runs of the same code
        proven: True when every live edit ran during the workload, False when one did
        not, None when no edit is live
        witness: {participant: {key: calls}}
    reset(kind="state") -> {"ok", "reset": [what], "cost_s"}
        kind "state": drain in-flight work, clear volatile state (caches, buffers)
        kind "respawn": restart the worker group from warm caches; clears poison
    uninstall(id=None) -> {"ok", "undone": [ids], "state", "cost_s"}
        undoes id and every later edit, newest first (default: the newest)
    status() -> {"ok", "state", "txns", "fingerprint", "poison"}
        state: base | edited | poisoned; fingerprint: digest of the code identity (content
        hash and object identity) of every function in the engine modules, per participant

Harness:

    patches() -> {"noop", "visible", "chain": [3 patches], "multi", "unreached",
                  "refused": {"syntax", "closure", "module_state", "empty"}}
        noop: changes code, keeps behavior. visible: changes the workload's output.
        chain: each on top of the previous, each visible. multi: >= 3 functions in >= 2
        modules. unreached: an edit the workload never runs as Python (startup-only code,
        code inside a captured graph). refused: one edit per refusal class.
    fresh_check(patch, workload=None) -> {"ok", "output"}
        the workload in a new process started from the base files plus patch (None: base)
    inject(fault, **kw) -> {"ok"}
        fail_apply (after=n swaps on one participant), fail_commit (one participant's
        commit fails), crash, hang (in the next check), lost (a participant stops
        answering), tamper (code changed outside the journal)
    close()
"""
from typing import Protocol


class LabUnderTest(Protocol):
    name: str

    def install(self, patch: dict) -> dict: ...

    def check(self, workload: dict | None = None) -> dict: ...

    def reset(self, kind: str = "state") -> dict: ...

    def uninstall(self, id: str | None = None) -> dict: ...

    def status(self) -> dict: ...

    def patches(self) -> dict: ...

    def fresh_check(self, patch: dict | None, workload: dict | None = None) -> dict: ...

    def inject(self, fault: str, **kw) -> dict: ...

    def close(self) -> None: ...
