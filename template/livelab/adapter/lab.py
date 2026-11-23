"""TemplateLab: runnable LabUnderTest skeleton for onboarding a new engine.

Running `python3 -m rsibox conformance template` (or `python3 -m conformance --lab
adapter.lab:TemplateLab` from `core/livelab`) instantiates and closes this class cleanly,
failing all 11/11 conformance cases (c0-c10) with numbered, actionable `NotImplementedError`
messages that guide an engine author from 0/11 to 11/11.

Fast path: once ready to implement, replace `TemplateLab`'s body by subclassing
`rsibox.BaseLab`, which already implements `install`, `check`, `reset`, `uninstall`,
`status`, `live_text` stack tracking, witness merging, and poison-on-fault handling.
You only implement `_start_engine()`, `_stop_engine()`, `_run_workload()`, `_reset_state()`,
plus the conformance harness (`patches()`, `fresh_check()`, `inject()`) using
`rsibox.edit_once`, `rsibox.PackageOverlay`, `rsibox.run_fresh_subprocess`, and
`rsibox.inject_participant_fault`.
"""
import adapter  # noqa: F401  # puts core/livelab on sys.path
import rsibox  # noqa: F401  # BaseLab, ipc, proc, hooks, harness
from kernel import receipt as R

STEPS = {
    "status": (
        "Step 1/6 (status -> unlocks base checks in c0-c10): subclass rsibox.BaseLab (or build "
        "a Coordinator/StepGapCoordinator over your participants and wrap it in "
        "kernel.session.Session(coordinator, modules)) so status() returns "
        "{'ok', 'state', 'txns', 'fingerprint', 'poison'}."
    ),
    "check": (
        "Step 2/6 (check -> unlocks c0_repeatable with Step 3): implement _run_workload(workload) "
        "on rsibox.BaseLab (or call self.coordinator.prove(txn) for each txn in "
        "self.session.stack) and return {'ok', 'output', 'proven', 'witness', 'findings', 'cost_s'}."
    ),
    "reset": (
        "Step 3/6 (reset -> unlocks c0_repeatable and c8 respawn): implement _reset_state() and "
        "_stop_engine()/_start_engine() on rsibox.BaseLab so kind='state' drains volatile caches "
        "and kind='respawn' restarts workers and calls self.session.recovered(self.coordinator)."
    ),
    "patches": (
        "Step 4a/6 (patches -> unlocks c1, c2, c4, c6, c7, c9, c10): return real source edits "
        "(using rsibox.edit_once) against your engine modules with keys {'noop', 'visible', "
        "'chain' (3 stacked edits), 'multi' (>=3 functions across >=2 modules), 'unreached', "
        "'refused': {'syntax', 'closure', 'module_state', 'empty'}}."
    ),
    "install": (
        "Step 4b/6 (install -> unlocks c1, c2, c4, c6, c7, c9): inherited for free from "
        "rsibox.BaseLab (override _build_plan only for custom routing/recapture), or build a "
        "kernel plan from patch['files'] and call self.session.install(plan)."
    ),
    "uninstall": (
        "Step 4c/6 (uninstall -> unlocks c1, c2, c4, c7, c9): inherited for free from "
        "rsibox.BaseLab (calls self.session.uninstall(id) and pops matching live_text entries)."
    ),
    "fresh_check": (
        "Step 5/6 (fresh_check -> unlocks c3_fresh_equivalence and c10_not_proven): use "
        "rsibox.PackageOverlay + rsibox.run_fresh_subprocess (or rsibox.install_overlay) to "
        "run the workload in a fresh subprocess over base files plus (patch or {}).get('files')."
    ),
    "inject": (
        "Step 6/6 (inject -> unlocks c5_failed_install and c8_crash_and_hang): delegate "
        "'fail_apply', 'fail_commit', and 'tamper' to rsibox.inject_participant_fault, and wire "
        "engine-specific 'crash', 'hang', and 'lost' hooks."
    ),
}


class TemplateLab:
    """Starter `LabUnderTest` implementation. Copy into `repos/<engine>/livelab/adapter/lab.py`
    and replace each method step by step while running `python3 -m conformance`."""

    name = "template"

    def __init__(self, receipts=None, **kwargs):
        self.receipts = receipts
        self.options = dict(kwargs)
        self.writer = R.ReceiptWriter(receipts) if receipts else None

    def status(self):
        raise NotImplementedError(STEPS["status"])

    def check(self, workload=None):
        raise NotImplementedError(STEPS["check"])

    def reset(self, kind="state"):
        raise NotImplementedError(STEPS["reset"])

    def patches(self):
        raise NotImplementedError(STEPS["patches"])

    def install(self, patch):
        raise NotImplementedError(STEPS["install"])

    def uninstall(self, id=None):
        raise NotImplementedError(STEPS["uninstall"])

    def fresh_check(self, patch, workload=None):
        raise NotImplementedError(STEPS["fresh_check"])

    def inject(self, fault, **kwargs):
        raise NotImplementedError(STEPS["inject"])

    def close(self):
        """Safe no-op teardown so the conformance runner's `finally: lab.close()` always succeeds."""
        return None
