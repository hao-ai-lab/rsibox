"""TemplateLab: runnable LabUnderTest skeleton for onboarding a new engine.

Running `python3 -m conformance --lab adapter.lab:TemplateLab` from `core/livelab` instantiates
and closes this class cleanly, failing all 11/11 conformance cases (c0-c10) with numbered,
actionable `NotImplementedError` messages that guide an engine author from 0/11 to 11/11.
"""
import adapter  # noqa: F401  # puts core/livelab on sys.path
from kernel import receipt as R

STEPS = {
    "status": (
        "Step 1/6 (status -> unlocks base checks in c0-c10): build a Coordinator or "
        "StepGapCoordinator over your worker participants (kernel.coordinator / ray_participants), "
        "wrap it in kernel.session.Session(coordinator, modules), and return self.session.status() "
        "with keys {'ok', 'state', 'txns', 'fingerprint', 'poison'}."
    ),
    "check": (
        "Step 2/6 (check -> unlocks c0_repeatable with Step 3): run one deterministic workload "
        "pass at a quiesced step gap, call self.coordinator.prove(txn) for each txn in "
        "self.session.stack, and return {'ok', 'output', 'proven', 'witness', 'findings', 'cost_s'}."
    ),
    "reset": (
        "Step 3/6 (reset -> unlocks c0_repeatable and c8 respawn): for kind='state', drain "
        "in-flight requests/buffers, clear volatile caches, and restore initial weights if mutated; "
        "for kind='respawn', restart the worker group from warm caches and call "
        "self.session.recovered(self.coordinator). Return {'ok', 'reset', 'cost_s'}."
    ),
    "patches": (
        "Step 4a/6 (patches -> unlocks c1, c2, c4, c6, c7, c9, c10): return real source edits "
        "against your engine modules with keys {'noop', 'visible', 'chain' (3 stacked edits), "
        "'multi' (>=3 functions across >=2 modules), 'unreached', 'refused': {'syntax', "
        "'closure', 'module_state', 'empty'}}."
    ),
    "install": (
        "Step 4b/6 (install -> unlocks c1, c2, c4, c6, c7, c9): build a kernel plan from "
        "patch['files'] (setting each module's 'source' and 'base_hash'), call "
        "self.session.install(plan), and return {'ok', 'id', 'outcome', 'codes', 'mechanisms', "
        "'findings', 'cost_s'}."
    ),
    "uninstall": (
        "Step 4c/6 (uninstall -> unlocks c1, c2, c4, c7, c9): call self.session.uninstall(id) "
        "to undo id and any newer transactions in reverse order, pop matching live-source stack "
        "entries, and return {'ok', 'undone', 'state', 'cost_s'}."
    ),
    "fresh_check": (
        "Step 5/6 (fresh_check -> unlocks c3_fresh_equivalence and c10_not_proven): run the "
        "workload in a fresh Python subprocess over base module files overlaid with "
        "(patch or {}).get('files'), and return {'ok': True, 'output': output}."
    ),
    "inject": (
        "Step 6/6 (inject -> unlocks c5_failed_install and c8_crash_and_hang): wire conformance "
        "fault hooks for 'fail_apply' (after=n swaps), 'fail_commit', 'crash', 'hang', 'lost', "
        "and 'tamper' (unjournaled __code__ mutation), returning {'ok': True, 'fault': fault}."
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
