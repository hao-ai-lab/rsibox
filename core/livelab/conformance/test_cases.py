"""The conformance cases test labs; this file tests the cases.

A case is only worth something if a lab without the property fails it. Each test below runs a
case against an honest fake lab, which must pass, and, where the test has one, a fake lab that
lacks exactly the property the case names, which must fail. The fakes are not live labs (their "code" is one
number per participant); they exercise the cases' own logic, nothing else.

Run from core/livelab:  python3 -m unittest discover -s conformance -t . -v
"""
import unittest

from . import cases


class FakeLab:
    """Two participants whose workload output is `factor * [1, 2, 3]`. An install sets the
    factor (and a code version, so a no-op edit still changes code identity); uninstall
    restores the previous one."""
    name = "fake"

    def __init__(self, receipts=None):
        self.factor, self.version, self.stack, self.n = 1, 0, [], 0

    def _fp(self):
        return f"{self.factor}.{self.version}"

    def status(self):
        return {"ok": True, "state": "edited" if self.stack else "base",
                "txns": [t for t, _ in self.stack], "fingerprint": self._fp(), "poison": None}

    def check(self, workload=None):
        out = {p: [self.factor * i for i in (1, 2, 3)] for p in ("p0", "p1")}
        live = bool(self.stack)
        return {"ok": True, "output": out, "proven": True if live else None,
                "witness": {p: {"fake:f:1": 3} for p in out} if live else {},
                "findings": [], "cost_s": 0.0}

    def reset(self, kind="state"):
        return {"ok": True, "reset": [kind], "cost_s": 0.0}

    def install(self, patch):
        factor = patch["files"]["m"]
        self.n += 1
        tid = f"t{self.n}"
        self.stack.append((tid, (self.factor, self.version)))
        self.factor, self.version = factor, self.version + 1
        return {"ok": True, "id": tid, "outcome": "committed", "codes": [],
                "mechanisms": ["fake"], "findings": [], "cost_s": 0.0}

    def uninstall(self, id=None):
        tid, (self.factor, self.version) = self.stack.pop()
        return {"ok": True, "undone": [tid], "state": self.status()["state"], "cost_s": 0.0}

    def patches(self):
        return {"noop": {"files": {"m": 1}}, "visible": {"files": {"m": 2}}}

    def fresh_check(self, patch, workload=None):
        f = patch["files"]["m"] if patch else 1
        return {"ok": True, "output": {p: [f * i for i in (1, 2, 3)] for p in ("p0", "p1")}}

    def inject(self, fault, **kw):
        return {"ok": True}

    def close(self):
        pass


class FlakyCheck(FakeLab):
    """The workload's output drifts between checks with no code change."""
    def check(self, workload=None):
        r = super().check(workload)
        self.n += 1
        r["output"] = {p: v + [self.n] for p, v in r["output"].items()}
        return r


class SilentInstall(FakeLab):
    """Install reports committed but the running code never changes."""
    def install(self, patch):
        r = super().install(patch)
        self.factor, self.version = self.stack[-1][1]
        return r


class LeakyUninstall(FakeLab):
    """Uninstall reports base but leaves the edit running."""
    def uninstall(self, id=None):
        tid, _ = self.stack.pop()
        return {"ok": True, "undone": [tid], "state": "base", "cost_s": 0.0}


class OneRankSilent(FakeLab):
    """One participant reports no witness calls at all (an empty map)."""
    def check(self, workload=None):
        reply = super().check(workload)
        if reply["witness"]:
            reply["witness"]["p1"] = {}
        return reply


class MultiRoleAbstain(FakeLab):
    """A worker-only edit on a multi-role engine: workers/rank0 fires while core abstains."""
    def check(self, workload=None):
        reply = super().check(workload)
        if reply["witness"]:
            reply["witness"] = {"workers/rank0": {"m": 3}, "core": {}}
        return reply


class MultiRoleOneWorkerSilent(FakeLab):
    """A worker edit where workers/rank1 is silent beside workers/rank0."""
    def check(self, workload=None):
        reply = super().check(workload)
        if reply["witness"]:
            reply["witness"] = {"workers/rank0": {"m": 3}, "workers/rank1": {}, "core": {}}
        return reply


class OneParticipant(FakeLab):
    """An honest lab with a single participant."""
    def check(self, workload=None):
        reply = super().check(workload)
        reply["witness"] = {"p0": reply["witness"]["p0"]} if reply["witness"] else {}
        return reply


class CasesCatchMissingProperties(unittest.TestCase):
    def assertCase(self, case, lab, expected):
        r = cases.run(case, lab)
        self.assertEqual(r["result"], expected, r["error"] or r["evidence"])

    def test_c0_repeatable_catches_drifting_output(self):
        self.assertCase("c0_repeatable", FakeLab, "pass")
        self.assertCase("c0_repeatable", FlakyCheck, "FAIL")

    def test_c0_repeatable_rejects_noop_reset_and_supports_stateful_reset(self):
        """Input: (a) a stateful lab whose `check()` advances volatile state and `reset()`
        restores it; (b) a stateful lab whose `reset()` is a no-op; (c) a lab whose `reset()`
        reports empty/missing `reset` state (`[]`, `None`). Expected: (a) passes; (b) and (c)
        FAIL."""
        class StatefulLab(FakeLab):
            def __init__(self, receipts=None):
                super().__init__(receipts)
                self.volatile = 0

            def check(self, workload=None):
                r = super().check(workload)
                r["output"] = {p: v + [self.volatile] for p, v in r["output"].items()}
                self.volatile += 1
                return r

            def reset(self, kind="state"):
                self.volatile = 0
                return {"ok": True, "reset": ["volatile"], "cost_s": 0.0}

        class StatefulNoOpReset(StatefulLab):
            def reset(self, kind="state"):
                return {"ok": True, "reset": ["volatile"], "cost_s": 0.0}

        class EmptyResetList(FakeLab):
            def reset(self, kind="state"):
                return {"ok": True, "reset": [], "cost_s": 0.0}

        class MissingResetField(FakeLab):
            def reset(self, kind="state"):
                return {"ok": True, "cost_s": 0.0}

        self.assertCase("c0_repeatable", StatefulLab, "pass")
        self.assertCase("c0_repeatable", StatefulNoOpReset, "FAIL")
        self.assertCase("c0_repeatable", EmptyResetList, "FAIL")
        self.assertCase("c0_repeatable", MissingResetField, "FAIL")

    def test_c1_noop_passes_an_identity_change_that_keeps_output(self):
        self.assertCase("c1_noop", FakeLab, "pass")

    def test_c2_visible_catches_an_install_that_changes_nothing(self):
        self.assertCase("c2_visible", FakeLab, "pass")
        self.assertCase("c2_visible", SilentInstall, "FAIL")

    def test_c2_visible_needs_calls_on_every_participant_however_many(self):
        """Input: a 2-participant lab where p1's witness map is empty, and an honest lab with
        one participant. Expected: FAIL, then pass. Pain point: c2 took the minimum over the
        calls it found, so an empty map passed, and it asked for at least 2 participants, so
        an honest single-participant lab failed. Not: a lab that leaves a participant out of
        the witness entirely (the lab's own `proven` covers that)."""
        self.assertCase("c2_visible", OneRankSilent, "FAIL")
        self.assertCase("c2_visible", MultiRoleOneWorkerSilent, "FAIL")
        self.assertCase("c2_visible", MultiRoleAbstain, "pass")
        self.assertCase("c2_visible", OneParticipant, "pass")

    def test_c3_fresh_equivalence_surfaces_fresh_check_error(self):
        """Input: a lab whose `fresh_check` returns `{"ok": False, "output": None, "error": ...}`
        on the visible edit or on the chain-of-3 edit. Expected: `c3_fresh_equivalence` fails
        and surfaces the error string instead of `('None', ...)`."""
        class _ChainFakeLab(FakeLab):
            def patches(self):
                return {
                    **super().patches(),
                    "chain": [{"files": {"m": 2}}, {"files": {"m": 3}}, {"files": {"m": 4}}],
                }

        class _FailVisibleFresh(_ChainFakeLab):
            def fresh_check(self, patch, workload=None):
                if patch is not None:
                    return {"ok": False, "output": None, "error": "boom-visible"}
                return super().fresh_check(patch, workload)

        class _FailChainFresh(_ChainFakeLab):
            def fresh_check(self, patch, workload=None):
                if patch and patch["files"]["m"] == 4:
                    return {"ok": False, "output": None, "error": "boom-chain"}
                return super().fresh_check(patch, workload)

        r_vis = cases.run("c3_fresh_equivalence", _FailVisibleFresh)
        self.assertEqual(r_vis["result"], "FAIL")
        self.assertIn("boom-visible", r_vis["error"])
        r_chain = cases.run("c3_fresh_equivalence", _FailChainFresh)
        self.assertEqual(r_chain["result"], "FAIL")
        self.assertIn("boom-chain", r_chain["error"])

    def test_c4_uninstall_catches_an_undo_that_leaves_the_edit(self):
        self.assertCase("c4_uninstall", FakeLab, "pass")
        self.assertCase("c4_uninstall", LeakyUninstall, "FAIL")


if __name__ == "__main__":
    unittest.main()
