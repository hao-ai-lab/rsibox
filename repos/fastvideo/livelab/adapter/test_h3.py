"""Golden tests of the shared conformance lab (`adapter.h3.shared_factory`) on the toy (CPU).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v

The real H3 lab takes 60-150 s to load, so `python3 -m conformance --lab adapter.h3:H3Lab` hands
every case the same warm lab. These tests show that sharing it does not leak one case's state
into the next. `h3_config` itself needs FastVideo and the weights: it runs only on GPU.
"""
import unittest

from adapter.h3 import shared_factory
from adapter.toy_lab import ToyFastVideoLab
from conformance import cases

BASE = {"video": [123, 177, 231, 34], "audio": [5, 4, 3, 2, 1, 0]}


class SharedLabTest(unittest.TestCase):
    def setUp(self):
        self.built = []

        def make(receipts):
            lab = ToyFastVideoLab(ranks=2, receipts=receipts)
            self.built.append(lab)
            return lab
        self.factory = shared_factory(make)

    def tearDown(self):
        for lab in self.built:
            lab.close()

    def test_a_case_that_leaves_an_edit_installed_hands_the_next_case_a_base_lab(self):
        """Input: install the visible edit through the shared lab and stop there, as a case that
        fails half-way does, then let the harness close it. Expect: the next case gets the same
        process, at base, with the base output. Proves: close() undoes left-over edits instead
        of shutting the lab down. Not: an undo that itself fails (that poisons; next test)."""
        a = self.factory()
        self.assertEqual(a.install(a.patches()["visible"])["outcome"], "committed")
        a.close()
        b = self.factory()
        self.assertIs(b.lab, a.lab)
        self.assertEqual(b.status()["state"], "base")
        self.assertEqual(b.check()["output"], BASE)

    def test_all_eleven_cases_pass_on_one_lab_and_a_poisoned_lab_is_respawned_between_cases(self):
        """Input: c0-c10 in order on one shared 2-rank toy lab (c8 poisons it four times).
        Expect: every case passes and the lab was built once. Proves: the shared lab gives the
        same verdicts as a new lab per case, so the real run can share one warm H3 engine.
        Not: the real engine's respawn cost (GPU only)."""
        rows = [cases.run(n, self.factory) for n in cases.CASES]
        self.assertEqual([(r["case"], r["result"]) for r in rows],
                         [(n, "pass") for n in cases.CASES], [r["error"] for r in rows])
        self.assertEqual(len(self.built), 1)


if __name__ == "__main__":
    unittest.main()
