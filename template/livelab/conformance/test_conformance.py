"""The conformance cases as unit tests on the toy lab (CPU).

Run from template/livelab:  python3 -m unittest conformance.test_conformance -v
Each test runs one case of cases.py on a new ToyLab; its docstring is the case's.
"""
import unittest

from . import cases
from .toy_lab import ToyLab


class ToyConformance(unittest.TestCase):
    pass


def _make(name):
    def test(self):
        r = cases.run(name, ToyLab)
        if r["xfail_reason"]:
            self.skipTest(f"xfail: {r['xfail_reason']} ({r['result']})")
        self.assertEqual(r["result"], "pass", r["error"])
    test.__doc__ = cases.CASES[name].__doc__
    return test


for _name in cases.CASES:
    setattr(ToyConformance, "test_" + _name, _make(_name))
