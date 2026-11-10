#!/usr/bin/env python3
"""`python -m unittest discover` with a per-test JSON record.

Usage (same arguments as `unittest discover`, plus --json):
    python3 unittest_json.py --json out.json -s START [-t TOP] [-p PATTERN]

Behaves like `python3 -m unittest discover -v` run from the current directory
(sys.path[0] is the cwd, as with -m) and exits 0 only if every test passed or
skipped. The JSON file lists each test with its outcome, duration and skip reason.
"""
import argparse
import json
import os
import sys
import time
import unittest


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.records = []
        self._t0 = {}

    def startTest(self, test):
        self._t0[test.id()] = time.monotonic()
        super().startTest(test)

    def _rec(self, test, outcome, detail=""):
        t0 = self._t0.pop(test.id(), None)
        self.records.append({
            "id": test.id(),
            "outcome": outcome,
            "seconds": round(time.monotonic() - t0, 3) if t0 else None,
            "detail": detail[-2000:],
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._rec(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._rec(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        # Module import errors arrive as a _FailedTest / _ErrorHolder pseudo-test.
        self._rec(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._rec(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._rec(test, "xfailed")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._rec(test, "failed", "unexpected success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            kind = "failed" if issubclass(err[0], test.failureException) else "error"
            self._rec(subtest, kind, self._exc_info_to_string(err, test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("-s", "--start-directory", default=".")
    ap.add_argument("-t", "--top-level-directory", default=None)
    ap.add_argument("-p", "--pattern", default="test*.py")
    args = ap.parse_args()
    sys.path[0] = os.getcwd()  # match `python -m unittest`
    suite = unittest.TestLoader().discover(args.start_directory, args.pattern,
                                           args.top_level_directory)
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    with open(args.json, "w") as f:
        json.dump({"kind": "unittest", "tests": result.records}, f, indent=1)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
