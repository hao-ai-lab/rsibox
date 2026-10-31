"""The same cases for every engine. An engine ships only if it passes.

Each case returns measurements. A failed assertion fails the case; nothing is excused.
"""

import json
import os
import shutil
import time

from fastrsi import edits
from fastrsi.core import BASE, POISONED, Session, call


class CaseFailed(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CaseFailed(message)


class Suite:
    def __init__(self, engine_class, workdir, **options):
        self.workdir = os.path.abspath(workdir)
        self.engine = engine_class(os.path.join(self.workdir, "live"), **options)
        self.session = None
        self.spec = None
        self.base = None

    def edit(self, triple):
        path, old, new = triple
        return edits.replace(self.engine.root(), path, old, new)

    def run_exercise(self):
        return self.session.exercise(timeout=self.spec.exercise_timeout, **self.spec.exercise_args)

    def checked(self):
        """Reset, then run once. Reset makes the digest independent of earlier cases."""
        self.engine.reset()
        result = self.run_exercise()
        require(result is not None, f"exercise failed: {self.session.data['reason']}")
        return result["digest"]

    def install(self, triple, **fields):
        return self.session.install(self.edit(triple), **fields)

    def installed(self, triple, **fields):
        receipt = self.install(triple, **fields)
        require(receipt["outcome"] == "installed", f"install: {receipt}")
        return receipt

    def undone(self):
        receipt = self.session.undo()
        require(receipt["outcome"] == "undone", f"undo: {receipt}")

    def c1_noop(self):
        self.installed(self.spec.noop)
        digest = self.checked()
        self.undone()
        require(digest == self.base, f"no-op changed output: {digest} vs {self.base}")
        return {"digest": digest}

    def c2_visible(self):
        self.installed(self.spec.visible)
        digest = self.checked()
        hits = self.session.witness()
        self.undone()
        require(digest != self.base, "visible edit did not change output")
        counts = next(iter(hits.values()), [])
        running = sum(1 for count in counts if count)
        require(running >= self.spec.witness_processes,
                f"witness fired in {running}, need {self.spec.witness_processes}")
        return {"digest": digest, "witness": hits}

    def c3_fresh_equivalence(self):
        edit = self.edit(self.spec.visible)
        self.installed(self.spec.visible)
        live = self.checked()
        self.undone()
        other = self.engine.fresh(edit)
        try:
            other.reset()
            fresh = other.exercise(timeout=self.spec.exercise_timeout, **self.spec.exercise_args)
        finally:
            other.stop()
        require(live == fresh["digest"], f"live {live} differs from fresh {fresh['digest']}")
        return {"digest": live}

    def c4_uninstall(self):
        self.installed(self.spec.visible)
        self.undone()
        require(self.session.state == BASE, "not at base after undo")
        digest = self.checked()
        require(digest == self.base, f"output after undo {digest} vs base {self.base}")
        return {"digest": digest}

    def c5_failed_install(self):
        seen = {}
        edit = edits.replace_all(self.engine.root(), self.spec.multiple)
        for fault in ("prepare", "commit"):
            receipt = self.session.install(edit, fault=fault)
            require(receipt["outcome"] == "failed", f"{fault}: {receipt}")
            require(self.session.state == BASE, f"{fault}: state {self.session.state}")
            digest = self.checked()
            require(digest == self.base, f"{fault}: base not restored")
            seen[fault] = receipt["reason"]
        return seen

    def c6_refusal(self):
        codes = {}
        for code, triple in self.spec.refused.items():
            receipt = self.install(triple)
            require(receipt["outcome"] == "refused", f"{code}: {receipt}")
            require(receipt["code"] == code, f"{code}: refused as {receipt['code']}")
            codes[code] = receipt["reason"]
        require(self.checked() == self.base, "a refused edit changed output")
        return codes

    def c7_chain(self):
        reads = []
        for triple in self.spec.chain:
            self.installed(triple)
            reads.append(self.checked() != self.base)
            self.undone()
        digest = self.checked()
        require(digest == self.base, f"base moved after {len(self.spec.chain)} edits")
        return {"edits": len(self.spec.chain), "base_exact": True, "changed": sum(reads)}

    def c8_crash_and_hang(self):
        measured = {}
        for name, triple in (("crash", self.spec.crash), ("hang", self.spec.hang)):
            self.installed(triple)
            began = time.perf_counter()
            result = self.run_exercise()
            if result is not None:
                self.session.health()
            require(self.session.state == POISONED, f"{name}: state {self.session.state}")
            measured[f"{name}_detect_s"] = time.perf_counter() - began
            measured[f"{name}_reason"] = self.session.data["reason"]
            began = time.perf_counter()
            self.session.restart()
            digest = self.checked()
            measured[f"{name}_recover_s"] = time.perf_counter() - began
            require(digest == self.base, f"{name}: base not restored after restart")
        receipt = self.install(self.spec.visible, fault="exit")
        require(receipt["outcome"] == "poisoned", f"exit: {receipt}")
        self.session.restart()
        self.installed(self.spec.visible)
        call(self.session.data["targets"][0], "undo")
        require(self.session.health() == POISONED, "tampered process was not detected")
        measured["tamper_reason"] = self.session.data["reason"]
        self.session.restart()
        return measured

    def c9_soak(self, cycles=20):
        before = self.engine.stats()
        for _ in range(cycles):
            self.installed(self.spec.visible)
            self.undone()
        after = self.engine.stats()
        digest = self.checked()
        require(digest == self.base, f"base moved after {cycles} cycles")
        return {"cycles": cycles, "before": before, "after": after}

    CASES = ("c1_noop", "c2_visible", "c3_fresh_equivalence", "c4_uninstall",
             "c5_failed_install", "c6_refusal", "c7_chain", "c8_crash_and_hang", "c9_soak")

    def run(self, only=None):
        shutil.rmtree(self.workdir, ignore_errors=True)
        os.makedirs(self.workdir)
        self.engine.start()
        results = {}
        try:
            self.spec = self.engine.spec()
            self.session = Session(self.engine, os.path.join(self.workdir, "session"))
            self.engine.reset()
            self.base = self.run_exercise()["digest"]
            for name in self.CASES:
                if only and name not in only:
                    continue
                began = time.perf_counter()
                try:
                    detail = getattr(self, name)()
                    results[name] = {"passed": True, "seconds": time.perf_counter() - began,
                                     "detail": detail}
                except CaseFailed as error:
                    results[name] = {"passed": False, "seconds": time.perf_counter() - began,
                                     "error": str(error)}
                    if self.session.state != BASE:
                        self.session.restart() if self.session.state == POISONED \
                            else self.session.undo()
        finally:
            self.engine.stop()
        record = {"engine": self.engine.name, "time": time.time(),
                  "passed": all(r["passed"] for r in results.values()), "cases": results}
        with open(os.path.join(self.workdir, "conformance.json"), "w") as handle:
            json.dump(record, handle, indent=1)
        return record

