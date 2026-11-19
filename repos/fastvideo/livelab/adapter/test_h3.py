"""Golden tests of the shared conformance lab (`adapter.h3.shared_factory`) on the toy (CPU).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v

The real H3 lab takes 60-150 s to load, so `python3 -m conformance --lab adapter.h3:H3Lab` hands
every case the same warm lab. These tests show that sharing it does not leak one case's state
into the next. `h3_config` itself needs FastVideo and the weights: it runs only on GPU.
"""
import inspect
import os
import unittest
from unittest import mock

from adapter import h3
from adapter.h3 import H3Lab, shared_factory
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


class H3LabClassTest(unittest.TestCase):
    def test_h3lab_is_concrete_class_with_all_lab_methods_and_launch_env(self):
        """Input: `adapter.h3:H3Lab` inspected directly and driven with a monkeypatched `_make_h3`.
        Expected: `H3Lab` is a concrete class exposing all 9 `LabUnderTest` methods directly,
        reuses one warm lab across instantiations, and maps CLI launch options to env vars."""
        self.assertTrue(inspect.isclass(H3Lab))
        for method in ("install", "check", "reset", "uninstall", "status",
                       "patches", "fresh_check", "inject", "close"):
            self.assertTrue(callable(getattr(H3Lab, method, None)), method)

        env = H3Lab.launch_environment([
            "--", "--model", "/tmp/m", "--vae", "/tmp/v.safetensors", "--gpus", "2",
            "--check-timeout", "900", "--zygote", "--holder",
        ])
        self.assertEqual(env["FASTH3_MODEL"], os.path.abspath("/tmp/m"))
        self.assertEqual(env["FASTVIDEO_H3_VAE_NVFP4_DECODER"], os.path.abspath("/tmp/v.safetensors"))
        self.assertEqual(env["FASTRSI_H3_GPUS"], "2")
        self.assertEqual(env["FASTRSI_H3_CHECK_TIMEOUT"], "900.0")
        self.assertEqual(env["FASTRSI_H3_ZYGOTE"], "1")
        self.assertEqual(env["FASTRSI_H3_HOLDER"], "1")

        plain_env = H3Lab.launch_environment(["--gpus", "4"])
        self.assertEqual(plain_env, {"FASTRSI_H3_GPUS": "4"})

        captured = {}

        def fake_lab(cfg, **kw):
            captured["cfg"] = cfg
            captured["kw"] = kw
            return "fake-lab"

        from adapter import lab as lab_mod
        with mock.patch.dict(os.environ, {"FASTRSI_H3_ZYGOTE": "1", "FASTRSI_H3_HOLDER": "true",
                                          "FASTRSI_H3_CHECK_TIMEOUT": "750"}, clear=False), \
             mock.patch.object(h3, "h3_config", return_value={"generator": {}}), \
             mock.patch.object(lab_mod, "FastVideoLab", side_effect=fake_lab):
            self.assertEqual(h3._make_h3("r.jsonl"), "fake-lab")
        self.assertEqual(captured["kw"]["receipts"], "r.jsonl")
        self.assertEqual(captured["kw"]["check_timeout"], 750.0)
        self.assertIs(captured["kw"]["zygote"], True)
        self.assertIs(captured["kw"]["holder"], True)

        with mock.patch.dict(os.environ, {"FASTRSI_H3_ZYGOTE": "0", "FASTRSI_H3_HOLDER": ""}, clear=False), \
             mock.patch.object(h3, "h3_config", return_value={"generator": {}}), \
             mock.patch.object(lab_mod, "FastVideoLab", side_effect=fake_lab):
            h3._make_h3(None)
        self.assertIs(captured["kw"]["zygote"], False)
        self.assertIs(captured["kw"]["holder"], False)

        built = []

        def fake_make(receipts):
            lab = ToyFastVideoLab(ranks=2, receipts=receipts)
            built.append(lab)
            return lab

        saved = dict(H3Lab._shared)
        H3Lab._shared.clear()
        try:
            with mock.patch.object(h3, "_make_h3", side_effect=fake_make), \
                 mock.patch.object(h3.atexit, "register"):
                a = H3Lab()
                b = H3Lab()
                self.assertIs(a.lab, b.lab)
                self.assertEqual(len(built), 1)
                self.assertEqual(a.check()["output"], BASE)
                a.close()
        finally:
            for lab in built:
                lab.close()
            H3Lab._shared.clear()
            H3Lab._shared.update(saved)


if __name__ == "__main__":
    unittest.main()
