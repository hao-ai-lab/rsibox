"""Golden tests for the engine onboarding skeleton (`template/livelab/`).

Verifies that:
1. Importing `adapter` resolves `core/livelab` (`kernel`, `forksafe`, `conformance`) without
   copying shared runtime files.
2. Running the real conformance runner against `adapter.lab:TemplateLab` executes all 11 cases
   (`c0`..`c10`), tears down cleanly, and fails `11/11` (`rc == 1`) with actionable step-by-step
   `NotImplementedError` guidance instead of crashing in import, `__init__`, or `close`.
3. Every `LabUnderTest` method raises a distinct, numbered `NotImplementedError` naming the step
   (`Step 1/6`..`Step 6/6`) and required return contract.

Run from the repo root:
    python3 -m unittest discover -s template/livelab/adapter -t template/livelab -v
"""
import contextlib
import io
import os
import unittest

import adapter
from adapter import lab as template_lab
from conformance import __main__ as conformance_cli
from conformance import cases as conformance_cases
import rsibox
import forksafe
from kernel import coordinator, session


class TemplateSkeletonTest(unittest.TestCase):

    def test_sys_path_bootstrap_resolves_shared_core_livelab_modules(self):
        self.assertTrue(os.path.isdir(adapter.LIVELAB))
        self.assertTrue(hasattr(coordinator, "StepGapCoordinator"))
        self.assertTrue(hasattr(session, "Session"))
        self.assertTrue(hasattr(forksafe, "install_all_guards"))
        self.assertTrue(hasattr(rsibox, "BaseLab"))
        self.assertTrue(hasattr(rsibox, "PackageOverlay"))

    def test_conformance_runner_executes_all_eleven_cases_and_fails_eleven_of_eleven(self):
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            exit_code = conformance_cli.main(["--lab", "adapter.lab:TemplateLab"])
        report_text = output_buffer.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("0/11 as expected", report_text)
        self.assertIn("NotImplementedError: Step 1/6 (status", report_text)
        for case_name in conformance_cases.CASES:
            self.assertIn(case_name, report_text)

    def test_each_unimplemented_method_gives_numbered_step_by_step_guidance(self):
        skeleton = template_lab.TemplateLab()
        self.addCleanup(skeleton.close)
        calls = [
            ("Step 1/6", lambda: skeleton.status()),
            ("Step 2/6", lambda: skeleton.check()),
            ("Step 3/6", lambda: skeleton.reset("state")),
            ("Step 4a/6", lambda: skeleton.patches()),
            ("Step 4b/6", lambda: skeleton.install({"files": {}})),
            ("Step 4c/6", lambda: skeleton.uninstall()),
            ("Step 5/6", lambda: skeleton.fresh_check(None)),
            ("Step 6/6", lambda: skeleton.inject("crash")),
        ]
        for expected_step, invoke in calls:
            with self.subTest(step=expected_step):
                with self.assertRaises(NotImplementedError) as ctx:
                    invoke()
                self.assertIn(expected_step, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
