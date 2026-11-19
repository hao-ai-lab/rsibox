"""Exercise the script entry point with deterministic CPU-only transport replies."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.request

import walkthrough


class ScriptTest(unittest.TestCase):
    def run_script(self, fault=None, phases=None, alternate=False):
        script = os.environ.get("WALKTHROUGH_UNDER_TEST", walkthrough.__file__)
        starts = []
        current = None
        edit = None
        fingerprint_reads = 0
        undone = False
        fresh_sampler_requests = 0


        def launch(command, **kwargs):
            nonlocal current
            current = Path(kwargs["stdout"].name).stem.removeprefix("serve-")
            starts.append(current)
            return types.SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0)

        def request(req, **kwargs):
            nonlocal fresh_sampler_requests
            if isinstance(req, str):
                return types.SimpleNamespace(status=200)
            text = " Paris"
            if current == "fresh-sampler" or current == "live" and edit == "sampler":
                text = " Rome"
                if fault == "invisible":
                    text = " Paris"
                if fault == "disagreement" and current == "live":
                    text = " London"
            if current == "fresh-sampler":
                fresh_sampler_requests += 1
                if fault == "fresh-repeat" and fresh_sampler_requests == len(walkthrough.PROMPTS) + 1:
                    text = " London"
            if fault == "empty" and current == "fresh-base":
                text = " "
            if fault == "restoration" and current == "live" and fingerprint_reads >= 1 and edit is None and undone:
                text = " stale"
            return io.StringIO(json.dumps({"choices": [{"text": text}]}))

        def send(_, command):
            nonlocal edit, fingerprint_reads, undone
            op = command["op"]
            if op == "ping":
                return {"ok": True, "parallel": {"world": 2 if fault == "missing-rank" else 1}}
            if op == "txn":
                module = next(iter(command["plan"]["modules"]))
                edit = next(name for name, spec in walkthrough.EDITS.items()
                            if spec["module"] == module)
                return {"ok": fault != "commit", "txn": edit,
                        "outcome": "aborted" if fault == "commit" else "committed",
                        "receipt": {"actual_recipes": ["R1 rebuild" if fault == "recipe" else "R0 swap"]}}
            if op == "prove":
                reached = edit != "old_sampler" or fault == "legacy-reached"
                role = "core" if edit == "scheduler" and fault != "role" else "workers/rank0"
                method = walkthrough.METHODS[edit] if fault != "method" else "Other.method"
                key = f"{walkthrough.EDITS[edit]['module']}:{method}:100"
                return {"ok": reached, "receipt": {"witness": {role: {key: int(reached)}},
                        "findings": [] if reached else [{"check": "zero_witness"}]}}
            if op == "undo":
                edit = None
                undone = True
                return {"ok": fault != "undo", "receipt": {
                    "status": "partial" if fault == "undo" else "undone"}}
            if op == "fingerprint":
                fingerprint_reads += 1
                modules = command["modules"]
                unavailable = {"ok": False, "error": "module not loaded in this role"}
                owner = "core" if modules == [walkthrough.EDITS["scheduler"]["module"]] else "worker"
                code = {f"{spec['module']}:{walkthrough.METHODS[name]}:100": ["hash", 2]
                        for name, spec in walkthrough.EDITS.items() if spec["module"] in modules
                        and owner in spec["on"] and not (fault == "missing-method" and name == "old_sampler")}
                if owner == "core":
                    if fault == "fingerprint" and undone:
                        code[next(iter(code))] = ["hash", 3]
                    return {"ok": True, "core": {"ok": True, "code": code}, "workers": unavailable}
                worker = {"ok": True, "rank": 0, "code": code}
                return {"ok": True, "core": unavailable, "workers": {"ok": True, "ranks": [worker]}}
            return {"ok": fault != "at-base"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "supplier" / "vllm"
            sources = {}
            for name, spec in walkthrough.EDITS.items():
                source = Path(*spec["module"].split(".")).with_suffix(".py")
                original = spec["old_alt"] if alternate and name == "sampler" else spec["old"]
                sources[source] = original
                path = pkg.parent / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(original)
            argv = [script, "--model", "fixture", "--work", str(root / "work"),
                    "--out", str(root / "out.json")]
            if phases:
                argv += ["--phases", phases]
            stdout = io.StringIO()
            exit_reason = None
            modules = dict(sys.modules)
            modules["adapter"] = types.ModuleType("adapter")
            modules["adapter.client"] = types.SimpleNamespace(send=send)
            modules["kernel"] = types.SimpleNamespace(
                locate=types.SimpleNamespace(text_hash=lambda source: "hash"))
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "argv", argv),
                patch.object(sys, "path", sys.path[:]),
                patch.object(importlib.util, "find_spec",
                             return_value=types.SimpleNamespace(origin=str(pkg / "__init__.py"))),
                patch.object(subprocess, "Popen", side_effect=launch),
                patch.object(subprocess, "run", return_value=types.SimpleNamespace(stdout="0")),
                patch.object(os, "killpg"),
                patch.object(urllib.request, "urlopen", side_effect=request),
                contextlib.redirect_stdout(stdout),
            ):
                try:
                    runpy.run_path(script, run_name="__main__")
                except SystemExit as exc:
                    exit_reason = exc.code
            for source, original in sources.items():
                self.assertEqual((root / "work" / "src" / source).read_text(), original)
            records = [json.loads(line) for line in stdout.getvalue().splitlines()
                       if line.startswith("{")]
            return exit_reason, starts, records

    def test_full_run(self):
        for alternate in (False, True):
            with self.subTest(alternate=alternate):
                reason, starts, records = self.run_script(alternate=alternate)
                self.assertIsNone(reason)
                self.assertEqual(starts, ["warm", "fresh-base", "fresh-scheduler",
                                          "fresh-sampler", "fresh-old_sampler", "live"])
                self.assertEqual(records[-1]["live_equals_fresh"], dict.fromkeys(walkthrough.EDITS, True))

    def test_full_cli_rejects_bad_evidence(self):
        # The previous entry point exited successfully for disagreement and partial undo.
        cases = {
            "disagreement": "sampler: live output disagrees with fresh",
            "undo": "scheduler: undo failed",
            "invisible": "sampler: output did not visibly change",
            "empty": "fresh base: expected nonempty completion text",
            "fresh-repeat": "sampler fresh: repeated checks disagree",
            "commit": "scheduler: transaction did not commit",
            "recipe": "scheduler: expected R0 swap",
            "role": "Scheduler.schedule on core",
            "method": "Scheduler.schedule on core",
            "legacy-reached": "old_sampler: expected an unproven zero_witness result",
            "fingerprint": "scheduler: code fingerprint was not restored",
            "missing-method": "live base: could not read code fingerprint",
            "missing-rank": "live base: could not read code fingerprint",
            "at-base": "scheduler: adapter did not return to base",
            "restoration": "scheduler: undo did not restore base output",
        }
        for fault, message in cases.items():
            with self.subTest(fault=fault):
                reason, _, _ = self.run_script(fault)
                self.assertIn(message, reason or "")

    def test_live_only_is_diagnostic(self):
        reason, starts, records = self.run_script("undo", phases="live")
        self.assertIsNone(reason)
        self.assertEqual(starts, ["live"])
        self.assertIsNone(records[-1]["base"])
        self.assertEqual(records[-1]["live_equals_fresh"], {})
        self.assertTrue(any(record["step"] == "uninstall" and not record["ok"]
                            for record in records))
