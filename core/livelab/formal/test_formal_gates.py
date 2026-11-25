"""Unit tests for formal TLA+ anti-cheat gates, trace renderer, and runner helpers."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from formal import run_formal
from formal.gates import assumptions_check, check_anchors, check_lean_axioms, check_spec_pins, render_mermaid


class FormalGatesTest(unittest.TestCase):
    def test_positive_repo_gates(self) -> None:
        self.assertEqual(check_spec_pins.main([]), 0)
        self.assertEqual(check_anchors.main([]), 0)
        self.assertEqual(assumptions_check.main([]), 0)
        self.assertEqual(check_lean_axioms.main([]), 0)

    def test_check_spec_pins_catches_tampering_traversal_and_omitted_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_dir = root / "core" / "livelab" / "formal"
            spec_dir.mkdir(parents=True)
            spec = spec_dir / "Model.tla"
            spec.write_text("---- MODULE Model ----\nInit == TRUE\n====\n", encoding="utf-8")
            pins = root / "SPEC_PINS.sha256"
            check_spec_pins.update_pins(pins, root)
            self.assertEqual(check_spec_pins.main(["--pins", str(pins), "--repo", str(root)]), 0)

            # Tampering fails.
            spec.write_text("---- MODULE Model ----\nInit == FALSE\n====\n", encoding="utf-8")
            self.assertEqual(check_spec_pins.main(["--pins", str(pins), "--repo", str(root)]), 1)

            # Unpinned extra .tla on disk fails completeness check.
            check_spec_pins.update_pins(pins, root)
            (spec_dir / "Unpinned.tla").write_text("---- MODULE Unpinned ----\n====\n", encoding="utf-8")
            self.assertEqual(check_spec_pins.main(["--pins", str(pins), "--repo", str(root)]), 1)

            # Path traversal outside repo root fails even if hash matches an external file.
            outside = root.parent / f"outside_{root.name}.tla"
            try:
                outside.write_text("---- MODULE Outside ----\n====\n", encoding="utf-8")
                digest = check_spec_pins._sha256(outside)
                pins.write_text(f"{digest}  ../{outside.name}\n", encoding="utf-8")
                self.assertEqual(check_spec_pins.main(["--pins", str(pins), "--repo", str(root)]), 1)
            finally:
                outside.unlink(missing_ok=True)

            spec.unlink()
            self.assertEqual(check_spec_pins.main(["--pins", str(pins), "--repo", str(root)]), 1)

    def test_check_anchors_catches_broken_commented_nested_and_traversal_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Mod.tla").write_text(
                "---- MODULE Mod ----\n(* FakeOp == TRUE *)\n\\* LineFake == TRUE\nAct(x) == TRUE\n====\n",
                encoding="utf-8",
            )
            (root / "mod.py").write_text(
                "class C:\n    def ok(self):\n        return 1\n\ndef top():\n    def inner():\n        pass\n    return 2\n",
                encoding="utf-8",
            )
            tsv = root / "anchors.tsv"
            tsv.write_text(
                "# leading comment before header\n"
                "spec_file\tspec_symbol\tcode_file\tcode_symbol\tnotes\n"
                "Mod.tla\tAct\tmod.py\tC.ok\tvalid method\n"
                "Mod.tla\tAct\tmod.py\ttop\tvalid func\n",
                encoding="utf-8",
            )
            self.assertEqual(check_anchors.main(["--anchors", str(tsv), "--repo", str(root)]), 0)

            for bad_row in (
                "Mod.tla\tAct\tmod.py\tC.missing\tbroken method\n",
                "Mod.tla\tAct\tmod.py\tinner\tnested fn is not top-level\n",
                "Mod.tla\tFakeOp\tmod.py\ttop\tcommented TLA op\n",
                "Mod.tla\tLineFake\tmod.py\ttop\tline-commented TLA op\n",
                "../Mod.tla\tAct\tmod.py\ttop\tpath traversal\n",
            ):
                tsv.write_text(
                    "spec_file\tspec_symbol\tcode_file\tcode_symbol\tnotes\n" + bad_row,
                    encoding="utf-8",
                )
                self.assertEqual(check_anchors.main(["--anchors", str(tsv), "--repo", str(root)]), 1, bad_row)

    def test_assumptions_check_catches_vacuous_cheats_and_commented_cfg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tla = root / "Bad.tla"
            cfg = root / "Bad.cfg"
            tla.write_text(
                "---- MODULE Bad ----\nASSUME FALSE\nInit == TRUE\nNext == UNCHANGED <<x, y>>\n====\n",
                encoding="utf-8",
            )
            cfg.write_text("INIT Init\nNEXT Next\nINVARIANT Safe\n", encoding="utf-8")
            errs = assumptions_check.check_assumptions([root])
            self.assertTrue(any("ASSUME FALSE" in e for e in errs), errs)
            self.assertTrue(any("UNCHANGED" in e for e in errs), errs)
            self.assertEqual(assumptions_check.main(["--root", str(root)]), 1)

            tla.write_text("---- MODULE Bad ----\nSpec == TRUE\n====\n", encoding="utf-8")
            cfg.write_text(
                "SPECIFICATION Spec\nINVARIANTS\n  (* FakeInv *)\nCONSTRAINT StateBound\n",
                encoding="utf-8",
            )
            errs2 = assumptions_check.check_assumptions([root])
            self.assertTrue(any("disabled Spec" in e for e in errs2), errs2)
            self.assertTrue(any("empty INVARIANT" in e for e in errs2), errs2)

            # Line comment containing (* must NOT mask subsequent ASSUME FALSE.
            tla.write_text("---- MODULE Bad ----\n\\* (*\nASSUME FALSE\n\\* *)\n====\n", encoding="utf-8")
            cfg.write_text("SPECIFICATION Spec\nINVARIANT Safe\n", encoding="utf-8")
            errs3 = assumptions_check.check_assumptions([root])
            self.assertTrue(any("ASSUME FALSE" in e for e in errs3), errs3)

    def test_render_mermaid_emits_state_diagram(self) -> None:
        fixture = {
            "states": [
                {"index": 1, "action": "<Initial predicate>", "values": {"coord_state": "idle", "poisoned": False}},
                {"index": 2, "action": "<StartPrepare line 37>", "values": {"coord_state": "preparing", "poisoned": False}},
                {"index": 3, "action": "<AbortUnconfirmed line 63>", "values": {"coord_state": "recovery_required", "poisoned": True}},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(fixture), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = render_mermaid.main([str(trace_path)])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("stateDiagram-v2", out)
            self.assertIn("S1 --> S2: StartPrepare", out)
            self.assertIn("S2 --> S3: AbortUnconfirmed", out)

    def test_run_formal_cases_and_export_trace(self) -> None:
        all_cases = run_formal.cases()
        names = {c["name"] for c in all_cases}
        for base in (
            "pool", "coordinator", "swap", "shadow", "autoscaler",
            "transaction", "participant", "sequential-tx", "incarnation",
        ):
            self.assertIn(base, names)
        self.assertTrue(any(n.startswith("mutant-") for n in names))
        self.assertTrue(any(n.startswith("witness-") for n in names))

        sample_tlc = (
            "Error: Invariant NeverClosed is violated.\n"
            "State 1: <Initial predicate>\n"
            "/\\ closed = FALSE\n"
            "/\\ parked = 2\n"
            "State 2: <Close line 118>\n"
            "/\\ closed = TRUE\n"
            "/\\ parked = 0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.json"
            run_formal.export_trace(sample_tlc, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["states"]), 2)
            self.assertEqual(data["states"][1]["values"]["closed"], True)

    def _write_lean_fixture(self, root: Path, *, inv_body: str, lakefile: str | None = None, verify: str | None = None) -> None:
        (root / "RsiboxFormal").mkdir(parents=True, exist_ok=True)
        (root / "lean-toolchain").write_text("leanprover/lean4:v4.16.0\n", encoding="utf-8")
        (root / "lakefile.toml").write_text(
            lakefile or 'name = "rsibox_formal"\nversion = "0.1.0"\n[[lean_lib]]\nname = "RsiboxFormal"\n',
            encoding="utf-8",
        )
        (root / "RsiboxFormal.lean").write_text("import RsiboxFormal.Invariants\n", encoding="utf-8")
        (root / "RsiboxFormal" / "Invariants.lean").write_text(inv_body, encoding="utf-8")
        (root / "VerifyAxioms.lean").write_text(
            verify if verify is not None else "import RsiboxFormal\n#print axioms RsiboxFormal.ok\n",
            encoding="utf-8",
        )

    def test_check_lean_axioms_ignores_nested_and_line_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = (
                "namespace RsiboxFormal\n"
                "-- sorry admit axiom native_decide partial def @[extern \"c\"]\n"
                "/- outer /- inner sorry axiom Lean.ofReduceBool -/ still commented sorry -/\n"
                "theorem ok : True := trivial\n"
                "end RsiboxFormal\n"
            )
            self._write_lean_fixture(root, inv_body=body)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = check_lean_axioms.main(["--root", str(root)])
            self.assertEqual(rc, 0)

    def test_check_lean_axioms_rejects_cheats_mathlib_and_missing_coverage(self) -> None:
        cheats = [
            ("sorry", "theorem ok : True := by sorry\n"),
            ("admit", "theorem ok : True := by admit\n"),
            ("axiom", "axiom fake : False\ntheorem ok : True := trivial\n"),
            ("opaque", "opaque hidden : Nat\ntheorem ok : True := trivial\n"),
            ("partial def", "partial def loop : Nat := loop\ntheorem ok : True := trivial\n"),
            ("unsafe", "unsafe def bad : Nat := 0\ntheorem ok : True := trivial\n"),
            ("native_decide", "theorem ok : True := by native_decide\n"),
            ("Lean.ofReduceBool", "theorem ok : Lean.ofReduceBool = Lean.ofReduceBool := rfl\n"),
            ("Lean.trustCompiler", "theorem ok : Lean.trustCompiler = Lean.trustCompiler := rfl\n"),
            ("implemented_by", "@[implemented_by bad] def f : Nat := 0\ntheorem ok : True := trivial\n"),
            ("@[extern", "@[extern \"c_fn\"] opaque cfn : Nat\ntheorem ok : True := trivial\n"),
        ]
        for label, snippet in cheats:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_lean_fixture(root, inv_body=f"namespace RsiboxFormal\n{snippet}end RsiboxFormal\n")
                errs = check_lean_axioms.check_lean_axioms(root)
                self.assertTrue(any(label in e for e in errs), (label, errs))
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(check_lean_axioms.main(["--root", str(root)]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_lean_fixture(
                root,
                inv_body="namespace RsiboxFormal\ntheorem ok : True := trivial\nend RsiboxFormal\n",
                lakefile='name = "bad"\n[[require]]\nname = "mathlib"\n',
            )
            errs = check_lean_axioms.check_lean_axioms(root)
            self.assertTrue(any("mathlib" in e for e in errs), errs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_lean_fixture(
                root,
                inv_body="namespace RsiboxFormal\ntheorem ok : True := trivial\nend RsiboxFormal\n",
                verify="import RsiboxFormal\n-- #print axioms RsiboxFormal.ok\n",
            )
            errs = check_lean_axioms.check_lean_axioms(root)
            self.assertTrue(any("missing '#print axioms'" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
