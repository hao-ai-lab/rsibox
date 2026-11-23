"""Packaging tests for root `pyproject.toml` (`rsibox`).

Verifies PEP 621 metadata, `rsibox = "rsibox.cli:main"` entry point,
complete coverage of `core/livelab` runtime modules, exclusion of all
`test_*.py` / `*_test.py` files and survey fixtures from built artifacts,
and clean isolated execution of the unpacked wheel.

Run from the repository root:
    python3 -m unittest discover -s scripts -p 'test_*.py' -v
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
LIVELAB_DIR = REPO_ROOT / "core" / "livelab"

RUNTIME_ROOT_MODULES = ("forksafe", "livestream", "ray_participants")
RUNTIME_ROOT_PACKAGES = ("conformance", "rsibox", "kernel", "rsibox", "survey")
EXPECTED_PACKAGES = ("conformance", "rsibox", "kernel", "kernel.toy", "rsibox", "survey")
EXPECTED_TOP_LEVEL = sorted((*RUNTIME_ROOT_MODULES, *RUNTIME_ROOT_PACKAGES))

_HAS_SETUPTOOLS = importlib.util.find_spec("setuptools") is not None
_HAS_BDIST_WHEEL = (
    _HAS_SETUPTOOLS
    and (
        importlib.util.find_spec("setuptools.command.bdist_wheel") is not None
        or importlib.util.find_spec("wheel") is not None
    )
)


def _is_test_filename(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py")


def _discover_livelab_modules(livelab_dir: Path = LIVELAB_DIR) -> tuple[set[str], list[Path]]:
    """Return `(expected_py_modules, test_files)` from `core/livelab` on disk."""
    expected: set[str] = set()
    for mod in RUNTIME_ROOT_MODULES:
        if (livelab_dir / f"{mod}.py").is_file():
            expected.add(mod)

    for pkg in RUNTIME_ROOT_PACKAGES:
        for path in sorted((livelab_dir / pkg).rglob("*.py")):
            rel = path.relative_to(livelab_dir)
            if "fixture" in rel.parts or _is_test_filename(rel.name) or rel.name == "__init__.py":
                continue
            expected.add(".".join(rel.with_suffix("").parts))

    test_files = sorted(
        path.relative_to(livelab_dir)
        for path in livelab_dir.rglob("*.py")
        if _is_test_filename(path.name)
    )
    return expected, test_files


def _expected_wheel_python_files(livelab_dir: Path = LIVELAB_DIR) -> set[str]:
    """Return expected wheel-relative `.py` paths for runtime modules and packages."""
    expected_mods, _ = _discover_livelab_modules(livelab_dir)
    files = {f"{mod.replace('.', '/')}.py" for mod in expected_mods}
    for pkg in EXPECTED_PACKAGES:
        files.add(f"{pkg.replace('.', '/')}/__init__.py")
    return files


def _copy_packaging_tree(dest: Path, *, with_synthetic_tests: bool = False) -> Path:
    """Copy minimal packaging inputs to `dest` without touching the repo tree."""
    proj = dest / "repo"
    proj.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / name, proj / name)
    shutil.copytree(
        LIVELAB_DIR,
        proj / "core" / "livelab",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    if with_synthetic_tests:
        for fname in ("test_synthetic_packaging.py", "synthetic_extra_test.py"):
            (proj / "core" / "livelab" / "kernel" / fname).write_text(
                "raise AssertionError('must not package')\n", encoding="utf-8",
            )
    return proj


def _run_python(code: str, *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc_env = os.environ.copy() if env is None else env
    return subprocess.run(
        [sys.executable, "-W", "default", "-W", "error::ResourceWarning", "-c", code],
        cwd=str(cwd),
        env=proc_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_metadata_and_cli_entry_point(self):
        """Input: root `pyproject.toml`. Expected: valid PEP 621 metadata, zero deps, `rsibox` CLI script."""
        with PYPROJECT_PATH.open("rb") as fh:
            data = tomllib.load(fh)

        build_sys = data["build-system"]
        self.assertEqual(build_sys["requires"], ["setuptools>=64"])
        self.assertEqual(build_sys["build-backend"], "setuptools.build_meta")

        proj = data["project"]
        self.assertEqual(proj["name"], "rsibox")
        self.assertEqual(proj["version"], "0.1.0")
        self.assertTrue(proj["description"])
        self.assertEqual(proj["readme"], "README.md")
        self.assertEqual(proj["license"], {"text": "Apache-2.0"})
        self.assertEqual(proj["requires-python"], ">=3.11")
        self.assertEqual(proj["dependencies"], [])

        scripts = proj["scripts"]
        self.assertEqual(scripts, {"rsibox": "rsibox.cli:main", "rsibox": "rsibox.cli:main"})

        for cli_path in (LIVELAB_DIR / "rsibox" / "cli.py", LIVELAB_DIR / "rsibox" / "cli.py"):
            cli_ast = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
            top_defs = {node.name for node in cli_ast.body if isinstance(node, ast.FunctionDef)}
            self.assertIn("main", top_defs, cli_path)

        setuptools_cfg = data["tool"]["setuptools"]
        self.assertEqual(setuptools_cfg["package-dir"], {"": "core/livelab"})
        self.assertEqual(setuptools_cfg["packages"], [])
        self.assertIs(setuptools_cfg["include-package-data"], False)

    def test_py_modules_excludes_tests_and_covers_all_runtime_modules(self):
        """Input: `pyproject.toml` `py-modules` vs `core/livelab`. Expected: 1:1 runtime coverage, 0 tests."""
        with PYPROJECT_PATH.open("rb") as fh:
            data = tomllib.load(fh)

        py_modules = data["tool"]["setuptools"]["py-modules"]
        self.assertEqual(len(py_modules), len(set(py_modules)), "duplicate entries in py-modules")

        expected_runtime, test_files = _discover_livelab_modules()
        self.assertGreaterEqual(len(test_files), 15, "expected real test_*.py files under core/livelab")

        bad_entries = [
            mod for mod in py_modules
            if any(
                part.startswith("test_") or part.endswith("_test") or part == "__init__" or part == "fixture"
                for part in mod.split(".")
            )
        ]
        self.assertEqual(bad_entries, [])
        self.assertEqual(set(py_modules), expected_runtime)

        for pkg in EXPECTED_PACKAGES:
            self.assertTrue((LIVELAB_DIR / pkg.replace(".", "/") / "__init__.py").is_file())
            direct = [m for m in py_modules if m.rpartition(".")[0] == pkg]
            self.assertTrue(direct, f"{pkg} needs at least one non-__init__ module in py-modules")


@unittest.skipUnless(_HAS_SETUPTOOLS, "setuptools is not installed")
class SetuptoolsBuildTests(unittest.TestCase):
    def test_setuptools_build_py_excludes_tests_even_with_synthetic_test_files(self):
        """Input: isolated repo copy with extra `test_*.py` and `*_test.py`. Expected: `build_py` excludes all."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proj = _copy_packaging_tree(tmp_path, with_synthetic_tests=True)
            build_lib = tmp_path / "build_lib"
            script = (
                "import json, pathlib\n"
                "from setuptools.dist import Distribution\n"
                "from setuptools.config import pyprojecttoml\n"
                f"proj = pathlib.Path({str(proj)!r})\n"
                f"build_lib = {str(build_lib)!r}\n"
                "dist = Distribution({'src_root': str(proj), 'script_name': 'setup.py'})\n"
                "pyprojecttoml.apply_configuration(dist, str(proj / 'pyproject.toml'))\n"
                "cmd = dist.get_command_obj('build_py')\n"
                "cmd.build_lib = build_lib\n"
                "cmd.ensure_finalized()\n"
                "found = [(pkg, mod) for pkg, mod, _ in cmd.find_all_modules()]\n"
                "cmd.run()\n"
                "print(json.dumps({'found': found}))\n"
            )
            proc = _run_python(script, cwd=proj)
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            found = payload["found"]
            self.assertTrue(found)
            test_mods = [
                f"{pkg}.{mod}" if pkg else mod
                for pkg, mod in found
                if mod.startswith("test_") or mod.endswith("_test")
            ]
            self.assertEqual(test_mods, [])

            built_files = {
                path.relative_to(build_lib).as_posix()
                for path in build_lib.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                [f for f in sorted(built_files) if _is_test_filename(Path(f).name)],
                [],
            )
            self.assertEqual(built_files, _expected_wheel_python_files())

    def test_sdist_excludes_tests_and_preserves_runtime(self):
        """Input: built sdist in temp dir with extra test files. Expected: 0 tests, all runtime OK."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proj = _copy_packaging_tree(tmp_path, with_synthetic_tests=True)
            dist_dir = tmp_path / "dist"
            dist_dir.mkdir()
            sdist_script = (
                "from setuptools import build_meta\n"
                f"print(build_meta.build_sdist({str(dist_dir)!r}))\n"
            )
            proc = _run_python(sdist_script, cwd=proj)
            sdist_path = dist_dir / proc.stdout.strip().splitlines()[-1]

            with tarfile.open(sdist_path, "r:gz") as tf:
                sdist_names = tf.getnames()
            self.assertEqual(
                [n for n in sdist_names if _is_test_filename(Path(n).name) or "/fixture/" in n],
                [],
            )
            expected_py = _expected_wheel_python_files()
            sdist_py = {
                n.split("/core/livelab/", 1)[1]
                for n in sdist_names
                if "/core/livelab/" in n and n.endswith(".py")
            }
            self.assertEqual(sdist_py, expected_py)
            self.assertTrue(any(n.endswith("/pyproject.toml") for n in sdist_names))
            self.assertTrue(any(n.endswith("/README.md") for n in sdist_names))
            self.assertTrue(any(n.endswith("/LICENSE") for n in sdist_names))

    @unittest.skipUnless(_HAS_BDIST_WHEEL, "bdist_wheel is not available")
    def test_wheel_and_editable_exclude_tests_and_run_rsibox_cli(self):
        """Input: built wheel and editable wheel in temp dir. Expected: 0 tests, isolated CLI passes."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proj = _copy_packaging_tree(tmp_path, with_synthetic_tests=True)
            dist_dir = tmp_path / "dist"
            dist_dir.mkdir()
            build_script = (
                "import json, pathlib\n"
                "from setuptools import build_meta\n"
                f"out = {str(dist_dir)!r}\n"
                "wheel_name = build_meta.build_wheel(out)\n"
                "editable_name = build_meta.build_editable(out)\n"
                "strict_dir = pathlib.Path(out) / 'strict'\n"
                "strict_dir.mkdir()\n"
                "strict_name = build_meta.build_editable(\n"
                "    str(strict_dir), config_settings={'editable_mode': 'strict'},\n"
                ")\n"
                "print(json.dumps({\n"
                "    'wheel': wheel_name,\n"
                "    'editable': editable_name,\n"
                "    'strict': strict_name,\n"
                "}))\n"
            )
            proc = _run_python(build_script, cwd=proj)
            artifacts = json.loads(proc.stdout.strip().splitlines()[-1])
            expected_py = _expected_wheel_python_files()

            wheel_path = dist_dir / artifacts["wheel"]
            with zipfile.ZipFile(wheel_path) as zf:
                wheel_names = sorted(zf.namelist())
                self.assertEqual(
                    [n for n in wheel_names if _is_test_filename(Path(n).name) or "fixture/" in n],
                    [],
                )
                wheel_py = {n for n in wheel_names if n.endswith(".py")}
                self.assertEqual(wheel_py, expected_py)
                top_level = zf.read("rsibox-0.1.0.dist-info/top_level.txt").decode("utf-8").splitlines()
                self.assertEqual(sorted(top_level), EXPECTED_TOP_LEVEL)
                entry_points = zf.read("rsibox-0.1.0.dist-info/entry_points.txt").decode("utf-8")
                self.assertIn("rsibox = rsibox.cli:main", entry_points)
                self.assertIn("rsibox = rsibox.cli:main", entry_points)
                # setuptools >= 77 (PEP 639) writes dist-info/licenses/LICENSE;
                # setuptools 64..76 writes dist-info/LICENSE directly.
                self.assertTrue(
                    "rsibox-0.1.0.dist-info/licenses/LICENSE" in wheel_names
                    or "rsibox-0.1.0.dist-info/LICENSE" in wheel_names,
                    f"LICENSE missing from wheel dist-info: {wheel_names}",
                )

            editable_path = dist_dir / artifacts["editable"]
            with zipfile.ZipFile(editable_path) as zf:
                pth_names = [n for n in zf.namelist() if n.endswith(".pth")]
                self.assertEqual(len(pth_names), 1)
                pth_target = Path(zf.read(pth_names[0]).decode("utf-8").strip()).resolve()
                self.assertEqual(pth_target, (proj / "core" / "livelab").resolve())
                ep = zf.read("rsibox-0.1.0.dist-info/entry_points.txt").decode("utf-8")
                self.assertIn("rsibox = rsibox.cli:main", ep)
                self.assertIn("rsibox = rsibox.cli:main", ep)

            strict_dir = proj / "build"
            strict_trees = [p for p in strict_dir.glob("__editable__.*") if p.is_dir()]
            self.assertEqual(len(strict_trees), 1)
            strict_py = {
                p.relative_to(strict_trees[0]).as_posix()
                for p in strict_trees[0].rglob("*.py")
            }
            self.assertEqual(
                [n for n in sorted(strict_py) if _is_test_filename(Path(n).name)],
                [],
            )
            self.assertEqual(strict_py, expected_py)

            # Unpack the wheel into a fresh site dir and verify imports, doctor, and toy conformance.
            site_dir = tmp_path / "site"
            site_dir.mkdir()
            with zipfile.ZipFile(wheel_path) as zf:
                zf.extractall(site_dir)

            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(site_dir),
                "CUDA_VISIBLE_DEVICES": "",
            }
            import_check = (
                "import pathlib, kernel, kernel.toy, conformance, survey, "
                "rsibox, rsibox, forksafe, livestream, ray_participants\n"
                f"site = pathlib.Path({str(site_dir)!r}).resolve()\n"
                "for mod in (kernel, kernel.toy, conformance, survey, rsibox, rsibox, forksafe, livestream, ray_participants):\n"
                "    assert pathlib.Path(mod.__file__).resolve().is_relative_to(site), mod.__file__\n"
            )
            _run_python(import_check, cwd=tmp_path, env=clean_env)

            doc_proc = subprocess.run(
                [sys.executable, "-m", "rsibox", "doctor", "--repo-root", str(REPO_ROOT), "--json"],
                cwd=str(tmp_path),
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            doc = json.loads(doc_proc.stdout)
            self.assertTrue(doc["ok"], doc)
            self.assertEqual(Path(doc["livelab_root"]).resolve(), site_dir.resolve())

            conf_proc = subprocess.run(
                [sys.executable, "-m", "rsibox", "conformance", "toy", "--only", "c0_repeatable,c2_visible"],
                cwd=str(tmp_path),
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            self.assertIn("2/2 as expected", conf_proc.stdout)


if __name__ == "__main__":
    unittest.main()
