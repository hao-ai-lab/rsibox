"""AST-based architecture and boundary guards for RSIBox.

Pure AST + stdlib static checks (zero runtime imports of engine code or torch):
1. `core/livelab/` (excluding `survey/fixture`) never imports `repos` or engine-private adapters.
2. `core/livelab/kernel/` non-test runtime modules never import `rsibox`.
3. `core/livelab/kernel/` leaf modules never import orchestrator modules.
4. `repos/<engine>/` adapters never import `repos.<other>`.
5. `core/livelab/kernel/` and `core/livelab/rsibox/warm/` non-test runtime files contain no
   hardcoded loopback/localhost URLs or private LAN IP literals.
"""
from __future__ import annotations

import ast
import ipaddress
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVELAB_DIR = REPO_ROOT / "core" / "livelab"
KERNEL_DIR = LIVELAB_DIR / "kernel"
WARM_DIR = LIVELAB_DIR / "rsibox" / "warm"
REPOS_DIR = REPO_ROOT / "repos"

_FORBIDDEN_CORE_PACKAGES = frozenset({"repos", "adapter", "rsibox_megatron"})
# Existing test-only exception: tests MegatronLab.launch_environment CLI env mapping.
_CORE_TEST_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "core/livelab/rsibox/test_lab_cli.py": frozenset({"rsibox_megatron"}),
}

_KERNEL_LEAF_MODULES = frozenset({"findings", "identity", "receipt", "locate"})
_KERNEL_ORCHESTRATOR_MODULES = frozenset({"coordinator", "session", "toy_lab"})

_LOOPBACK_URL_RE = re.compile(
    r"(?:https?|wss?)://(?:(?:127\.0\.0\.1|localhost)\b|\[::1\])", re.IGNORECASE
)
_IPV4_RE = re.compile(r"(?<![0-9.])(\d{1,3}(?:\.\d{1,3}){3})(?![0-9.])")
_RFC5737_NETS = (
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
)


def _is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _iter_py_files(
    root: Path, *, exclude_tests: bool = False, exclude_fixture: bool = False
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if exclude_fixture and "fixture" in rel.parts:
            continue
        if exclude_tests and _is_test_file(path):
            continue
        files.append(path)
    return files


def _iter_imports(path: Path) -> list[tuple[int, int, str, list[str]]]:
    """Return `(lineno, level, module, names)` for every import statement in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    items: list[tuple[int, int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                items.append((node.lineno, 0, alias.name, []))
        elif isinstance(node, ast.ImportFrom):
            items.append((node.lineno, node.level, node.module or "", [a.name for a in node.names]))
    return items


def _resolve_relative_module(path: Path, level: int, module: str) -> str:
    """Resolve a relative `ImportFrom` under `REPO_ROOT` to its dotted module path."""
    pkg_parts = list(path.relative_to(REPO_ROOT).parent.parts)
    up = level - 1
    if up > len(pkg_parts):
        return module
    base_parts = pkg_parts[: len(pkg_parts) - up]
    return ".".join([*base_parts, *(module.split(".") if module else [])])


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """Collect `id()` of `ast.Constant` docstring nodes on modules, classes, and functions."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    ids.add(id(val))
    return ids


class ArchitectureGuardsTests(unittest.TestCase):
    def test_core_never_imports_repos(self) -> None:
        """`core/livelab/` (excluding `survey/fixture`) never imports `repos` or engine adapters."""
        for rel_str in _CORE_TEST_IMPORT_ALLOWLIST:
            self.assertTrue(_is_test_file(REPO_ROOT / rel_str), f"Runtime allowlist forbidden: {rel_str}")

        violations: list[str] = []
        for path in _iter_py_files(LIVELAB_DIR, exclude_fixture=True):
            rel_posix = path.relative_to(REPO_ROOT).as_posix()
            allowed = _CORE_TEST_IMPORT_ALLOWLIST.get(rel_posix, frozenset())
            for lineno, level, module, _names in _iter_imports(path):
                if level != 0 or not module:
                    continue
                top = module.split(".")[0]
                if top in _FORBIDDEN_CORE_PACKAGES and top not in allowed:
                    violations.append(f"{rel_posix}:{lineno} imports {module}")
        self.assertEqual(violations, [])

    def test_kernel_never_imports_rsibox(self) -> None:
        """Non-test files under `core/livelab/kernel/` never import `rsibox`."""
        violations: list[str] = []
        for path in _iter_py_files(KERNEL_DIR, exclude_tests=True):
            rel_posix = path.relative_to(REPO_ROOT).as_posix()
            for lineno, level, module, names in _iter_imports(path):
                resolved = _resolve_relative_module(path, level, module) if level > 0 else module
                parts = resolved.split(".") if resolved else []
                is_rsibox = (
                    (parts and parts[0] == "rsibox")
                    or parts[:3] == ["core", "livelab", "rsibox"]
                    or (parts == ["core", "livelab"] and "rsibox" in names)
                )
                if is_rsibox:
                    violations.append(f"{rel_posix}:{lineno} imports {resolved or '.'} ({names})")
        self.assertEqual(violations, [])

    def test_kernel_layered_dag(self) -> None:
        """Kernel leaf modules never import orchestrator modules."""
        violations: list[str] = []
        for path in _iter_py_files(KERNEL_DIR, exclude_tests=True):
            if path.parent != KERNEL_DIR or path.stem not in _KERNEL_LEAF_MODULES:
                continue
            rel_posix = path.relative_to(REPO_ROOT).as_posix()
            for lineno, level, module, names in _iter_imports(path):
                imported_kernel_mods: set[str] = set()
                if level == 1:
                    if module:
                        imported_kernel_mods.add(module.split(".")[0])
                    else:
                        imported_kernel_mods.update(names)
                elif level == 0:
                    parts = module.split(".")
                    if parts[0] == "kernel":
                        if len(parts) > 1:
                            imported_kernel_mods.add(parts[1])
                        else:
                            imported_kernel_mods.update(names)
                bad = sorted(imported_kernel_mods & _KERNEL_ORCHESTRATOR_MODULES)
                if bad:
                    violations.append(f"{rel_posix}:{lineno} imports orchestrator(s): {', '.join(bad)}")
        self.assertEqual(violations, [])

    def test_adapters_isolated_from_each_other(self) -> None:
        """Each `repos/<engine>/` adapter never imports `repos.<other>`."""
        engines = sorted(
            d.name
            for d in REPOS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
        )
        self.assertTrue(engines, "expected at least one engine directory under repos/")
        violations: list[str] = []
        for engine in engines:
            engine_mod = engine.replace("-", "_")
            for path in _iter_py_files(REPOS_DIR / engine, exclude_fixture=True):
                rel_posix = path.relative_to(REPO_ROOT).as_posix()
                for lineno, level, module, names in _iter_imports(path):
                    if level != 0 or not module:
                        continue
                    parts = module.split(".")
                    if parts[0] == "repos":
                        if len(parts) > 1:
                            targets = [parts[1]]
                        elif names:
                            targets = names
                        else:
                            violations.append(f"{rel_posix}:{lineno} ({engine}) bare `import repos`")
                            continue
                        for target in targets:
                            if target not in (engine, engine_mod):
                                violations.append(f"{rel_posix}:{lineno} ({engine}) imports repos.{target}")
                    elif parts[0] == "rsibox_megatron" and engine != "megatron":
                        violations.append(f"{rel_posix}:{lineno} ({engine}) imports {module}")
        self.assertEqual(violations, [])

    def test_location_transparency_core_contracts(self) -> None:
        """Core kernel and warm runtime contracts contain no hardcoded loopback or private IP literals."""
        violations: list[str] = []
        for root_dir in (KERNEL_DIR, WARM_DIR):
            for path in _iter_py_files(root_dir, exclude_tests=True):
                rel_posix = path.relative_to(REPO_ROOT).as_posix()
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                doc_ids = _docstring_node_ids(tree)
                for node in ast.walk(tree):
                    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                        continue
                    if id(node) in doc_ids:
                        continue
                    text = node.value
                    if _LOOPBACK_URL_RE.search(text):
                        violations.append(f"{rel_posix}:{node.lineno} contains loopback URL literal")
                    for match in _IPV4_RE.finditer(text):
                        token = match.group(1)
                        try:
                            addr = ipaddress.IPv4Address(token)
                        except ValueError:
                            continue
                        if any(addr in net for net in _RFC5737_NETS):
                            continue
                        if addr.is_private or addr.is_loopback or addr.is_link_local or not addr.is_global:
                            violations.append(f"{rel_posix}:{node.lineno} contains non-public IP literal {token}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
