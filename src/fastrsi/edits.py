"""What an edit is, and whether it can go live. Runs on the host, on source text only."""

import ast
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

KEEP_DECORATORS = {"staticmethod", "classmethod", "property"}


class Refused(Exception):
    """A normal outcome: this edit needs a restart. Nothing was changed."""

    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True)
class FileEdit:
    path: str
    base: str
    new: str


@dataclass(frozen=True)
class Edit:
    files: tuple[FileEdit, ...]

    @property
    def digest(self):
        hasher = hashlib.sha256()
        for item in self.files:
            hasher.update(f"{item.path}\0{item.base}\0{item.new}\0".encode())
        return hasher.hexdigest()[:16]


def replace(root, path, old, new):
    """An edit that replaces one exact string in one file."""
    with open(os.path.join(root, path)) as handle:
        base = handle.read()
    if base.count(old) != 1:
        raise ValueError(f"{path}: expected one match for {old!r}, found {base.count(old)}")
    return Edit((FileEdit(path, base, base.replace(old, new)),))


def replace_all(root, triples):
    """One edit from several replacements in one file."""
    path = triples[0][0]
    with open(os.path.join(root, path)) as handle:
        base = handle.read()
    new = base
    for _, old, replacement in triples:
        if new.count(old) != 1:
            raise ValueError(f"{path}: expected one match for {old!r}")
        new = new.replace(old, replacement)
    return Edit((FileEdit(path, base, new),))


def from_patch(root, patch):
    """An edit from a unified diff with a/ and b/ prefixes, applied to a copy of root."""
    with open(patch) as handle:
        paths = [line[6:].split("\t")[0].strip() for line in handle if line.startswith("+++ b/")]
    if not paths:
        raise ValueError(f"{patch}: no files in patch")
    with tempfile.TemporaryDirectory() as work:
        for path in paths:
            os.makedirs(os.path.dirname(os.path.join(work, path)) or work, exist_ok=True)
            shutil.copy(os.path.join(root, path), os.path.join(work, path))
        subprocess.run(["patch", "-s", "-p1", "-d", work, "-i", os.path.abspath(patch)], check=True)
        files = []
        for path in paths:
            with open(os.path.join(root, path)) as base, open(os.path.join(work, path)) as new:
                files.append(FileEdit(path, base.read(), new.read()))
    return Edit(tuple(files))


def _definitions(tree):
    found = {}

    def walk(node, prefix):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[prefix + child.name] = child
            elif isinstance(child, ast.ClassDef):
                found[prefix + child.name] = child
                walk(child, prefix + child.name + ".")

    walk(tree, "")
    return found


def _skeleton(tree):
    """The file with every function body removed: what a code swap cannot change."""

    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.body = [ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    return ast.dump(Strip().visit(tree))


def _decorator_name(node):
    return node.func.id if isinstance(node, ast.Call) and hasattr(node.func, "id") else (
        node.id if isinstance(node, ast.Name) else ast.dump(node))


def classify_file(base, new):
    """Qualnames whose body changed. Raises Refused for anything a code swap cannot do."""
    try:
        old_tree, new_tree = ast.parse(base), ast.parse(new)
    except SyntaxError as error:
        raise Refused("syntax", f"edit does not parse: {error}") from error
    old_defs, new_defs = _definitions(old_tree), _definitions(new_tree)
    if added := sorted(set(new_defs) - set(old_defs)):
        raise Refused("new-definition", f"new definition: {added[0]}")
    if removed := sorted(set(old_defs) - set(new_defs)):
        raise Refused("removed-definition", f"removed definition: {removed[0]}")
    changed = []
    for name, old in old_defs.items():
        node = new_defs[name]
        if type(node) is not type(old):
            raise Refused("signature", f"{name}: definition kind changed")
        if isinstance(old, ast.ClassDef):
            continue
        if ast.dump(old.args) != ast.dump(node.args) or (
            ast.dump(old.returns) if old.returns else None
        ) != (ast.dump(node.returns) if node.returns else None):
            raise Refused("signature", f"{name}: signature changed")
        if [ast.dump(d) for d in old.decorator_list] != [ast.dump(d) for d in node.decorator_list]:
            raise Refused("decorator", f"{name}: decorators changed")
        if [ast.dump(s) for s in old.body] != [ast.dump(s) for s in node.body]:
            extra = {_decorator_name(d) for d in old.decorator_list} - KEEP_DECORATORS
            if extra:
                raise Refused("decorated", f"{name}: wrapped by {sorted(extra)[0]}")
            changed.append(name)
    if _skeleton(old_tree) != _skeleton(new_tree):
        raise Refused("module-level", "code outside a function body changed")
    return changed


def classify(edit):
    """Map each path to the functions to swap, or raise Refused."""
    route = {}
    for item in edit.files:
        if not item.path.endswith(".py"):
            raise Refused("not-python", f"{item.path}: no route for this file type")
        if changed := classify_file(item.base, item.new):
            route[item.path] = changed
    if not route:
        raise Refused("no-change", "no function body changed")
    return route
