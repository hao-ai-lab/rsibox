"""Walk an engine's upstream tree, classify every function (rules.py), and sum the result.
Reads source text with `ast`; imports and runs nothing of the engine."""
import ast
import os
import subprocess

from . import rules as R

SCHEMA = "fastrsi-survey/1"
LIMITS = [
    "Static: the scan cannot tell which functions a workload reaches. Shares are of the source "
    "tree, not of the code a run executes.",
    "Only rules with basis 'syntax' are firm. 'default_op' (no rule matched) assumes a per-call "
    "function body; such a function may run once at start-up or only inside a captured graph.",
    "'covered' means the lab has the install path and reaches the file. It is not a receipt: "
    "no edit was installed and no witness read.",
    "Nested functions and lambdas count with the function that encloses them. Files that do "
    "not parse are counted in 'unparsed' and in no share.",
    "Native sources are counted by file and line, apart from the Python shares.",
    "Lines are non-blank lines that are not only a comment; docstrings count.",
]


class Files:
    """Lab files read from a folder (the working tree)."""
    def __init__(self, root):
        self.root = root

    def read(self, path):
        try:
            with open(os.path.join(self.root, path), encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None


class Commit:
    """Lab files read from one commit of a git repository."""
    def __init__(self, repo, commit):
        self.repo, self.commit = repo, commit

    def read(self, path):
        p = subprocess.run(["git", "-C", self.repo, "show", f"{self.commit}:{path}"],
                           capture_output=True)
        return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def _dotted(node):
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _calls(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def functions(tree):
    """(qualname, node) of every function a module or a class body defines. A nested function
    belongs to the function that encloses it."""
    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + n.name, n
            elif isinstance(n, ast.ClassDef):
                yield from walk(n.body, prefix + n.name + ".")
            else:   # if / try / with at module or class level
                for field in ("body", "orelse", "finalbody", "handlers"):
                    yield from walk(getattr(n, field, None) or [], prefix)
    yield from walk(tree.body, "")


def _top(path, roots):
    """The top-level module of a file: the first folder (or file) under its source root."""
    for root in sorted(roots, key=len, reverse=True):
        if path == root or path.startswith(root.rstrip("/") + "/"):
            rest = path[len(root.rstrip("/")) + 1:]
            return (root.rstrip("/") + "/" + rest.split("/")[0]) if rest else root
    return path.split("/")[0]


def _walk(source, roots, exclude):
    for root in roots:
        top = os.path.join(source, root)
        if os.path.isfile(top):
            yield root
            continue
        for d, dirs, files in os.walk(top):
            dirs[:] = sorted(x for x in dirs if x != ".git")
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(d, f), source).replace(os.sep, "/")
                if not R.match(rel, exclude):
                    yield rel


def classify_tree(source, rules):
    """One record per function, plus one per Python file for its module-level lines and one per
    native file. Depends on the upstream tree and the rules only, not on any lab."""
    records, unparsed, seen = [], [], set()
    roots = rules["source_roots"]
    for rel in list(_walk(source, roots, rules["exclude"])) + list(
            _walk(source, rules["native_roots"], rules["exclude"])):
        if rel in seen:
            continue
        seen.add(rel)
        ext = os.path.splitext(rel)[1]
        if ext != ".py" and ext not in R.NATIVE_EXT:
            continue
        with open(os.path.join(source, rel), encoding="utf-8", errors="replace") as f:
            text = f.read()
        code = [bool(l.strip()) and not l.lstrip().startswith("#") for l in text.splitlines()]
        nlines = sum(code)
        base = {"path": rel, "module": _top(rel, roots)}
        if ext != ".py":
            records.append({**base, "kind": "native", "qualname": "<native>", "line": 1,
                            "lines": nlines, "group": "NATIVE", "install": "restart",
                            "capability": None, "rule": "native_file"})
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            unparsed.append({"path": rel, "lines": nlines})
            continue
        inside = 0
        for qual, fn in functions(tree):
            first = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
            span = sum(code[first - 1:fn.end_lineno])
            inside += span
            group, install, cap, rule = R.classify(
                rules, rel, fn.name, [_dotted(d) for d in fn.decorator_list], _calls(fn))
            records.append({**base, "kind": "function", "qualname": qual, "line": first,
                            "lines": span, "group": group, "install": install,
                            "capability": cap, "rule": rule})
        group, install, cap, rule = R.module_level(rules, rel)
        records.append({**base, "kind": "module_level", "qualname": "<module>", "line": 1,
                        "lines": nlines - inside, "group": group, "install": install,
                        "capability": cap, "rule": rule})
    for r in records:
        r["basis"] = R.BASIS[r["rule"]]
        r["firm"] = r["basis"] == "syntax"
        r["reach"] = r["capability"] is not None and R.reached(rules, r["capability"], r["path"])
    return records, unparsed


def cover(records, caps):
    """Set `covered` on each record: the lab has the capability and reaches the file."""
    for r in records:
        r["covered"] = bool(r["reach"] and caps.get(r["capability"], False))
    return records


def _cell():
    return {i: {"total": 0, "covered": 0} for i in R.INSTALLS}


def _sum(rows, weight):
    by_install, by_group, by_rule, firm, total = _cell(), {}, {}, 0, 0
    for r in rows:
        w = weight(r)
        total += w
        by_install[r["install"]]["total"] += w
        by_install[r["install"]]["covered"] += w if r["covered"] else 0
        by_group[r["group"]] = by_group.get(r["group"], 0) + w
        by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + w
        firm += w if r["firm"] else 0
    live = sum(by_install[i]["covered"] for i in R.LIVE)
    return {"total": total, "by_install": by_install, "by_group": dict(sorted(by_group.items())),
            "by_rule": dict(sorted(by_rule.items())), "firm": firm, "uncertain": total - firm,
            "live_covered": live, "respawn_covered": by_install["respawn"]["covered"],
            "restart": by_install["restart"]["total"],
            "not_covered": total - live - by_install["respawn"]["covered"]
                           - by_install["restart"]["total"]}


def summarize(records):
    """Totals and per-module counts, in functions and in Python lines; native files apart."""
    py = [r for r in records if r["kind"] != "native"]
    fns = [r for r in py if r["kind"] == "function"]
    native = [r for r in records if r["kind"] == "native"]
    by_ext = {}
    for r in native:
        e = os.path.splitext(r["path"])[1]
        c = by_ext.setdefault(e, {"files": 0, "lines": 0})
        c["files"] += 1
        c["lines"] += r["lines"]
    modules = {}
    for name in sorted({r["module"] for r in records}):
        mine = [r for r in records if r["module"] == name]
        modules[name] = {
            "functions": _sum([r for r in mine if r["kind"] == "function"], lambda r: 1),
            "lines": _sum([r for r in mine if r["kind"] != "native"], lambda r: r["lines"]),
            "native": {"files": sum(r["kind"] == "native" for r in mine),
                       "lines": sum(r["lines"] for r in mine if r["kind"] == "native")}}
    return {"functions": _sum(fns, lambda r: 1), "lines": _sum(py, lambda r: r["lines"]),
            "native": {"files": len(native), "lines": sum(r["lines"] for r in native),
                       "install": "restart", "by_ext": dict(sorted(by_ext.items()))},
            "modules": modules}


def git_head(path):
    """HEAD of the git checkout whose top folder is `path`; None for any other folder."""
    p = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel", "HEAD"],
                       capture_output=True, text=True)
    top, _, head = p.stdout.strip().partition("\n")
    same = p.returncode == 0 and os.path.realpath(top) == os.path.realpath(path)
    return head if same else None


def report(rules, source, lab, lab_commit, date):
    """The survey of one engine: classify the upstream tree, read the lab's capabilities from
    `lab` (Files or Commit), and sum. Returns (report dict, records)."""
    records, unparsed = classify_tree(source, rules)
    caps = R.capabilities(rules, lab)
    cover(records, caps)
    up = dict(rules["upstream"])
    up["scanned_commit"] = git_head(source)
    up["pin_match"] = bool(up.get("commit")) and up["scanned_commit"] == up.get("commit")
    out = {"schema": SCHEMA, "engine": rules["engine"], "scan_date": date,
           "ruleset_version": R.RULESET_VERSION, "engine_rules_version": rules["rules_version"],
           "engine_rules_sha256": rules["sha256"], "upstream": up,
           "lab": {"commit": lab_commit, "capabilities": caps},
           "unparsed": {"files": len(unparsed), "lines": sum(u["lines"] for u in unparsed)},
           **summarize(records), "limits": LIMITS}
    return out, records
