"""python3 -m survey scan|history ...   (from core/livelab; see README.md)"""
import argparse
import datetime
import json
import subprocess
import sys

from . import history as H
from . import rules as R
from . import scan as S


def _write(obj, path):
    text = json.dumps(obj, indent=1) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def table(d):
    """The report's totals as text: one row per install class, in functions and in lines."""
    rows = [f"{d['engine']}: {d['functions']['total']} functions, {d['lines']['total']} lines; "
            f"lab has {', '.join(k for k, v in d['lab']['capabilities'].items() if v) or 'nothing'}",
            f"{'install class':16}{'functions':>10}{'covered':>9}{'lines':>9}{'covered':>9}"]
    for i in R.INSTALLS:
        f, l = d["functions"]["by_install"][i], d["lines"]["by_install"][i]
        cov = ("-", "-") if i == "restart" else (f["covered"], l["covered"])
        rows.append(f"{i:16}{f['total']:>10}{cov[0]:>9}{l['total']:>9}{cov[1]:>9}")
    for unit in ("functions", "lines"):
        s = d[unit]
        rows.append(f"{unit}: live {s['live_covered']}, respawn {s['respawn_covered']}, restart "
                    f"{s['restart']}, no path yet {s['not_covered']}; firm {s['firm']}, "
                    f"uncertain {s['uncertain']} ({100 * s['uncertain'] / max(s['total'], 1):.0f}%)")
    rows.append(f"native: {d['native']['files']} files, {d['native']['lines']} lines (restart); "
                f"unparsed: {d['unparsed']['files']} files")
    return "\n".join(rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="survey", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "history"):
        p = sub.add_parser(name)
        p.add_argument("--rules", required=True, help="the engine's rules file (JSON)")
        p.add_argument("--source", required=True, help="the engine's upstream tree")
        p.add_argument("--repo", default=".", help="the checkout that holds the lab's files")
        p.add_argument("--date", default=datetime.date.today().isoformat())
        p.add_argument("--out", help="write the JSON here instead of stdout")
    sub.choices["scan"].add_argument("--commit", help="read the lab at this commit (default: the files)")
    sub.choices["scan"].add_argument("--functions", help="also write one JSON line per record here")
    sub.choices["scan"].add_argument("--summary", action="store_true", help="print a text table")
    sub.choices["history"].add_argument("--ref", default="origin/main")
    sub.choices["history"].add_argument("--lab", help="lab folder (default: the rules' lab_path)")
    a = ap.parse_args(argv)
    rules = R.load(a.rules)
    if a.cmd == "history":
        lab_path = a.lab or rules.get("lab_path")
        if not lab_path:
            ap.error("history needs --lab or a lab_path in the rules")
        _write(H.series(rules, a.source, a.repo, a.ref, lab_path, a.date), a.out)
        return 0
    commit = a.commit and subprocess.run(
        ["git", "-C", a.repo, "rev-parse", a.commit + "^{commit}"], capture_output=True,
        text=True, check=True).stdout.strip()
    lab = S.Commit(a.repo, commit) if commit else S.Files(a.repo)
    out, records = S.report(rules, a.source, lab, commit or S.git_head(a.repo), a.date)
    if a.functions:
        with open(a.functions, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    if a.summary:
        sys.stdout.write(table(out))
    if a.out or not a.summary:
        _write(out, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
