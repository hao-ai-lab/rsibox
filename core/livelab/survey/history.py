"""Coverage against the lab's history: the upstream tree and the rules stay fixed, the lab's
capabilities are read at each commit that touched the lab's folder."""
import subprocess

from . import rules as R
from . import scan as S


def commits(repo, ref, lab_path):
    """[(sha, committer date, subject)] of the first-parent commits of `ref` that touched
    `lab_path`, oldest first."""
    p = subprocess.run(["git", "-C", repo, "log", "--first-parent", "--reverse",
                        "--format=%H%x09%cI%x09%s", ref, "--", lab_path],
                       capture_output=True, text=True, check=True)
    return [tuple(line.split("\t", 2)) for line in p.stdout.splitlines() if line]


def series(rules, source, repo, ref, lab_path, date):
    records, unparsed = S.classify_tree(source, rules)
    points, last = [], None
    for i, (sha, when, subject) in enumerate(commits(repo, ref, lab_path)):
        caps = R.capabilities(rules, S.Commit(repo, sha))
        S.cover(records, caps)
        s = S.summarize(records)
        point = {"index": i, "commit": sha, "date": when, "subject": subject,
                 "capabilities": caps, "changed": caps != last}
        for unit in ("functions", "lines"):
            point[unit] = {k: s[unit][k] for k in
                           ("total", "live_covered", "respawn_covered", "restart", "not_covered")}
            point[unit]["by_install_covered"] = {
                i_: c["covered"] for i_, c in s[unit]["by_install"].items()}
        points.append(point)
        last = caps
    up = dict(rules["upstream"])
    up["scanned_commit"] = S.git_head(source)
    return {"schema": S.SCHEMA + "/history", "engine": rules["engine"], "scan_date": date,
            "ruleset_version": R.RULESET_VERSION, "engine_rules_version": rules["rules_version"],
            "engine_rules_sha256": rules["sha256"], "upstream": up, "ref": ref,
            "lab_path": lab_path, "points": points,
            "note": "Rules and upstream tree fixed at this run; only the lab's files vary. "
                    "A step is the commit whose files first satisfy a capability's evidence.",
            "limits": S.LIMITS}
