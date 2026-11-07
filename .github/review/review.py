#!/usr/bin/env python3
"""Polish a PR's title and description with a reviewer agent, check it kept the facts, apply it.

The reviewer is either a local Antigravity CLI (`--engine agy`) or the review service
(`--engine service`, used by CI). Whatever the reviewer returns is validated before it is
applied: a rewrite that drops or adds a number, reference or code span, breaks the title rule,
loses a Loop record field or adds AI attribution is rejected, and nothing changes.

Standard library only. Needs `gh` (authenticated) to read and update the PR.

  review.py --pr 3                       # show the suggestion and the checks, change nothing
  review.py --pr 3 --apply               # also update the PR when every check passes
  review.py --pr 3 --engine service --url <review-service-url> --apply
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE_RE = re.compile(r"^(feat|fix|perf|docs|test|ci|refactor|chore)(\([a-z0-9-]+\))!?: .+")
ATTRIBUTION_RE = re.compile(
    r"(co-authored-by:.*(claude|anthropic|codex|openai|gemini|antigravity|copilot))"
    r"|generated with \[?(claude|codex|gemini|antigravity)", re.I)
LOOP_FIELDS = ["Engine", "Upstream PRs", "Function groups", "Rule added",
               "Overheads measured", "Receipts", "Not verified"]


def sh(*args, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, text=True, check=True).stdout


def strip_comments(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def facts(text):
    """What a rewrite must keep: numbers, references, hashes and code spans.

    Backticks are formatting, not facts: numbers and references are read from the whole text,
    and a code span only counts as new or lost if its text is new or lost (see check)."""
    text = strip_comments(text)
    code = {c[1:-1] for c in re.findall(r"`[^`\n]+`", text)}
    flat = text.replace("`", "")
    refs = re.findall(r"[\w.-]+/[\w.-]+#\d+|#\d+|\b[0-9a-f]{7,40}\b", flat)
    nums = re.findall(r"(?<![\w#])\d[\d,]*(?:\.\d+)?", flat)
    return {"code": code, "refs": set(refs), "numbers": set(nums),
            "plain": re.sub(r"\s+", " ", flat)}


def check(old_title, old_body, new_title, new_body):
    """Return a list of problems; empty means the suggestion is safe to apply."""
    problems = []
    if not TITLE_RE.match(new_title):
        problems.append(f"title does not match the rule: {new_title!r}")
    if len(new_title) > 72:
        problems.append(f"title is {len(new_title)} characters (max 72)")
    m_old, m_new = TITLE_RE.match(old_title), TITLE_RE.match(new_title)
    if m_old and m_new and m_old.group(1, 2) != m_new.group(1, 2):
        problems.append("title changed its type or scope")
    a, b = facts(old_title + "\n" + old_body), facts(new_title + "\n" + new_body)
    for kind in ("refs", "numbers"):
        if a[kind] - b[kind]:
            problems.append(f"dropped {kind}: {sorted(a[kind] - b[kind])[:8]}")
        if b[kind] - a[kind]:
            problems.append(f"added {kind}: {sorted(b[kind] - a[kind])[:8]}")
    # Wrapping existing text in backticks (or unwrapping it) is fine; new or lost text is not.
    lost = sorted(c for c in a["code"] if " ".join(c.split()) not in b["plain"])
    new = sorted(c for c in b["code"] if " ".join(c.split()) not in a["plain"])
    if lost:
        problems.append(f"dropped code: {lost[:8]}")
    if new:
        problems.append(f"added code: {new[:8]}")
    if "## Loop record" in old_body:
        for field in LOOP_FIELDS:
            if f"{field}:" not in new_body:
                problems.append(f"Loop record field missing: {field}")
    if ATTRIBUTION_RE.search(new_title + "\n" + new_body):
        problems.append("adds AI attribution")
    return problems


def parse(reply):
    def block(name):
        m = re.search(rf"<<<{name}\n(.*?)\n?{name}>>>", reply, re.S)
        if not m:
            raise ValueError(f"reviewer reply has no {name} block")
        return m.group(1).strip()
    return block("TITLE"), block("BODY"), block("NOTES")


def ask_agy(prompt, model):
    agy = os.environ.get("AGY", "agy")
    return sh(agy, "-p", prompt, "--model", model, "--print-timeout", "540s")


def ask_service(url, token, payload):
    """POST a job, then poll it. Asynchronous because a review can outlast a proxy timeout."""
    # Cloudflare's Browser Integrity Check rejects the default Python-urllib agent (error 1010).
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}",
               "User-Agent": "fastrsi-review-ci/1"}
    cf = {"CF_ACCESS_CLIENT_ID": "CF-Access-Client-Id",  # optional Cloudflare Access token
          "CF_ACCESS_CLIENT_SECRET": "CF-Access-Client-Secret"}
    headers.update({h: os.environ[k] for k, h in cf.items() if os.environ.get(k)})
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    job = json.load(urllib.request.urlopen(req, timeout=30))
    deadline = time.time() + 600
    while time.time() < deadline:
        r = urllib.request.Request(f"{url}/{job['id']}", headers=headers)
        state = json.load(urllib.request.urlopen(r, timeout=30))
        if state["status"] == "done":
            return state["reply"]
        if state["status"] == "error":
            raise RuntimeError(f"review service: {state.get('error')}")
        time.sleep(5)
    raise TimeoutError("review service did not finish in 600 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "GindaChen/fastrsi"))
    ap.add_argument("--engine", choices=["agy", "service"], default="agy")
    ap.add_argument("--model", default="gemini-4-argon-high")
    ap.add_argument("--url", default=os.environ.get("FASTRSI_REVIEW_URL"))
    ap.add_argument("--apply", action="store_true", help="update the PR if every check passes")
    args = ap.parse_args()

    pr = json.loads(sh("gh", "api", f"repos/{args.repo}/pulls/{args.pr}"))
    title, body = pr["title"], pr["body"] or ""
    files = json.loads(sh("gh", "api", f"repos/{args.repo}/pulls/{args.pr}/files", "--paginate"))
    diffstat = "\n".join(f"{f['filename']} +{f['additions']}/-{f['deletions']}" for f in files)
    rules = (HERE / "pr_description.md").read_text()
    prompt = f"{rules}\nTITLE: {title}\n\nFILES CHANGED (context only):\n{diffstat}\n\nDESCRIPTION:\n{body}\n"

    if args.engine == "agy":
        reply = ask_agy(prompt, args.model)
    else:
        reply = ask_service(args.url, os.environ["FASTRSI_REVIEW_TOKEN"], {
            "kind": "pr-description", "repo": args.repo, "pr": int(args.pr),
            "prompt": prompt, "model": args.model})
    new_title, new_body, notes = parse(reply)

    print(f"--- reviewer notes\n{notes}\n--- title\n{title}\n{new_title}")
    problems = check(title, body, new_title, new_body)
    if (new_title, new_body.strip()) == (title, body.strip()):
        print("--- no change suggested")
        return 0
    if problems:
        print("--- rejected, nothing changed:\n" + "\n".join(f"- {p}" for p in problems))
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::warning title=Description review rejected::{len(problems)} fact check(s) "
                  "failed, so the suggested rewrite was not applied. See the job log.")
        return 0 if os.environ.get("GITHUB_ACTIONS") else 2
    print("--- every check passed")
    if args.apply:
        sh("gh", "api", "-X", "PATCH", f"repos/{args.repo}/pulls/{args.pr}",
           "-f", f"title={new_title}", "-f", f"body={new_body}")
        sh("gh", "api", f"repos/{args.repo}/issues/{args.pr}/comments", "-f",
           f"body=Description reviewed and polished. Changes:\n\n{notes}\n\n"
           "The previous version is in this PR's edit history.")
        print("--- applied")
    else:
        print(f"--- suggested body\n{new_body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
