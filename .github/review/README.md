# Agent reviews of PRs

Two reviews share one review service:

- **Description review:** an Antigravity agent polishes each PR's title and description (never
  the code). A checker makes sure the rewrite kept every fact before it is applied.
- **Code review:** Antigravity and Codex review the diff against `AGENTS.md`. Their findings are
  checked and posted as one advisory PR review: it never approves or blocks, a human merges.

| File | Does |
|---|---|
| `pr_description.md` | The description reviewer prompt |
| `review.py` | Fetches the PR, asks the reviewer, checks the reply, applies it |
| `test_review.py` | Golden cases: rewrites the checker must accept or reject |
| `../workflows/pr-description-review.yml` | Runs the description review in CI |
| `code_review.md` | The code reviewer prompt, sent with every code-review request |
| `code_review.py` | Fetches diff and description, asks each engine, checks, merges, posts one review |
| `test_code_review.py` | Golden cases: findings kept, dropped, merged; never an approval; confinement |
| `../workflows/pr-code-review.yml` | Runs the code review in CI |
| `server.py`, `confine.py` | The review service and the Landlock wrapper it runs reviewers in |

## Code review

On demand only: when the repository owner comments `/review` on a PR (open or merged), or the
workflow is run by hand with the PR number. Merging never waits for it. `code_review.py`:

1. fetches the diff (cut at 90,000 characters; the review says so when it is cut) and the
   description, and builds the prompt: `code_review.md`, the PR as untrusted data, `AGENTS.md`;
2. sends it to each engine through the service (`kind: "code-review"`, `engine: "agy" | "codex"`);
3. keeps a finding only if it is well-formed JSON and points at a line on the new side of the
   diff; merges findings from both engines on the same line into one comment naming both;
4. posts one review with event `COMMENT`. An engine that is down or answers garbage is
   skipped with a notice; the PR never fails because a reviewer is unavailable.

```bash
python3 .github/review/code_review.py --pr 22                 # print, post nothing
python3 .github/review/code_review.py --pr 22 --post          # also post it
```

It needs the same `FASTRSI_REVIEW_URL` and `FASTRSI_REVIEW_TOKEN` as the description review.

## What the description checker guarantees

A suggestion is applied only if, compared with the original:

- every number, PR/issue reference, commit hash and backticked code span is still there, and
  none was added;
- the title matches the title rule, keeps its type and scope, and is at most 72 characters;
- every Loop record field is still there (when the original has the section);
- it adds no AI attribution.

It does not judge whether the new wording is better. That is the reviewer's job, and the PR's
edit history keeps the previous version.

## Run it locally

Needs `gh` and the Antigravity CLI (`agy`; set `AGY` if it is not on `PATH`).

```bash
python3 .github/review/review.py --pr 3            # show suggestion and checks
python3 .github/review/review.py --pr 3 --apply    # apply if every check passes
```

## Run it in CI: the review service

CI cannot run `agy` or `codex` (they need your logins), so CI calls a service you host at
the review service URL, which runs them for it.

**Contract (v1), asynchronous** because a review can take minutes and Cloudflare closes proxied
requests after 100 s:

```
POST /v1/review   {"kind", "engine", "repo", "pr", "prompt", "model"}  ->  {"id"}
GET  /v1/review/<id>   ->  {"status": "queued|running|done|error", "reply", "error"}
```

`reply` is the raw reviewer output (TITLE/BODY/NOTES blocks, or the code review's JSON). The
service only runs the prompt; checking, applying and posting happen in CI, so the service never
needs GitHub access.

**The server** is `server.py` (standard library only). Requests carry `kind`
(`pr-description` or `code-review`) and `engine` (`agy`, the default, or `codex` for code
review). Each job runs in a fresh empty directory with a minimal environment and a fresh HOME that
holds only copies of the reviewer's credential files (never its real state directory with
other conversations or settings), inside `confine.py`: Linux Landlock lets the reviewer read
only system paths (without `/run`, except specific DNS resolver files) and its own binary file
(never its parent directory recursively; extra bundle directories require
`FASTRSI_REVIEW_EXTRA_READ`), isolates `/dev/shm` per job via `--private-shm` (denying global
`/dev/shm` if namespaces are unavailable), and lets it write only the job directory. A refreshed
credential is copied back only if it is valid JSON for the same account and the real file was
not changed meanwhile. Known limit: a prompt-injected reviewer that keeps the account fields
can still write unusable tokens into its copy, which would log the host CLI out until you log
in again (denial of service; it cannot plant another account's credentials). Removing
copy-back closes that, at the risk of a rotated refresh token logging the host out instead. This matters
because agy's file tools cannot be turned off from its command line: in our test an
unconfined headless `agy -p --sandbox` read a file outside its directory when asked to;
confined, the same request got "permission denied" even with every agy tool approved. Codex
runs with its read-only sandbox and its shell, exec, web, browser, app and plugin tools
disabled. A reply or error that contains text from a secret file (the service token, the reviewers'
credentials, SSH keys) is withheld. In CI the code review runs its script from the PR's base
commit, never from the PR, because that job holds the service token. Every call except `/v1/health` needs
`Authorization: Bearer <token>`.

Trap: Cloudflare's Browser Integrity Check answers 403 (error 1010) to the default
`Python-urllib` user agent before the request reaches the server, so `review.py` sends its own.

**Setup steps:**

1. Run `server.py` (next to `confine.py`) on a Linux host with Landlock and `agy` and/or `codex`
   (set `AGY` / `CODEX` if they are not on `PATH`), behind HTTPS. Optionally put a Cloudflare Access policy with a service
   token in front.
2. In the GitHub repo settings:
   - variable `FASTRSI_REVIEW_URL` = the review service URL
   - secret `FASTRSI_REVIEW_TOKEN` = the bearer token
   - secrets `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`, only if you use Access

Until the URL and token are set, the CI review step skips with a notice and changes nothing.
The golden cases run either way.

**Things to know:**

- Runs on demand only: the repository owner comments `/describe` on a PR (open or merged), or
  the workflow is run by hand. On a merged PR it updates the PR page only; the squash commit
  keeps the description it was merged with.
- `/review` and `/describe` count only from the repository owner's account. Agents that post
  with that account can trigger them too, so they do it only when asked.
- Edits made with the workflow's `GITHUB_TOKEN` do not trigger other workflows, so the PR rules
  check does not rerun on the new title. The checker enforces the same title and attribution
  rules itself.
- Repository secrets are not given to workflows from forks, so fork PRs skip the review.
