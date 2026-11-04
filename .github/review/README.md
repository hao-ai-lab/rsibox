# PR description review

An Antigravity agent polishes each PR's title and description (never the code). A checker then
makes sure the rewrite kept every fact before it is applied.

| File | Does |
|---|---|
| `pr_description.md` | The reviewer prompt |
| `review.py` | Fetches the PR, asks the reviewer, checks the reply, applies it |
| `test_review.py` | Golden cases: rewrites the checker must accept or reject |
| `../workflows/pr-description-review.yml` | Runs it in CI |

## What the checker guarantees

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

CI cannot run `agy` (it needs your Antigravity login), so CI calls a service you host at
`https://services.fastrsi.ai/v1/review`, which runs `agy` for it.

**Contract (v1), asynchronous** because a review can take minutes and Cloudflare closes proxied
requests after 100 s:

```
POST /v1/review   {"kind": "pr-description", "repo", "pr", "prompt", "model"}  ->  {"id"}
GET  /v1/review/<id>   ->  {"status": "queued|running|done|error", "reply", "error"}
```

`reply` is the raw reviewer output (the TITLE/BODY/NOTES blocks). The service only runs the
prompt; checking and applying happen in CI, so the service never needs GitHub write access.

**Setup:**

1. Run the service on a machine logged in to Antigravity, for example b200: a small HTTP server
   that, per job, runs `agy -p "$prompt" --model "$model"` and stores the reply.
2. Expose it with the existing Cloudflare tunnel: add an ingress rule
   `hostname: services.fastrsi.ai`, `path: ^/v1/review`, pointing at the server's local port.
3. Protect it: the server checks `Authorization: Bearer <token>`. Optionally also put a
   Cloudflare Access policy with a service token in front.
4. In the GitHub repo settings:
   - variable `FASTRSI_REVIEW_URL` = `https://services.fastrsi.ai/v1/review`
   - secret `FASTRSI_REVIEW_TOKEN` = the bearer token
   - secrets `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`, only if you use Access

Until the URL and token are set, the CI review step skips with a notice and changes nothing.
The golden cases run either way.

**Things to know:**

- Runs on opened, reopened and ready-for-review. To review again after editing, add the
  `review-description` label or run the workflow by hand with the PR number.
- Edits made with the workflow's `GITHUB_TOKEN` do not trigger other workflows, so the PR rules
  check does not rerun on the new title. The checker enforces the same title and attribution
  rules itself.
- Repository secrets are not given to workflows from forks, so fork PRs skip the review.
