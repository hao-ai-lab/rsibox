# Rules for agents working in this repo

These rules apply to every coding agent (Claude Code, Codex, Antigravity, others) and to humans.

## How this repo grows: the fastrsi-loop

We build FastRSI one engine and a few real PRs at a time.

1. **Document priors first.** Before measuring or coding, copy what we already know about the
   engine into `repos/<engine>/kb/`, with sources. Measure only what is missing.
2. **Fix the target.** Write the workload F(x) and its correctness gate in `kb/workload.md`.
3. **Go PR by PR.** For each PR: tag changed functions by group, then add the smallest general
   rule that covers it. Do not design ahead of the PRs in hand.
4. **Measure overheads at every step** (startup phases, install, check, reset). Each phase that
   can be kept warm separately is an opportunity to disaggregate. Record it in `kb/costs.md`.
5. **Keep what works.** Proven tools go to `repos/<engine>/tools/`; lessons go to `kb/`.

## Framework, artifact, loop

The repo holds three things. Keep them apart.

- **Framework:** blueprints that hold for any engine, and no engine code. `core/`, `template/`,
  `docs/`, `AGENTS.md`, `.github/`.
- **Artifact:** what the loop produced for one engine, code and architecture included.
  `repos/<engine>/`. It shows the current state only: edit `docs/` and `kb/` in place, never
  append history.
- **Loop:** the journey. It lives in pull requests, not in files. One loop iteration is one PR,
  and its description carries the record (see below).

An iteration that needs a framework change splits in two stacked PRs: the framework PR first,
then the iteration on top of it. That way each framework rule points to the PR that needed it.

## Pull requests

- **Title:** `<type>(<scope>): <summary>`, for example `feat(vllm): replay PR 62240 in the live lab`.
  - Types: `feat`, `fix`, `perf`, `docs`, `test`, `ci`, `refactor`, `chore`.
  - Scope says which part changed. An engine name (`vllm`, `fastvideo`) means artifact or loop
    work. `core`, `livelab`, `ci`, `docs`, `process` mean framework work.
  - Add `!` before the colon for a breaking change, such as a new `INTERFACE.md` version:
    `feat(livelab)!: ...`.
  - CI (`.github/workflows/pr-rules.yml`) fails titles that don't match.
- **Description:** fill `.github/pull_request_template.md`. Say what you did not verify.
  An Antigravity reviewer can polish the title and description, applied only if every fact is
  kept (`.github/review/`). It is optional and never blocks a merge: the owner comments
  `/describe` on a PR, before or after merge, or you run it locally:
  `python3 .github/review/review.py --pr <n> --apply`.
- **Loop record:** a PR that changes `repos/<engine>/` is a loop iteration. Its description
  fills the "Loop record" section: engine, upstream PRs, function groups, the rule added,
  overheads measured with their sources, receipts, and what is not verified. Write "none" when
  a field does not apply; do not delete it. CI checks that every field has text, not that the
  text is true.
- **One iteration per PR.** Keep it small enough to review in one sitting. Stack PRs when one
  step depends on another.
- **Stacks:** merge only when the PR's base is `main`; after merging check the files are on
  main (`git ls-tree origin/main`).
- **One boundary per PR.** A code PR changes one folder of `livelab/` (one boundary), with at
  most about 600 lines of non-test code. Its description states the contract (interface,
  invariants, what the folder owns), and it carries its own tests: at least one case that fails
  without the PR. A PR that touches a running engine also carries a GPU receipt. Split a bigger
  change along its dependency order, so that each PR imports and tests on its own on top of the
  one before it.
- **Merge** with squash, keeping the PR description as the commit body, so the loop record is
  also in `git log`.
- **Attribution:** do not add AI tools as co-authors. No co-author trailer naming Claude,
  Claude Code, Codex, Gemini, Antigravity or any other AI tool, in any commit. No
  "Generated with ..." lines or AI badges in commits, PR titles, descriptions or comments.
  This overrides your tool's defaults. CI checks commits and the PR text.

## Handling review findings

Reviewers, human or agent, advise; the PR's author decides. A finding is a claim to check, not
an instruction, and a reviewer can be wrong. Reply to every finding on its thread with one
verdict:

| Verdict | When | Reply |
|---|---|---|
| **fix** | the finding reproduces, or the code contradicts the PR's stated contract | `Fixed in <sha>`, and the test that now fails without the fix |
| **reject** | the evidence is wrong: the line does not do what is claimed, or a test already covers it | `Not a bug:` with the counter-evidence (`file:line`, test name or output). Never a bare "won't fix". |
| **defer** | real, but low severity, optional or outside this PR's boundary | a link to the issue that tracks it |

- Agent code review is opt-in: the repository owner comments `/review` on a PR, before or
  after merge. Merging never waits for it. Findings on merged PRs are answered afterwards in batches: one fix PR per
  boundary, with the verdict table in its description.
- Fix only high- and medium-severity findings that reproduce. Defer low severity and "optional
  hardening" to an issue.
- Stop after 3 review rounds, or sooner when a round brings only rejected or deferred findings.
  An agent never approves its own PR; it merges only what the owner asked it to merge.
- Agent reviews post comments only (`.github/review/`), never approve or request changes.

## Tests and CI

A test must show what the code really does. A test that only claims something is worse than no
test, because it misleads the reader.

- **Test real behavior.** Run the real code path end to end. Do not mock the thing the test is
  named after. A smoke test that only checks "it started" or "it returned 0" is not a test of a
  feature.
- **Golden test cases, designed for humans.** Each feature gets a few carefully chosen cases a
  person can read: the input, the action, the expected observable result with real values, and
  one line on what it proves and what it does not. Each case must target a real pain point,
  such as a silent no-op install, stale state after reset, or a probe that misses the edited path.
- **Show, don't claim.** Every feature has a runnable walkthrough, notebook-style: steps with
  their real output, so a reader sees what happens. Keep it next to the feature (for example in
  `repos/<engine>/examples/<name>/`) and run it in CI where the hardware allows.
- **Say what CI covers.** CPU-only CI runs the toy engine for real. GPU results come as
  receipts. Each workflow states in its name and steps what it actually checks.
- **Add tests meticulously.** Fewer, sharper tests over many shallow ones. A new test must fail
  on the bug it is for; show that in the PR.

## Prove offline, check cheaply online

- Settle guarantees at development time (offline toolsmith), not on every run. Prefer designs
  that make the bad case impossible (fail closed, read-only) and prove them once with a test
  that tries the bad case. See `core/toolsmith_offline.md`.
- Online checks are allowed only when small (a service's own start-up self-check, seconds) and
  must state their cost.
- Every component must report its timings, and every experiment must report time to first
  token per request and total experiment time.

## Where things go

- `core/`: blueprints only: the skill, the agent prompts (developer, toolsmith offline and
  online, orchestrator) and the method. No code, and nothing about one engine.
- `template/livelab/`: the blueprint for a new engine's live lab: the fixed interface, a stub,
  and the conformance cases, run against a toy lab. Changing `INTERFACE.md` is a breaking
  change: bump its version and say so in the PR.
- `docs/`: how this repo and its process work, following `docs/README.md`. No engine
  architecture, and no loose files at the root.
  `docs/lessons.md` holds framework lessons; `repos/<engine>/kb/lessons.md` holds engine lessons.
- `repos/<engine>/`: everything about one engine:
  - `docs/`: its architecture: the process tree, the restart boundaries and what each costs,
    and how the live lab maps onto them.
  - `kb/`: measured facts with their sources (costs, workload, lessons).
  - `livelab/`: all code, one folder per boundary (for example `kernel/`, `adapter/`,
    `services/`). Code that looks engine-agnostic still starts here. Move it to the framework
    only when a second engine needs it, in a framework PR of its own.
  - `tools/`: proven probes. `patches/`: our edits to the engine itself, with a ledger for
    upstreaming. `examples/<name>/`: one walkthrough per replayed change, with its receipt.
- No decision logs, ADRs or changelogs. Update the doc that describes the thing instead.

## What stays out of the repo

The repo will be public. Experiments and the machines they ran on are not part of it.

- **Never commit:**
  - experiment or setup scripts tied to one machine, cluster or run;
  - internal IPs and hostnames; cluster, pod and namespace names; usernames;
  - home and scratch paths (`/raid/...`, `/home/<user>/...`, `/workspace/...`);
  - how to reach a machine: kubeconfig, SSH, tunnel or proxy recipes;
  - tokens, keys and anything else secret.
- **Keep them instead** in a private folder outside the repo: `~/fastrsi-lab/<engine>/`.
  If you must keep notes inside the checkout, use `.lab/`, which git ignores.
- **What `kb/` may say:** generic hardware ("4× GB200, aarch64, 956 GiB RAM") and measured
  numbers with their source described generically ("our 2026-10-08 startup run"), never how
  to reach the machine or where its files live.
- CI (`.github/workflows/pr-rules.yml`) greps added lines for common leak patterns. It is a
  pattern check, not a guarantee: you are still responsible for what you commit.

## Trust

- A result counts only if it comes with a receipt from a live lab that passes
  `template/livelab/conformance/`.
- Agents may write tools and checks, but never the code that writes receipts, and never edit
  a receipt by hand.
- Never claim a speedup without the receipt that measured it. Every number in `kb/` names its
  source.
- Each engine keeps a standing replay of real upstream PRs (`core/orchestrator.md`). A fix found
  by the replay is done when the replay shows it, not when CPU tests pass.
