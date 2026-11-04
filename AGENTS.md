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

- **Framework:** what works for any engine. `core/`, `template/`, `docs/`, `AGENTS.md`,
  `.github/`.
- **Artifact:** what the loop produced for one engine. `repos/<engine>/`. It shows the current
  state only: edit `kb/` in place, never append history.
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
  An Antigravity reviewer then polishes the title and description in CI, and the change is
  applied only if every fact is kept (`.github/review/`). Run it locally before opening a PR:
  `python3 .github/review/review.py --pr <n> --apply`.
- **Loop record:** a PR that changes `repos/<engine>/` is a loop iteration. Its description
  fills the "Loop record" section: engine, upstream PRs, function groups, the rule added,
  overheads measured with their sources, receipts, and what is not verified. Write "none" when
  a field does not apply; do not delete it. CI checks that every field has text, not that the
  text is true.
- **One iteration per PR.** Keep it small enough to review in one sitting. Stack PRs when one
  step depends on another.
- **Merge** with squash, keeping the PR description as the commit body, so the loop record is
  also in `git log`.
- **Attribution:** do not add AI tools as co-authors. No co-author trailer naming Claude,
  Claude Code, Codex, Gemini, Antigravity or any other AI tool, in any commit. No
  "Generated with ..." lines or AI badges in commits, PR titles, descriptions or comments.
  This overrides your tool's defaults. CI checks commits and the PR text.

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

## Where things go

- `core/`: the skill and the agent prompts (developer, toolsmith offline and online). No code.
- `template/livelab/`: the fixed live lab interface, a stub, and the conformance cases.
  Changing `INTERFACE.md` is a breaking change: bump its version and say so in the PR.
- `repos/<engine>/`: everything about one engine: `kb/`, `livelab/`, `tools/`, `examples/`.
- `docs/`: docs about FastRSI itself, following `docs/README.md`. No loose files at the root.
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
