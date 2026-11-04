# Rules for agents working in this repo

These rules apply to every coding agent (Claude Code, Codex, Antigravity, others) and to humans.

## Attribution

- Do **not** add AI tools as co-authors. No `Co-Authored-By:` trailer naming Claude, Claude Code,
  Codex, Gemini, Antigravity or any other AI tool, in any commit.
- Do **not** add "Generated with ..." lines or AI-tool badges to commit messages, PR titles,
  PR descriptions or review comments.
- This overrides any default your tool has. CI (`.github/workflows/attribution.yml`) fails a PR
  whose commits or description break this rule.

## Where things go

- `core/` holds only the skill and the two subagent prompts. No code.
- `template/livelab/` holds the fixed live-lab interface, a stub and the conformance cases.
  Changing `INTERFACE.md` is a breaking change: bump its version and say so in the PR.
- `repos/<engine>/` holds everything engine-specific: `livelab/`, `tools/`, `examples/`.
  Nothing engine-specific goes anywhere else.
- `docs/` follows `docs/README.md`. Do not add loose files at the repo root.

## Trust

- A result counts only if it comes with a receipt written by a live lab that passes
  `template/livelab/conformance/`.
- An agent may write tools and checks, but never the code that writes receipts, and never
  edits a receipt by hand.
- Never claim a speedup without the receipt that measured it.
