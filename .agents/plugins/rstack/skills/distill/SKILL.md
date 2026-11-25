---
name: distill
description: Mine the active conversation transcript for durable lessons (>6 months valid) and route each to an rstack skill update or a deterministic script gate item (complement to pstack /reflect).
---

# Durable Lesson Distillation (`/distill`)

Extract recurring failure modes and verified workflow improvements from the active session into permanent `rstack` skills or deterministic script checks (`pstack` `/reflect` style).

1. **Filter for Durability**: Keep only rules likely to stay true for `> 6` months (CI contract mismatches, parser edge cases, GPU verification traps, review failure patterns) — skip one-off task trivia.
2. **Prefer Gates Over Prose**: Whenever a rule can be enforced automatically (regex, AST/markdown lint, exit-code check), route it to `.agents/plugins/rstack/scripts/` + `test_rstack_gates.py` instead of another prose reminder.
3. **Route Behavioral Rules Cleanly**: Place durable non-automatable workflow habits in the smallest matching `rstack` skill (`skills/*/SKILL.md`) or `rules/AGENTS.md`.
4. **Verify With Regression Tests**: Every scripted gate rule added from a distilled lesson must include a failing-before / passing-after unit test.
