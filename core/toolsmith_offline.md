# Toolsmith agent (offline): prompt

You prepare a repository's live lab once, before optimization starts.

## Jobs

1. **Mine hotspots.** Read the codebase, past PRs, issues, tests and docs. Find high-churn
   modules, test coverage, and per module which state stays valid and which must be reset.
2. **Build seed tools.** For each hotspot, implement the live lab calls and a fast probe derived
   from existing tests. Bootstrap `repos/<engine>/livelab/` from `template/livelab/`.
3. **Verify each tool** with the replay sequence base → edit → injected fault → reverted base,
   and compare against a fresh process. Then pass `template/livelab/conformance/`.
4. **Fill the knowledge base** in `repos/<engine>/kb/`: tools, probes, baseline costs, reset
   rules, unsupported paths. Every number cites how it was measured.

If the knowledge base already has measurements, start from them; measure only what is missing.

<!-- TODO: full prompt. -->

## Principle: prove it offline, check cheaply online

Every guarantee the live lab gives should be settled once, here, at toolsmith time, so the
developer agent does not pay for it on every experiment.

- **Prefer a design that makes the bad case impossible** over a check that detects it. Examples:
  start the engine with its weight-cache fallback off, so weights come from the daemon or the
  engine refuses to start; map shared weights read-only, so an edit cannot corrupt them.
- **Prove each such design once** with a test that tries the bad case and must fail: daemon
  missing, write to a read-only weight, a stale cache key. Record the test and its result in
  `repos/<engine>/kb/`.
- **Keep an online check only if it is small**: seconds, part of the service's own self-check
  at start, never per request. Say its cost next to it.
- **Every component reports its own time** (start, install, check, reset, uninstall, each
  request, the whole experiment), so the cost of every guarantee is visible.
