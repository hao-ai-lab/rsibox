<!-- Title: <type>(<scope>): <summary>. Rules: AGENTS.md. -->

## What and why

<!-- One or two sentences. Link the issue or parent PR if stacked. -->

## Evidence

<!-- Receipts, test output or a walkthrough with real output; "none" for docs-only changes.
     Then one line on what this PR does not show. -->

## Loop record

<!-- Required when this PR changes repos/<engine>/ or core/livelab/. Delete the section otherwise.
     Every field needs text; write "none" when it does not apply. CI checks this.
     The three GPU verification fields (Ideal full GPU test, Actual tests run, Why unrun GPU paths were skipped)
     are mandatory whenever repos/<engine>/ or core/livelab/ changes. -->

- Engine: <!-- e.g. vllm -->
- Upstream PRs: <!-- e.g. vllm-project/vllm#62240, and its base commit -->
- Function groups: <!-- OP, GRAPH, STATE, RESOURCE, NATIVE, RANKS, restart; per changed function -->
- Rule added: <!-- the smallest general rule this needed, and where it now lives -->
- Overheads measured: <!-- phase, seconds, source (receipt or kb/measurements file) -->
- Receipts: <!-- paths, or none -->
- Not verified: <!-- what a reader should not assume -->
- Ideal full GPU test: <!-- complete end-to-end GPU test for this change: GPU count, model/workload, expected metric/behavior -->
- Actual tests run: <!-- exact GPU or CPU commands executed in the trajectory, test counts, and measured timings -->
- Why unrun GPU paths were skipped: <!-- explicit justification for any unrun part of the ideal GPU test, or "None — full GPU test executed" -->
