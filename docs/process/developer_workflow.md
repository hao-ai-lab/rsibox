# Developer and Multi-Agent Workflow SOP

This document specifies the developer and multi-agent operating procedure for rsibox across `core/`, `template/`, and `repos/<engine>/`. It governs how the top-level Engineering Manager ("Boss") delegates work to specialized subagents, enforces pre-review verification, orders and bundles pull requests for review, triages review findings, and reports empirical measurements.

---

## 1. Boss–Subagent Delegation Architecture

### Role of the Top-Level Agent (Engineering Manager / Boss)

The top-level agent operates strictly as an **Engineering Manager**:

- **Owner Interface & Planning:** Interfaces with the repository owner, clarifies architectural goals, and maintains the dependency DAG across branches, stacked PRs, and tracking issues.
- **Boundary & Contract Definition:** Establishes clear module boundaries (one boundary per PR, $\le 600$ lines of non-test code) and specifies exact invariants, input/output schemas, and test criteria before delegating implementation.
- **Quality & Conformance Review:** Audits subagent diffs, test coverage, leak-check results, and CI status before presenting any work to the owner.
- **Zero Heavy Inline Execution:** The Boss never runs long-running builds, GPU benchmarks, test suites, or multi-file code edits inline in the top-level context. All concrete implementation, test execution, and documentation drafting are delegated to isolated subagents working in dedicated git worktrees.

### Model Routing Policy

Subagents are routed by task complexity and specialization:

| Task Category | Assigned Model | Reasoning Effort | Responsibilities |
|---|---|---|---|
| **Complex Architecture & Kernel Coding** | **Claude Opus** | `medium` | Transaction kernel logic (`core/livelab/kernel/`), GPU/CUDA graph machinery, multi-rank coordination, root-cause debugging, and intricate algorithmic changes. |
| **Documentation, Prose Polish & Triage** | **Gemini 4** | Default / `inherit` | Technical documentation (`docs/`, `kb/`), variable-naming and style cross-checks, PR description formatting, and review comment triage. |
| **Post-Completion Trajectory Audit (`letitdo`)** | **Inherited (`inherit`)** | Default / `inherit` | Auditing worker subagent trajectories (`transcript.jsonl` and raw step outputs) before the Boss accepts any completion claim. |
| **Prohibited Models** | **Never use Gemini 3.8** | N/A | Gemini 3.8 is strictly prohibited across all workflows and subagent invocations. |

### Mandatory Boss $\to$ Worker $\to$ `letitdo` Trajectory Auditor Triangle

The Boss never accepts a worker subagent's completion claim on trust alone. Every delegated implementation, test, or benchmark task follows a closed three-role verification loop:

1. **Worker Execution:** The worker subagent executes the assigned code changes, unit/conformance tests, or supervised GPU runs in an isolated git worktree and reports its findings and commit/PR status to the Boss.
2. **`letitdo` Trajectory Audit:** Immediately upon receiving a worker's completion report, the Boss spawns or messages the dedicated `letitdo` auditor subagent (`Model: "inherit"`).
   - `letitdo` inspects the worker's actual conversation transcript (`transcript.jsonl`) and raw step command outputs (`steps/*/output.txt`).
   - `letitdo` verifies whether claimed GPU or CPU commands were genuinely executed in the trajectory or merely mocked/simulated, whether any test failures or skipped paths were glossed over, and whether the 3-part GPU verification contract is completely and honestly filled out.
   - If any gap, unrun path, naming violation, or overstated claim is found, `letitdo` directly messages the worker subagent (`send_message`) with concrete remediation instructions and waits for the worker to fix and re-verify.
3. **Auditor Sign-Off:** The Boss accepts the task and presents the PR to the repository owner only after `letitdo` confirms that the trajectory matches every claim in the PR description.

### Cluster Scheduler & Allocation Discipline

Before launching any multi-GPU or single-GPU experiment on a managed compute cluster, agents must verify the cluster's native allocation mechanism rather than guessing scheduler commands:

- **Ray-Managed Clusters vs. Slurm Allocations:** On clusters managed through shared Ray worker tickets (`raycluster`), jobs must be launched inside the assigned Ray worker container/allocation under the supervised launcher rather than invoking `srun` or `sbatch`. Conversely, on Slurm-managed clusters, commands must run inside the active Slurm allocation. Never assume Slurm commands (`srun`/`sbatch`) exist on a Ray-ticket cluster, and never run GPU workloads on login nodes.

### Mandatory Pre-Presentation Cross-Check Pass

Every code and documentation PR must pass an independent cross-check before being presented to the repository owner:

1. **Variable Naming Hygiene:** Scan all added and modified code for cryptic 1–2 letter abbreviations and replace them with descriptive, pronounceable English words:

   | Prohibited Abbreviation | Required Descriptive Name |
   |---|---|
   | `ws` | `world_size` (or `workspace`) |
   | `sel` | `selection` |
   | `st` | `status` (or `stat_result` / `state`) |
   | `bcls` | `base_classes` |
   | `mod` | `module` |
   | `cb` | `callback` |

2. **Tone & Engineering Prose:** Verify that comments, docstrings, and markdown documents use sober, factual engineering prose. Strip marketing language, speculative claims, and unverifiable superlatives. Every performance or behavior claim must cite a concrete code path, test name, or measured receipt.

---

## 2. Pre-Review SOP & Dependency-Aware PR Bundling

### Step 1: Pre-Flight Checklist (Before Requesting Owner Review)

No PR may be presented to the owner until all four pre-flight gates pass:

- [ ] **Dependency DAG Verification:** Verify parent/child branch relationships across active stacks so no PR is presented with an unmerged or conflicting base.
- [ ] **Rebase onto Latest `origin/main`:** Rebase the branch cleanly onto `origin/main` and confirm zero merge conflicts.
- [ ] **CI & Conformance Green:** Verify that all GitHub Actions checks (PR rules, leak check, CPU unit tests, and the sandbox verification suite (`c0`–`c10`)) pass on the pushed head SHA.
- [ ] **Zero Unresolved Review Threads:** Confirm that every review comment from prior rounds has been fixed, rejected with counter-evidence, or deferred with a linked tracking issue and resolved.

### Step 2: Review Priority & Bundling Strategy

To respect reviewer bandwidth and keep architectural changes easy to audit, PRs are presented in two priority tiers:

1. **Priority 1 — Core Code PRs (Architectural & Kernel Logic):**
   - Presented **first**, strictly ordered by their dependency DAG from bottom to top.
   - Presented in focused batches of **1–2 PRs at a time**.
   - Accompanied by clickable file and line links (`[file.py](file:///path/to/file.py#L10-L40)`) pointing directly to the core logic decisions and invariants that require owner sign-off.
2. **Priority 2 — Bundled Test & Documentation PRs:**
   - Once the core code PRs in a stack are reviewed and merged, related unit-test expansions, golden-case docstrings, and documentation updates (`docs/`, `kb/`, `README.md`) are consolidated into a **single bundled Test PR** or **single bundled Documentation PR** rather than trickled out as fragmented small PRs.

### Step 3: Review Comment Triage SOP

When the owner or an opt-in reviewer posts findings on a PR, triage each comment by severity:

| Priority | Category | Required Action |
|---|---|---|
| **P0 / P1** | Correctness bugs, broken invariants, contract mismatches, security/leak issues, or architectural blockers | Fix immediately in the active PR branch before merge, add or update a test that fails without the fix, and reply with `Fixed in <sha>` and the test name. |
| **P2** | Non-blocking style nitpicks, minor naming suggestions outside touched lines, optional hardening, or out-of-boundary follow-ups | Log the item in a centralized tracking issue, reply inline on the review thread with a direct link to the tracking issue, and resolve the review thread so non-blocking comments do not obscure P0/P1 gates. |

---

## 3. Empirical Honesty & Two-Tier Reporting Standards

### Strict Empirical Honesty

1. **No Cherry-Picking:** Report all runs, including negative results, run-to-run variance, shared-node contention, and regressions. If a candidate optimization regresses full evaluation or alters numerical output beyond the workload gate, state that plainly.
2. **Cold-Cache vs. Warm-Cache Transparency:** Always separate cold-cache measurements (first-time container pull, cold OS page cache, cold Triton/Inductor/FlashInfer compilation) from warm-cache measurements (warm OS page cache, path-stable disk caches, resident holder daemons, or in-process live swaps).
3. **Witness & Ledger Verification:** Never claim an installed edit took effect without checking the participant witness ledger (`core/livelab/kernel/witness.py`) and receipt logs. Surface `zero_witness` findings (`0` live calls, or capture-only calls without post-commit graph replay) and fallback restarts explicitly rather than counting a silent no-op as a pass.

### Mandatory 3-Part GPU Verification Contract on Every PR

Every pull request that modifies `repos/<engine>/` or `core/livelab/` must include three explicit, non-empty verification bullets in its `## Loop record` (enforced by `.github/workflows/pr-rules.yml`):

1. **`- Ideal full GPU test:`** State what the complete end-to-end GPU verification for this change looks like—including the target GPU count/topology (e.g., `1× GB200`, `4× GB200 TP4`, or `2×4 GB200`), the real model and workload (e.g., `Qwen3-8B` or `DeepSeek-V3` under `vllm serve`), and the expected observable metric or bit-exactness invariant.
2. **`- Actual tests run:`** List the exact GPU or CPU commands actually executed in the agent trajectory, including test counts, pass/skip breakdowns, and measured wall-clock timings.
3. **`- Why unrun GPU paths were skipped:`** Provide an explicit, honest engineering justification for why any portion of the ideal full GPU test was not executed in this PR (for example, pure-Python AST/state-machine logic verified exhaustively on CPU, or multi-node hardware contention), or write `None — full GPU test executed` when the full GPU test ran end to end.

### Two-Tier Reporting Format

Every performance investigation, startup audit, or benchmark campaign produces two complementary deliverables:

1. **Tier 1 — Persistent Detailed Technical Report:**
   - Recorded in the appropriate knowledge-base file (`repos/<engine>/kb/costs.md`, `docs/`, or a tracking issue/PR body).
   - Includes generic hardware and topology metadata (e.g., `1× GB200`, `4× GB200 TP4`), software pins, exact CLI flags, full phase-by-phase breakdown tables, witness/receipt summaries, and root-cause analysis.
2. **Tier 2 — Crisp Executive Summary:**
   - Presented in the PR summary or chat response to the owner.
   - Leads with the bottom-line numbers (baseline vs. new latency/throughput and $\Delta$ or speedup factor), states the single primary root cause, and highlights any concrete decision or approval needed from the owner.
