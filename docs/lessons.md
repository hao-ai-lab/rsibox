# Lessons from building a live lab

What building FastRSI for vLLM taught us about the method, for any engine. Each lesson says what
happened, the rule it produced and how to apply it. Engine-specific lessons live in
`repos/<engine>/kb/lessons.md`; the orchestrator's working rules in `core/orchestrator.md`. Edit
lessons in place when a rule changes; this is not a log.

### 1. Draw restart boundaries before writing mechanisms

**What happened.** The first prototype grew seven separate ways to change a running engine
(in-place model rebuild, engine-core transactions, module patching, install by qualified name, a
save watcher choosing among them, a native side library, a second adapter path), plus startup
monkeypatches: about 45k lines and 245 run records. Its records show 51 hand repairs, and nearly
every hard bug was in aligning long-lived state in place (new fields, constructors, caches,
captured graphs, copies in other processes), not in swapping a function body. The v1 redesign keeps
one mechanism per restart tier: T0 swap, T1 component restart, T2 warm restart. It is about 7.7k
lines of code plus 4.4k of tests, and on the same 19 upstream PRs it needed a full restart on 0
(prototype: 2) and went live in place on 8 (prototype: 6; 7 once constructor edits always block).

**Rule.** Swap only code that holds no state of its own. Restart the lowest component that owns the
invalidated state, with expensive state kept outside it. Detectors classify; they do not repair.

**Apply it.**
- Before building, draw the engine's process tree and its state holders, and give each boundary a
  measured cost (`repos/<engine>/docs/architecture.md`).
- When a new special case needs an in-place repair, route it to a restart tier instead.

### 2. A toy proves the protocol; only a live engine proves the engine

**What happened.** The kernel passed all 11 conformance cases on the CPU toy lab (about 1.5 s per
PR). The toy is there because CI has no GPU and faults (crash, hang, lost rank, tampering) must be
deterministic. The first replay of real PRs through v1 on GPUs still found 4 bugs the toy could not
show (lesson 5). The fix PRs that followed (#52–#55) passed their CPU tests; only the replay rerun
on main proved them on an engine, and it also showed the trade-off one of them brought (lesson 4).

**Rule.** CPU tests are a precondition. A change on an engine path is done when a live run shows
it.

**Apply it.**
- Mark every test `needs: cpu` or `needs: gpu`; write "pending replay" in a PR's "Not verified"
  until its GPU receipt exists.
- Schedule the GPU run in the same turn the CPU work lands; never leave granted GPUs idle.

### 3. Find the cost ladder before optimizing

**What happened.** A one-line scheduler edit on one GB200 took 228 s to check with a restart; setup
was 204 s of it. Keeping one more piece of state at each step cut the check to 157 s (autotune
config), 100 s (+ weights), 58 s (+ CUDA graphs) and 23.6 s (+ the live process); a cheaper check
then gave 14.1 s and 2.4 s, 95× in all (our earlier GB200 measurement, warm on-disk caches; see
`repos/vllm/kb/costs.md`). Our 2026-11-06 start-up
runs put cold vs warm at 181.9 vs 54.9 s (eager) and 223.6 vs 59.1 s (CUDA graphs). Two intended
cold runs failed, so the runs that copied their empty caches were cold in effect while labeled
warm.

**Rule.** Measure each piece of state that can be kept separately; that list is the plan. Graphs
and compilation, not Python, decide edit-to-run.

**Apply it.**
- Record start-up per phase from process start, and label every run by what its caches really
  held.
- Reuse a warm cache root in place. A copied root is warm only once a start from it rebuilds
  nothing: some tools key their caches on absolute paths.
- Keep the cost map in `repos/<engine>/kb/costs.md` and pick work by expected seconds removed.

### 4. Replay real upstream PRs as the end-to-end proof

**What happened.** We replayed 19 merged upstream vLLM PRs through v1 on GB200 (two models; TP1
and TP2; eager and CUDA graphs; our 2026-11-06/07 replay run). On the first pass 8 went live in
place at 0.38–1.94 s edit-to-run and 11 through a fast restart (T1) forked from a warm process; a
warm plain restart took 43–231 s. It found 4 bugs (lesson 5). After their fixes (#52–#55) we reran
the affected PRs and the in-place ones on main:
- vllm-project/vllm#61323 went from a plain-restart fallback (113 s eager, 195 s graph) to a forked
  restart (9.2 s, 44.9 s);
- vllm-project/vllm#59941 was refused before commit instead of committing with witness 0;
- vllm-project/vllm#62872 now undoes back to base, but costs 16.6 s (eager) and 51.2 s (graph)
  instead of 0.38 s, because a constructor edit now always goes to T1;
- the 7 other in-place PRs were unchanged.

Across both passes T1 took 9.2–68.8 s on Qwen3-8B and 115–185 s on Qwen3.5-35B-A3B-FP8 at TP2 with
CUDA graphs. Real PRs chose the hard cases for us: constructors, decorators, Triton kernels, op
registries.

**Rule.** The replay is the standing gate for each engine. A rule or fix counts once the replay
shows it.

**Apply it.**
- Keep a fixed replay set per engine, run it on main after each merge batch, and diff its outcome
  table against the last run.
- Choose each PR's workload from its diff and prove the base reaches the changed lines first; a
  zero witness means "not reached", never a pass.

### 5. Code that runs at build, import or capture time is not reached by a swap

**What happened.** The bugs only GPU runs found came from code that runs once, at build, import,
compile or capture time, or from engine state read as ours:
- a Triton kernel compiles from its source text, so swapping its Python code changed nothing
  (prototype, 2026-11-05; v1 routes Triton edits to a restart);
- an edited constructor left the objects built at start-up unchanged (witness 0; fixed in #53);
- a graph-split decorator runs at import and during capture, so captured graphs kept the old
  decision (witness 0 on both ranks and the engine core; #54);
- re-importing an edited module for a restart ran an op registration a second time; it refuses
  duplicates, and every forked server died (#52);
- names the engine imported lazily during the workload were read as our undo leftovers (#55).

The last four came from the first v1 replay (our 2026-11-06/07 run); the rerun on main showed each
fix working.

**Rule.** Before a swap, ask when the target runs. Build-, import- and capture-time code goes to a
restart tier, and registries are refreshed together with their modules.

**Apply it.**
- Treat a zero witness after a committed swap as a finding with a route, never as silence.
- Tell state the engine created since base from state the lab must undo.

### 6. Programs schedule GPUs; agents judge results

**What happened.** With LLM agents as GPU runners, 10 of 29 GPU slots sat idle while 40 jobs
waited: an agent sleeps while its job runs and frees the GPU only on its next turn. Scripted
runners that own a node's GPUs did better: two closed 33 and 34 jobs in about 2 h 20 min each, and
on one, from mid-afternoon on, no GPU sat idle more than 2–3 min. The runner loop now starts every
job in the background and checks for free GPUs every 60 s.

**Rule.** A program claims GPUs and runs scripted steps; agents take only jobs marked `needs_agent`
(`core/orchestrator.md` rule 2).

**Apply it.**
- Build the queue, the runner and a status view before feature work.
- Give agents failures and findings to judge, not GPU slots to babysit.

### 7. Share GPUs through atomic locks, checked and taken in one step

**What happened.** Agents sharing nodes hit: one GPU locked under two spellings of the node name; a
leader that assumed a node was idle while another agent had locked all four GPUs two minutes
earlier; a GPU taken in the middle of a measurement run; another agent's server answering `/health`
on the same port, so a launcher reported a dead engine healthy after 0.01 s; and a pool daemon that
died and left four warm engines holding about 150 GB per GPU at 0% use for about a day. The cluster
also reaps GPU processes that run without an allocation (after about 10.5–12 min), and one
process-group kill took a whole node down.

**Rule.** One lock per GPU under one canonical name, taken atomically; check and take in the same
step; a holder that dies must not keep its GPUs (`core/orchestrator.md` rule 16).

**Apply it.**
- Read locks and the port before launch, never "it looked idle".
- Kill only processes or groups you created.

### 8. Merge a stack only when the PR's base is main

**What happened.** With stacked PRs, a PR whose base had not yet moved to `main` merged into the
old stack branch. Five PRs (#24, #25, #30, #38, #40) landed off main and needed two landing PRs
(#36, #56). Retargeting is asynchronous; an 8 s wait was not enough.

**Rule.** Read the base right before each merge; after a batch, check the files are on main.

**Apply it.**
- `gh pr view <n> --json baseRefName` must say `main`; afterwards `git ls-tree origin/main
  <path>`.

### 9. Review findings are advice; check each round at the new head

**What happened.** Agent reviewers were often right and sometimes wrong. The review-CI PR (#27)
took 3 rounds, and round 2 re-raised findings the author had reported as fixed: the reply was not
evidence. A whole-codebase review used five reviewer roles across Claude, Codex and Antigravity,
and a checker verified each finding against the code before it was listed.

**Rule.** Every finding gets fix, reject (with counter-evidence) or defer (with an issue). Fix only
high and medium findings that reproduce. Stop after 3 rounds (AGENTS.md, "Handling review
findings").

**Apply it.**
- Verify a re-raised finding against the code at the new head, not against the last reply.
- Have one agent check another's findings before anyone acts on them.

### 10. Keep slow review off the merge path

**What happened.** Agent code review in CI ran one PR at a time, about 7 min each, and every push
waited for it. It is now opt-in: the owner comments `/review` or `/describe` (#49, #51). Findings
on merged PRs were answered later in batch fix PRs, one per boundary (#46–#48).

**Rule.** Nothing slow gates a merge unless the owner asks for it.

**Apply it.**
- Answer review findings in one fix PR per boundary, with the verdict table in its description.

### 11. Talk in records

**What happened.** Prose relays between agents bent facts: a 5 s deadline became 1 s, "verified on
4 GPUs" was read as "in progress", and a retyped token was wrong.

**Rule.** Messages carry record paths, commit hashes and copied values; prose is for judgment
(`core/orchestrator.md` rule 6).

**Apply it.**
- Link the record instead of retelling it; never retype a number, a hash or a secret.
