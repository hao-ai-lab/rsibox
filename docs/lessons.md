# Lessons from building a live lab

What building rsibox for vLLM taught us about the method, for any engine. Each lesson says what
happened, the rule it produced and how to apply it. Engine-specific lessons live in
`repos/<engine>/kb/lessons.md`; the orchestrator's working rules in `core/agents/orchestrator.md`. Edit
lessons in place when a rule changes; this is not a log.

### 1. Draw restart boundaries before writing mechanisms

**What happened.** The first prototype grew seven separate ways to change a running engine
(in-place model rebuild, engine-core transactions, module patching, install by qualified name, a
save watcher choosing among them, a native side library, a second adapter path), plus startup
monkeypatches: about 45k lines and 245 run records. Its records show 51 hand repairs, and nearly
every hard bug was in aligning long-lived state in place (new fields, constructors, caches,
captured graphs, copies in other processes), not in swapping a function body. The v1 redesign keeps
one mechanism per restart boundary: in-place swap, component restart, warm restart. It is about 7.7k
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
place at 0.38–1.94 s edit-to-run and 11 through a fast restart forked from a warm process; a
warm plain restart took 43–231 s. It found 4 bugs (lesson 5). After their fixes (#52–#55) we reran
the affected PRs and the in-place ones on main:
- vllm-project/vllm#61323 went from a plain-restart fallback (113 s eager, 195 s graph) to a forked
  restart (9.2 s, 44.9 s);
- vllm-project/vllm#59941 was refused before commit instead of committing with witness 0;
- vllm-project/vllm#62872 now undoes back to base, but costs 16.6 s (eager) and 51.2 s (graph)
  instead of 0.38 s, because a constructor edit now always routes to a forked restart;
- the 7 other in-place PRs were unchanged.

Across both passes a forked restart took 9.2–68.8 s on Qwen3-8B and 115–185 s on Qwen3.5-35B-A3B-FP8 at TP2 with
CUDA graphs. Real PRs chose the hard cases for us: constructors, decorators, Triton kernels, op
registries.

**Rule.** The replay is the standing gate for each engine. A rule or fix counts once the replay
shows it.

**Apply it.**
- Keep a fixed replay set per engine, run it on main after each merge batch, and diff its outcome
  table against the last run.
- Choose each PR's workload from its diff and prove the base reaches the changed lines first; a
  zero witness means "not reached", never a pass.
- After a fuzzy patch apply onto an older engine commit, check that each hunk landed in the
  function upstream put it in and added the same lines, then compile on CPU before touching a GPU.
  In the Megatron replay (#86), two upstream diffs (#8246, #8524) applied with fuzz 3 into the
  wrong function; compilation caught those in 0.1 s, and a misplacement that still compiles is
  caught only by the function-placement check.

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
(`core/agents/orchestrator.md` rule 2).

**Apply it.**
- Build the queue, the runner and a status view before feature work.
- Give agents failures and findings to judge, not GPU slots to babysit.

### 7. Share GPUs through atomic locks, checked and taken in one step

**What happened.** Agents sharing nodes hit: one GPU locked under two spellings of the node name; a
leader that assumed a node was idle while another agent had locked all four GPUs two minutes
earlier; a GPU taken in the middle of a measurement run; another agent's server answering `/health`
on the same port, so a launcher reported a dead engine healthy after 0.01 s; a pool daemon that
died and left four warm engines holding about 150 GB per GPU at 0% use for about a day; and a
Megatron harness killed while holding the restart-GPU lock, leaving the lock and an idle candidate
job behind so the next user waited 22 minutes (#86). The cluster
also reaps GPU processes that run without an allocation (after about 10.5–12 min), and one
process-group kill took a whole node down.

**Rule.** One lock per GPU under one canonical name, taken atomically; check and take in the same
step; a lock records its holder's PID so a waiter breaks it when that process is gone, and a holder
that dies must not keep its GPUs (`core/agents/orchestrator.md` rule 16).

**Apply it.**
- Read locks and the port before launch, never "it looked idle".
- Record the holder's PID in every lock, break locks whose holder process is gone, and give every
  started candidate job an explicit exit.
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
(`core/agents/orchestrator.md` rule 6).

**Apply it.**
- Link the record instead of retelling it; never retype a number, a hash or a secret.

### 12. Most real changes need a restart, so make restarts cheap, and name them in words

**What happened.** Replaying real upstream PRs in three engines, about half could not be swapped
live: they change constructors, module-level defaults or environment registries. In FastVideo,
three of four needed a restart at 86-107 s each (#85); in Megatron, a code restart cost 79-89 s
(#86). We then built cheaper restart actions and measured each (our 2026-11-07 runs, GB200; sources
#87 for FastVideo, #82, #83 and #88 for Megatron). Each cell says what the time runs to:

| Action | FastVideo MiniMax H3 video model, 1 GPU | Megatron, Qwen3-14B architecture, TP 2 |
|---|---|---|
| live swap | 0.06 s to installed | 0.07-0.6 s to installed |
| in-place rebuild (constructor edits, new fields) | 0.19-1.27 s to rebuilt | not built |
| zygote respawn with a weights holder | 4.8-5.6 s to ready | not built |
| flag or code restart from the zygote | 14.1-15.3 s to ready | 15-16.5 s to first step (code: 19 s) |
| environment restart, environment on local disk | 51.4 s to ready | 29.3-29.7 s to first step |
| plain restart | 43-45 s to ready (respawn), 69.5 s (warm restart) | 58 s to first step |

What stays: work that only a live process keeps, such as communicators, allocator and
optimizer state, and kernel plans (9-17 s of Megatron's first step; disk caches saved 1.3-1.7 s).
Labels like "T1" hid these costs from readers and agents alike.

**Rule.** Give every engine a ladder of restart actions, each priced, and name each action in
words with its overhead and whether that figure is measured or anticipated: "code restart from the
zygote (19 s to first step, measured)", never "T1".

**Apply it.**
- Build the ladder from the start-up breakdown: one action per piece of state that can be kept
  (imports, weights, compiled caches, the live process).
- A zygote that has imported the engine must re-import the edited package in each child, or
  refuse; otherwise children run old code.
- Anything kept across a restart (a weights holder, a cache) is stamped with what it depends on
  and refuses when that changes, with a finding naming the file.
- An in-place rebuild must release what the replaced modules held: CUDA graph pools held by old
  functions ran a GPU out of memory on the first use after a rebuild.
- Show agents the priced menu in the same words. With it, agents chose flag restarts for
  configuration ideas and live swaps for code; without it they rarely chose a restart.

### 13. A gain measured in one long-running lab is a hypothesis until a fresh process repeats it

**What happened.** (Sources: #79, #87.) A FastVideo optimization run (an agent proposing edits, each kept only if faster) reported 4.21 → 3.947 s per video inside one engine
instance, every kept edit at least 1% faster than the run before it in paired measurements. Re-measured in fresh processes with alternating pairs, the
same set ran about 4.18 s, within noise of the start. A forked Megatron worker ran 2-11% slower in
steady state than a launcher-started one with the same CPU binding, and an unbound job ran at
0.72x of a bound one.

**Rule.** Keep a change only after fresh processes confirm it, paired and alternating, with
identical placement. Compare throughput only between processes started the same way.

**Apply it.**
- Bind CPUs and memory to the GPUs' socket in every timed run, and report the binding.
- Use forked workers for start-up cost and correctness checks, not for throughput comparisons
  against launcher-started ones.
- Before a final number is claimed, re-measure the whole kept set from a fresh start.

### 14. Test a second input shape before keeping a speedup

**What happened.** (Source: #79.) Two kept FastVideo edits passed the correctness gate at one sequence length. A
second length showed one recompiling at every new length, and the earlier per-layer CUDA graph
patches holding 74.9 GiB, so a 243-frame video after a 124-frame one ran out of memory.

**Rule.** The correctness gate runs at least two input shapes in one process, and counts recompiles.

**Apply it.**
- Add a second shape to the workload's correctness gate (`kb/workload.md`).
- Run with recompile logging and report the count; a change that recompiles per shape is
  benchmark-only unless guarded by a flag.
- Put memory that grows per shape (graph pools, caches) behind an opt-in flag with a budget.

### 15. A second reviewer checks every kept change

**What happened.** (Sources: #79, #86.) A reviewer running a different model rated 10 of 11 retained FastVideo edits sound and one
wrong: it argued the diff could not produce the measured gain. A GPU A/B agreed, and the
re-measurement also withdrew the run's headline number. All four retained Megatron edits were rated sound,
with named risks we recorded.

**Rule.** Before a gain is reported, a reviewer that did not write the change checks each kept
change against its measurement and rates it sound, suspect or wrong. This rates a kept change; it
is separate from the fix / reject / defer answers to review findings in `AGENTS.md`. A "wrong"
rating is settled on the GPU, not by argument.

**Apply it.**
- One review call per keep, with the diff, the hypothesis, the gate result and the paired timings.
- Post the verdict table on the PR that carries the retained edits, risks included.

### 16. Stop a harness only between evaluations, and reconcile live transactions at start

**What happened.** (Source: #86.) In Megatron's 2026-11-07 hill-climb, a harness killed in the
middle of an evaluation (14:10) left the candidate's swap installed in the base lab. When the stray
transaction was found and undone an hour later (15:16), undoing it also rolled back the two kept
swaps installed after it, because rollback unwinds from the target transaction through the newest.

**Rule.** Stop a harness only between evaluations, and at start compare the lab's live transactions
with the kept record before running a new candidate.

**Apply it.**
- Have the harness check a `STOP` file between evaluations instead of killing it mid-evaluation.
- At harness start, read the lab's active transactions (`status()`), compare them against the kept
  record, and undo any unknown transaction immediately before new swaps stack on top of it.

### 17. A per-call file-system walk is a per-participant tax

**What happened.** On a 21-process verl job every `status`, `install` and `uninstall` took 24-40 s.
Each participant's fingerprint resolved the loaded reward file by calling `os.path.realpath` on every
`sys.modules` entry (thousands of stats on a network file system, 0.2-0.9 s per process), and
`Session.status` asked every participant twice. The same job answered `status` in 2.5-5.6 s once
realpath results were memoized and `status` reused the round `verify()` had just taken. A second
shared-file-system race followed: the gate wrote the response before advancing `ready.json`, so a
fast client read the old request number and failed with "already pending".

**Rule.** Anything the kernel does once per participant per call must not touch the file system
proportionally to the process's size, and a file protocol must publish "next" before "done".

**Apply it.**
- Measure a verb's cost per participant (`fingerprint_s` in a verl `status` with `timing: true`) before
  calling a lab slow; multiply by the participant count.
- Memoize path resolution per process; a module's file never moves.
- In file-based gates, write the next request number first, then the response, and let the client
  wait out a briefly stale number instead of failing.
