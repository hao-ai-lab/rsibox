# kernel_vllm: the transaction kernel inside vLLM (prototype)

The engine side of `template/livelab/kernel`: one transaction swaps code on the workers and in
the engine core together, between the same two engine steps, with receipts. Prototype: TP 1
only so far.

| File | What it does |
|---|---|
| `host.py` | The engine-core host. Patches `EngineCoreProc` when imported with `FASTRSI_KERNEL_DIR` set: custom input-queue items run jobs on the busy-loop thread; a phase-2 job runs only when no batch is in flight (pause keep: `scheduler.has_requests` is shadowed to False while `batch_queue` drains). Runs the kernel's `StepGapCoordinator` over two participants (the workers, the engine core) and serves a file control channel. Records every step's duration. |
| `worker_ext.py` | `KernelWorkerExtension` (`--worker-extension-cls`): the participant protocol on each worker rank, reached by `collective_rpc("fastrsi_kernel", args=(method, payload))` issued from the busy loop. |
| `side.py` | One participant side (worker or core): a kernel `LocalParticipant` whose texts are analyzed in a child process (kernel `ProcessAnalyzer`; `FASTRSI_KERNEL_ANALYZE=inline` keeps them in-process), prepare in a background thread (`prepare_start` / `prepare_poll`, repeatable until commit, abort or undo), per-role plan filtering (`"on": ["worker"]`, for modules and repair cells; a cell gets the role's scope and the kernel's `journal`), a `captured_target` check on worker targets in model code (`vllm.model_executor.models` / `.layers`) when the worker's config runs CUDA graphs or torch.compile, with `zero_witness` on such targets routed to R1 at proof, a fault hook for tests, and replies that never raise (collective_rpc reply desync, kb/lessons.md). |
| `client.py` | Sends one op over the file channel (`ctl.json` -> `reply/<seq>.json`): `txn`, `prove`, `undo`, `status`, `ping`. Replies carry the kernel's receipt records; receipts also go to `receipts.jsonl` next to the channel. |
| `live_test.py` | The live golden tests below, against a running engine. |
| `test_adapter.py` | CPU tests of the adapter's own logic (worker-group prepare polling, the analyzer wiring): `PYTHONPATH=../../../template/livelab python3 -m unittest kernel_vllm.test_adapter -v` from `repos/vllm/livelab`. |

## How a transaction runs

```
client ── ctl.json ──► host thread (engine-core process)
  phase 1, engine keeps stepping:
    workers: collective_rpc(prepare_start) from the loop; each rank stages in a thread;
             collective_rpc(prepare_poll) every 20 ms until every rank is done (a rank
             that finished keeps answering with its result)
    core:    prepare in the host thread
    both:    parse, unit compile and outside-changes compare in a child process per side
  phase 2, one job on the busy loop, after in-flight batches land (pause keep):
    validate (workers + core) -> commit (workers + core), or abort; undo of a failed commit
    deadline D (default 5 s): each side answers by D/2 from its own lane thread
    (deadline_missed if the call still runs); a miss rolls the others back by D and
    poisons the host (txn refused, undo allowed); collective_rpc timeout = time left + 2 s
```

A role the plan gives no module (a worker-only edit, on the engine core) abstains: it votes
yes and its commit changes nothing. Group replies keep every rank (`ranks`), so proofs carry
witness counts per rank (`workers/rank0`, ...) and the txn receipt carries each rank's code
hashes and `commit_hashes.ranks_agree` (kernel README).

Run the engine with the host and the extension (the module and `kernel` on PYTHONPATH, a
sitecustomize that imports `kernel_vllm.host` when `FASTRSI_KERNEL_DIR` is set):

```
FASTRSI_KERNEL_DIR=<dir> VLLM_BATCH_INVARIANT=1 vllm serve <Qwen3-8B> --port <port> \
  --distributed-executor-backend mp \
  --worker-extension-cls kernel_vllm.worker_ext.KernelWorkerExtension
python3 -m kernel_vllm.live_test --url http://127.0.0.1:<port> --dir <dir> --restart-s <s>
```

`--distributed-executor-backend mp` puts the worker in its own process even at TP 1, so
`collective_rpc` crosses a real process boundary (the default `uni` runs the worker inside
the engine-core process).

## Live results (2026-11-05, 1x B200, Qwen3-8B, vLLM 0.33.0, TP 1, batch-invariant)

Our 2026-11-05 run of `live_test.py`. Real output, trimmed to the fields that matter.

**T1: worker + engine-core swap in one transaction, under 6 streaming requests (400 tokens each).**
The worker edit bans the first token of "Paris" and " Paris" in `PenaltiesState.apply`; the
engine-core edit adds a step counter to `Scheduler.schedule`.

```
"outcome": "committed",
"edit_to_run_s": 0.8781,          "prepare_s": 0.840898,   "gap_s": 0.000587,
"edit_to_finish_s": 6.448928,     "baseline_restart_s": {"seconds": 88.22, ...},
"witness": {"workers": {"...penalties:PenaltiesState.apply:88": 236},
            "core":    {"...scheduler:Scheduler.schedule:699": 238}},
"proof_findings": [],
"streams": 6, "lost_token_streams": [],
"banned_tokens_streamed_before_commit": 9, "banned_tokens_streamed_after_commit": 0,
"host_steps_before":     {"steps": 199, "median_s": 0.020318, "max_s": 0.042565},
"host_steps_during_txn": {"steps": 22,  "median_s": 0.01975,  "max_s": 0.221011},
"held_steps": 1,
"single_after_commit": " a city in the country of France. The capital of Italy is a city in ..."
```

- Edit to run: 0.88 s, against 88.2 s for a warm restart of the same engine (launch to first
  token, measured once). Edit to finish (all 6 streams done, witnesses read): 6.4 s.
- The engine stopped for 0.6 ms (`gap_s`), one held step. Every stream got all 400 tokens.
- Prepare took 0.84 s: workers 0.47 s (a thread in the worker process) and core 0.33 s (the
  host thread in the engine-core process). During it, one step took 221 ms against a median
  of 20 ms. Prepare runs outside the gap, but its heap scan holds the GIL of the process it
  runs in (the toy shows the same effect). This run cannot tell which process caused the slow step.

**T2: forced failure on the workers at validate.**

```
"outcome": "aborted", "status": "aborted",
"validate": {"workers": false, "core": true}, "committed_by": [],
"findings": [["worker", "participant_failed"]], "still_serving": true, "gap_s": 0.000752
```

**T3: undo of T1.** `"undo_ok": true, "status": "undone", "undo_s": 0.000683,
"identical_to_base": true`: the greedy single request gives the same text as before T1
(" Paris. The capital of Italy is Rome. ...").

**Locate without stalling (kernel v2.2), before / after, same engine session.** Our
2026-11-05 runs of `live_test.py --scan always` (the old locate: heap scans for functions
and suspended frames) and then the default (namespace locate), each a full T1-T3 pass:

| T1 | before (`--scan always`) | after (default) |
|---|---|---|
| prepare_s | 1.158 | 0.210 |
| max chunk gap across 6 streams during prepare | 248.7 ms | 37.2 ms |
| median chunk gap before the transaction | 19.9 ms | 19.9 ms |
| max engine step during prepare (host) | 206.6 ms | 26.3 ms |
| edit_to_run_s / edit_to_finish_s | 1.186 / 5.958 | 0.248 / 5.279 |
| gap_s | 0.0008 | 0.0004 |
| worker stage: locate / frames / total | 0.355 / 0.279 / 0.648 s (heap) | 0.00008 / 0.0001 / 0.012 s (namespace) |
| core stage: locate / frames / total | 0.168 / 0.169 / 0.469 s (heap) | 0.0004 / 0.0001 / 0.130 s (namespace) |

After the change the longest chunk gap while preparing is 1.9x the median (the goal was
about 2x). What core prepare still spends (0.13 s) is parsing and compiling
`scheduler.py` (3800 lines, about 10 ms per C call). Both runs: no lost tokens, no banned
token after the commit, witnesses on both sides, T2 aborted with nobody committed, T3
identical to base.

**Prepare compiles only the changed definitions; a hung commit has a deadline.** Our
2026-11-05 TP1 runs (same engine setup, control channel on a node-local tmpfs), three
alternating full T1-T3 passes per mode, `--compile module` (the whole `scheduler.py` and
penalties module, as before) against the default per-definition units:

| T1, engine-core side, 3 runs each | whole module | per definition |
|---|---|---|
| core parse + compile (stage timings) | 52.7-77.7 ms | 37.5-38.1 ms |
| core stage total | 105.6-174.2 ms | 86.7-89.1 ms |
| max chunk gap during prepare / median 19.9-22.6 ms | 31.8-57.0 ms (1.6-2.5x) | 30.3-37.1 ms (1.5-1.9x) |
| max engine step during prepare (host) | 29.3-45.8 ms | 29.9-34.0 ms |

What was left is one parse of the base and one of the patched text (about 14 ms each in the
engine process, single C calls) and the outside-changes comparison (two `ast.dump` calls). At
TP4 (8.6 ms median step) that made the longest step during a core prepare 3.4-3.95x the
median.

**Prepare analyzes the texts in a child process (kernel `ProcessAnalyzer`).** Each side starts
one child of its interpreter; `stage` sends it the two texts and gets back plain data and code
objects, so the parses, the unit compile and the compare no longer hold the engine process's
GIL. Our 2026-11-05 TP1 run, one engine session, 6 streams of 500 tokens, one transaction per
row then undo, 4 alternating runs each; "in-process" is the same plan with `"analyze":
"inline"` (the previous path):

| 4 runs each, median gap before 20.1-20.2 ms | in-process | child process |
|---|---|---|
| core-only plan (`Scheduler.schedule`): max chunk gap during prepare | 27.7-48.8 ms (1.37-2.42x) | 21.7-22.9 ms (1.07-1.14x) |
| core-only plan: max engine step during prepare (host) | 27.0-35.1 ms | 21.4-21.5 ms |
| core-only plan: prepare_s | 0.152-0.186 s | 0.152-0.170 s |
| worker-only plan (`PenaltiesState.add_request`): max chunk gap | 22.0-22.8 ms (1.09-1.13x) | 21.8-22.4 ms (1.08-1.11x) |

No lost tokens, every transaction committed and undone. The same session's T1-T3 and T4n/T4
passed as before (T1: max chunk gap during prepare 23.4 ms against a 20.2 ms median; both
sides report `"analyze": "process"`). The worker's penalties module is small, so its prepare
barely stretched steps either way. TP4 not rerun yet.

T4n / T4 (`--tests T0,T4n,T4`): while 6 streams of 800 tokens run, the worker hangs 4 s in
commit (fault hook, delay only):

| | no deadline (T4n) | deadline 1 s (T4) |
|---|---|---|
| outcome / host state | committed / committed | rolled_back / aborted, poisoned |
| gap_s / caller waited | 4.002 / 4.229 s | 0.501 / 0.632 s |
| max chunk gap during the txn (median 20.6 ms) | 4.027 s | 0.525 s |
| worker version log | switched (then undone) | never switched; late commit applied nothing |
| next txn / undo | aborted (stale base) / undone | refused (poisoned) / undone |

Both: no lost tokens, still serving, 0 steps started in the gap, identical greedy text after
undo. After the T4 undo every participant reports `undone`, the hung worker included (its late
commit applied nothing). Before, a hung rank stayed `aborted`; in the 4-rank run the ranks
that had committed said `undone` for the same transaction. At TP1 there is one rank, so `ranks_agree` is null (nothing to compare) and
`witness_uneven` cannot fire; both are covered by the kernel's CPU tests, not live.

## Findings from the live run

- `apply_all_penalties` (the dashboard demo's target) never runs on vLLM 0.33: the default V2
  model runner applies penalties in `vllm.v1.worker.gpu.sample.penalties.PenaltiesState.apply`.
  The first run swapped `apply_all_penalties`; the kernel reported `zero_witness` on the
  workers, and "Paris" was still generated. The swap was never counted as a pass.
- Phase 1 was not free on the engine: a heap scan in the engine-core or worker process slowed
  that process's steps while it ran (one 221 ms step in the first run). Fixed by the namespace
  locate (table above). Nested targets (closures) still need one heap scan.
- TP 4 ran once (2026-11-05, 4x GB200): the step-gap commit switched all ranks at the same
  worker step. It found a half-committed worker group, an undo cascade, witness counts summed
  across ranks, `ranks_agree` dropped from the receipt, and an 8 s engine stall from one slow
  rank; all four are fixed in the kernel now, but the fixes are not yet rerun at TP 4.
- A worker process that cannot answer at all (stuck in C code, dead) is caught only by the
  collective_rpc timeout. That leaves its late reply in the executor's response queue, where
  the next call reads it; the worker group then refuses every call, but the engine's own next
  step can still read the stray reply. Respawn the engine. Not exercised live.
- Before this change T2 aborted in prepare, not at the injected validate fault: the engine
  core voted no (`empty_change`, it had no module) and the workers were staged against the
  file while T1's text was live (`base_mismatch`). T2 now stages against T1's text and the
  core abstains; it aborts at validate (`participant_failed` on the worker).
- At TP4, 1 of 33 worker prepares aborted for nothing: a rank dropped its prepare result on
  its first finished poll, but the group polls every rank while any is pending, so ranks 0-2
  answered "no prepare started" while rank 3 still prepared. Nothing was committed. Fixed:
  a rank keeps its result until commit, abort or undo (`test_adapter.py`).
- The receipt writer (`kernel/receipt.py`) is still a proposal that needs a human owner
  (AGENTS.md, Trust).
