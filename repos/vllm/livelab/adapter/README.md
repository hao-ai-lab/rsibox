# Adapter: the transaction kernel inside a live vLLM engine

The engine side of the transaction kernel (`core/livelab/kernel`): one transaction swaps
code on every worker rank and in the engine core together, between the same two engine steps,
with receipts. This folder holds the
live code swap path (in-place function-body swap) and selective recapture of CUDA graphs in the
commit's gap (`recapture.py`). Recompiles and warm weight reload are not here: an edit that
needs one is a finding, not a silent swap.

## Contract

- **Host** (`host.py`). Imported in the engine-core process with `FASTRSI_KERNEL_DIR=<dir>`, it
  patches `EngineCoreProc` so custom input-queue items run jobs on the busy-loop thread, and
  runs the kernel's `StepGapCoordinator` over two participants: the workers (`WorkerGroup`,
  every rank behind one `collective_rpc`) and the engine core (an `Endpoint` in its own
  process). It serves a file control channel (`ctl.json` -> `reply/<seq>.json`) with the ops
  `txn`, `prove`, `undo`, `status`, `pstatus`, `fingerprint`, `versions`, `at_base`, `ping`,
  `graphs`, `recapture_extend` (alias `r1_extend`).
- **Phase 1** (prepare) runs in a thread on every side (`side.Side`): the host sends
  `prepare_start` and then `prepare_poll` every 20 ms until no rank is pending, each one
  `collective_rpc` from the loop, so the engine keeps stepping while the ranks stage. A rank
  that finished early repeats its vote until the transaction ends. **Phase 2** (validate, commit or abort, undo of a failed commit) is
  one loop job that runs only when no batch is in flight; while it waits, nothing new is
  scheduled ("pause keep"). Every collective_rpc is issued from the loop thread, so every rank
  handles it between the same two steps.
- **Per-rank replies.** A worker-group reply keeps every rank's own reply (`ranks`), so
  receipts carry each rank's code hashes and witness counts (`workers/rank<r>`). The group
  expects a reply from every rank this engine core reaches; at data parallel >1 it also
  expects the other DP groups, which it cannot reach, so a transaction there is refused with a
  blocking `participant_missing` (DP>1 unsupported, said plainly).
- **Never raise across the RPC** (`worker_ext.Endpoint`). A method that raises on one rank
  leaves the other ranks' replies queued in the executor, and every later call reads the reply
  of the call before it. Every call returns an ok-dict tagged with role, rank and pid.
- **Roles.** A plan module or repair cell runs only on the roles in its `"on"` list (default:
  all). A role with nothing to do abstains (votes yes, changes nothing) instead of vetoing with
  `empty_change`.
- **Deadline.** Phase 2 has a deadline (`FASTRSI_KERNEL_PHASE2_DEADLINE`, default 5 s; 0 turns
  it off). Each side runs its phase-2 calls on its own lane thread and answers by the
  deadline, so every rank replies and a late one is `deadline_missed` naming its rank; the
  kernel rolls the others back in the same gap and poisons the host. `collective_rpc` gets the
  time left plus `FASTRSI_KERNEL_RPC_GRACE` (2 s) as a backstop timeout. A timeout leaves a late reply in the executor's queue, so the worker group then
  refuses every call (`deadline_missed`) and the kernel poisons the host: undo only, then
  respawn. The same for a call from outside the loop (prepare, status) whose job the loop
  started but did not finish while the caller waited: it may still apply, so the group
  refuses every later call and poisons the host itself.
- **Graph detection** (`graphs.py`, `side.Side`). On a worker whose config runs CUDA graphs
  or torch.compile, a target in model code (`vllm.model_executor.models` / `.layers`) is
  checked at prepare. Reached only through an op that splits the graph: an advisory
  `captured_partly` (Python on prefill, graphs on decode). Captured only, or undecided: a
  blocking `captured_target`, so every rank votes no and the edit is reported, not applied.
  Both route to CUDA graph recapture. With `"rebuild": ["recapture"]` in the plan they turn advisory (the host
  recaptures); without it the finding names the plan field, or restarting with the edit (or
  running eager). Accepted, the edit commits, and a `zero_witness` at proof is routed the same way. The split decorator
  itself (`eager_break_during_capture` and code nested in it, in any module) is a blocking
  `captured_target` too: it runs only when vLLM decorates its ops at import or captures its
  graphs (vLLM #59941 committed via live code swap with witness 0 on every rank). Calls made while the
  stream captures a graph count apart from live calls.
- **Selective CUDA graph recapture** (`recapture.py`, `host.RecaptureCoordinator`). A plan with
  `"rebuild": ["recapture"]` (or `"rebuild": ["R1"]`, optional `"recapture"` or `"r1"`:
  `{"modes": ["FULL" | "PIECEWISE"], "sizes": [num_tokens]}`) is recaptured on every rank right
  after its commit, inside the same gap. Per chosen mode, every graph wrapper of that mode gets a
  new, empty entry dict (the old one is kept) and the dispatcher's keys for the mode are emptied;
  the chosen descriptors are captured through the runner's own `_warmup_and_capture` (no Dynamo, no
  Inductor), under `witness.capture()`, and go back into the keys one by one. Descriptors not chosen
  run eager on the new code (or in a mode left alone) until `recapture_extend` captures them.
  All pieces of one descriptor are captured together (each piece replays at the addresses
  of the previous piece's output). Invariants: no step replays a graph of the old code next
  to Python running the new one; undo of the transaction puts back every old entry dict and
  key set (journal, no capture), so base graphs replay again and the new graphs are freed;
  a capture that fails on any rank is `rebuild_failed`, and the transaction is undone on
  every participant in the same gap (outcome rolled_back), with the phase-2 deadline: a
  participant that misses it poisons the host. `recapture_extend` runs as one gap job on every
  rank; if any rank fails, the ranks that captured drop that extend in the same gap
  (`recapture_extend_drop`), so every rank replays the same descriptors (a drop that fails
  poisons the host), and a job that outlives the gap timeout poisons the host. While the host
  is poisoned, `recapture_extend` is refused. The receipt's `rebuild` holds
  each rank's captured descriptors, seconds, eager leftovers and memory before/after, and
  `recapture_s` the gap's recapture time. Selective recapture runs only on vLLM's V1 `GPUModelRunner`
  (`VLLM_USE_V2_MODEL_RUNNER=0`); on another runner it is `rebuild_failed`, said plainly.
  A long recapture needs the op's `gap_timeout_s` above its duration.
- **Version log.** Each side records which code version every watched function runs at each
  step (`versions`), and any change between a step's two calls (`execute_model`,
  `sample_tokens`), so the ranks' logs show whether a step ran mixed versions.
- **Import order.** The host's import finder asks only the finders after it on
  `sys.meta_path` for `vllm.v1.engine.core` (an overlay first), so the core runs the same text
  the plan's base is, and a finder in front of it that asks every other finder does not
  recurse.

| File | What it owns |
|---|---|
| `host.py` | `Host` (coordinator, control channel, step timings), `WorkerGroup`, `parallel`, the `EngineCoreProc` patch, the import finder. |
| `worker_ext.py` | `Endpoint` (the participant protocol in one process: role filter, abstain, never raise) and `KernelWorkerExtension` (`--worker-extension-cls`; serves a `Side` and wraps the step calls). |
| `side.py` | `Side(Endpoint)`: background prepare, per-rank phase-2 lane and deadline, graph detection and routing, version log, a fault hook for tests. |
| `recapture.py` | Selective CUDA graph recapture on one rank: `Engine` (what recapture needs from a model runner; `vllm_engine` binds vLLM's V1 runner), `Recapture.run` / `extend`, `graphs`. |
| `graphs.py` | Where vLLM runs a model-code target, from the compilation config (cudagraph mode, compile mode, `splitting_ops`, breakable graphs' `eager_break_during_capture` ops) and the module source. |
| `client.py` | Sends one op over the file channel and returns the reply. |
| `conformance_lab.py` | `VllmLab` (`core/livelab/conformance` `LabUnderTest`): drives a live tensor-parallel `vllm serve` (launched via `services/engine.py start`) through the file channel across all 11 conformance cases (`c0`-`c10`). |

## Run it

The adapter on `PYTHONPATH` (`repos/vllm/livelab`; importing it puts the kernel's folder,
`core/livelab` or `FASTRSI_LIVELAB`, on `sys.path`), a sitecustomize that imports
`adapter.host` when `FASTRSI_KERNEL_DIR` is set, and:

```
FASTRSI_KERNEL_DIR=<dir> vllm serve <model> --distributed-executor-backend mp \
  --worker-extension-cls adapter.worker_ext.KernelWorkerExtension
python3 -m adapter.client <dir> '{"op": "txn", "plan": {...}}'
```

`--distributed-executor-backend mp` puts the worker in its own process even at TP 1, so
`collective_rpc` crosses a real process boundary.

## Conformance lab

`conformance_lab.py` exposes `VllmLab` (`LabUnderTest`) for `core/livelab/conformance`:

```
FASTRSI_VLLM_MODEL=<model-path> FASTRSI_VLLM_VENV=<venv> FASTRSI_VLLM_WORK=<scratch-dir> \
  PYTHONPATH=../../repos/vllm/livelab python3 -m conformance --lab adapter.conformance_lab:VllmLab
```

`VllmLab` launches `vllm serve` via `services/engine.py start` (`-tp FASTRSI_VLLM_TP`, default 1, `--distributed-executor-backend mp`, `KernelWorkerExtension`), runs greedy `/v1/completions` (`temperature=0`, `seed=0`) on fixed prompts, edits both worker (`vllm.v1.worker.gpu.sample.sampler`, `vllm.model_executor.layers.logits_processor`) and engine-core (`vllm.v1.core.sched.utils`, `vllm.v1.core.sched.scheduler`) modules, and runs `fresh_check` from a private `.py` copy farm on `port + 1`. Note: running `engine.py start` together with `adapter.host` requires the `_After` import-finder recursion guard (`#144` + `#148`).

## Golden tests

From the repo root (CPU, about 8 s; CI runs the same command):

```
python3 -m unittest discover -s repos/vllm/livelab/adapter -t repos/vllm/livelab -v
```

The transport is the only stand-in: `collective_rpc` becomes a loop over in-process endpoints,
each rank with its own copy of the toy engine.

| Test | Pain point |
|---|---|
| `test_two_ranks_switch_in_one_gap_and_undo_restores_the_same_code` | a rank that keeps the edit after undo, or a one-role edit vetoed by the other role's `empty_change` |
| `test_a_rank_that_raises_replies_and_the_others_roll_back` | a raising RPC method that misaligns every later reply |
| `test_a_timed_out_rpc_poisons_the_host_and_refuses_later_calls` | a late reply read by the next call |
| `test_a_loop_job_that_outlives_its_wait_poisons_the_host` | a prepare still staging in the loop after its caller counted it as a no vote, with the group still in use |
| `test_a_rank_that_raises_systemexit_still_replies` | a `SystemExit` escaping the RPC, so a rank sends no reply |
| `test_a_missing_rank_or_dp_group_refuses_the_transaction` | witness, receipt and undo that covered 1 of 4 DP groups with no word |
| `test_a_cell_for_one_role_commits_while_the_other_abstains` | a repair cell dropped, so no live run could use one |
| `test_status_without_a_txn_is_the_hosts_overall_status` | `status` without a txn raising KeyError |
| `test_a_gap_job_waits_for_the_batch_in_flight_and_nothing_new_is_scheduled` | a commit while a batch is in flight, finishing it on mixed code |
| `test_the_core_comes_from_the_overlay_and_is_patched` | a false `base_mismatch` when an overlay serves the engine core |
| `test_a_second_after_finder_in_front_does_not_recurse` | two finders that each ask every other finder, recursing until `RecursionError` at engine start |
| `test_ranks_that_finish_early_keep_their_vote` | ranks that answered "no prepare started" after their first finished poll: a false abort |
| `test_a_no_vote_waits_for_the_rank_still_preparing` | a no vote that ends the poll while another rank is still preparing: its vote lost, its staging racing the abort |
| `test_an_abort_while_staging_drops_what_the_prepare_staged` | an abort during a background prepare, after which the prepare stages the transaction and watches its keys anyway |
| `test_a_rank_that_hangs_in_commit_is_named_by_the_deadline` | one hung rank holding every step, or a late RPC reply misaligning the next call |
| `test_an_edit_inside_a_captured_graph_is_a_finding_and_not_applied` | a model-code edit that committed with no finding and never ran (graphs only) |
| `test_accepted_captured_target_proves_nothing_and_says_why` | an accepted graph edit whose zero witness gives no reason |
| `test_split_op_target_is_advisory_and_captured_only_stays_blocking` | a veto of code that does run in Python on prefill through a split op |
| `test_an_edit_of_the_split_decorator_is_refused_on_a_graph_engine` | an edit of the graph-split decorator (outside model code) that committed with no finding and a zero witness on every rank (vLLM #59941) |
| `test_one_bucket_is_recaptured_in_the_gap_and_undo_puts_the_base_graphs_back` | a committed edit whose graphs replay the old code (stale graph), or an undo that leaves the edit's graphs replaying |
| `test_extend_captures_a_left_out_bucket_later` | a bucket left eager forever after a selective R1 |
| `test_an_extend_that_fails_on_one_rank_is_dropped_on_every_rank` | one TP rank replaying a graph while another runs the same step eager after a failed extend |
| `test_a_drop_that_fails_poisons_the_host` | ranks left replaying different graphs after a drop that failed, with the host still taking transactions |
| `test_a_dropped_extend_gives_every_mode_its_left_out_buckets_back` | a dropped extend that gave back the left-out buckets of the last mode only |
| `test_undo_after_an_extend_with_nothing_to_capture` | an extend's undo entry raising `UnboundLocalError`, so undo of the transaction was incomplete |
| `test_an_extend_that_outlives_the_gap_poisons_the_host` | an extend still capturing in the loop after its caller gave up, with the host still taking transactions |
| `test_an_undo_after_a_failed_recapture_has_the_phase2_deadline` | the undo after a failed recapture holding the gap with no deadline, and a late rank never poisoning the host |
| `test_pieces_of_a_bucket_are_recaptured_together_and_other_modes_are_untouched` | recapturing one piece only, so the next piece reads another capture's output address |
| `test_a_capture_that_fails_on_one_rank_undoes_the_edit_on_every_rank` | ranks replaying different code after a failed capture, or a dispatcher key with no graph |
| `test_five_edit_recapture_undo_cycles_keep_the_graph_count` | graphs of every past edit kept alive (graph-pool leak) |
| `test_a_plan_asking_for_r1_turns_graph_findings_advisory` | a finding that told users to restart though the host can recapture |
| `test_the_extension_logs_which_version_each_step_runs` | a mid-step change logged that was only a key joining between a step's two calls |
| `test_an_aborted_transaction_leaves_nothing_watched` | ranks whose version logs differ after an aborted transaction |

## Not here yet

- R1 on vLLM's V2 model runner (`CudaGraphManager`), ubatched (DBO) and encoder graphs;
  recompiles (torch.compile) and R6; DP>1; pipeline parallel.
