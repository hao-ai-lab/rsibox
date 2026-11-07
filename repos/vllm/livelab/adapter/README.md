# Adapter: the transaction kernel inside a live vLLM engine

The engine side of `../kernel`: one transaction swaps code on every worker rank and in the
engine core together, between the same two engine steps, with receipts. This folder holds the
T0 path (R0, a function-body swap). Rebuilds after a commit (R1 recapture or recompile, R6
weight reload) are not here: an edit that needs one is a finding, not a silent swap.

## Contract

- **Host** (`host.py`). Imported in the engine-core process with `FASTRSI_KERNEL_DIR=<dir>`, it
  patches `EngineCoreProc` so custom input-queue items run jobs on the busy-loop thread, and
  runs the kernel's `StepGapCoordinator` over two participants: the workers (`WorkerGroup`,
  every rank behind one `collective_rpc`) and the engine core (an `Endpoint` in its own
  process). It serves a file control channel (`ctl.json` -> `reply/<seq>.json`) with the ops
  `txn`, `prove`, `undo`, `status`, `pstatus`, `fingerprint`, `versions`, `at_base`, `ping`.
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
  Both route R1 (recapture with the commit), which this adapter cannot run yet; the finding
  says so and names T1 (restart with the edit, or run eager) as what works today. Accepted,
  the edit commits, and a `zero_witness` at proof is routed the same way. Calls made while the
  stream captures a graph count apart from live calls.
- **Version log.** Each side records which code version every watched function runs at each
  step (`versions`), and any change between a step's two calls (`execute_model`,
  `sample_tokens`), so the ranks' logs show whether a step ran mixed versions.
- **Import order.** The host's import finder asks the other finders on `sys.meta_path` for
  `vllm.v1.engine.core` (an overlay first), so the core runs the same text the plan's base is.

| File | What it owns |
|---|---|
| `host.py` | `Host` (coordinator, control channel, step timings), `WorkerGroup`, `parallel`, the `EngineCoreProc` patch, the import finder. |
| `worker_ext.py` | `Endpoint` (the participant protocol in one process: role filter, abstain, never raise) and `KernelWorkerExtension` (`--worker-extension-cls`; serves a `Side` and wraps the step calls). |
| `side.py` | `Side(Endpoint)`: background prepare, per-rank phase-2 lane and deadline, graph detection and routing, version log, a fault hook for tests. |
| `graphs.py` | Where vLLM runs a model-code target, from the compilation config (cudagraph mode, compile mode, `splitting_ops`, breakable graphs' `eager_break_during_capture` ops) and the module source. |
| `client.py` | Sends one op over the file channel and returns the reply. |

## Run it

The adapter and `kernel` on `PYTHONPATH` (both live in `repos/vllm/livelab`), a sitecustomize
that imports `adapter.host` when `FASTRSI_KERNEL_DIR` is set, and:

```
FASTRSI_KERNEL_DIR=<dir> vllm serve <model> --distributed-executor-backend mp \
  --worker-extension-cls adapter.worker_ext.KernelWorkerExtension
python3 -m adapter.client <dir> '{"op": "txn", "plan": {...}}'
```

`--distributed-executor-backend mp` puts the worker in its own process even at TP 1, so
`collective_rpc` crosses a real process boundary.

## Golden tests

From the repo root (CPU, about 2 s; CI runs the same command):

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
| `test_ranks_that_finish_early_keep_their_vote` | ranks that answered "no prepare started" after their first finished poll: a false abort |
| `test_a_no_vote_waits_for_the_rank_still_preparing` | a no vote that ends the poll while another rank is still preparing: its vote lost, its staging racing the abort |
| `test_an_abort_while_staging_drops_what_the_prepare_staged` | an abort during a background prepare, after which the prepare stages the transaction and watches its keys anyway |
| `test_a_rank_that_hangs_in_commit_is_named_by_the_deadline` | one hung rank holding every step, or a late RPC reply misaligning the next call |
| `test_an_edit_inside_a_captured_graph_is_a_finding_and_not_applied` | a model-code edit that committed with no finding and never ran (graphs only) |
| `test_accepted_captured_target_proves_nothing_and_says_why` | an accepted graph edit whose zero witness gives no reason |
| `test_split_op_target_is_advisory_and_captured_only_stays_blocking` | a veto of code that does run in Python on prefill through a split op |
| `test_the_extension_logs_which_version_each_step_runs` | a mid-step change logged that was only a key joining between a step's two calls |
| `test_an_aborted_transaction_leaves_nothing_watched` | ranks whose version logs differ after an aborted transaction |

## Not here yet

- R1 and R6 (rebuilds after the commit) and their routes; DP>1; pipeline parallel.
