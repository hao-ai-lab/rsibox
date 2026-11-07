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
  `txn`, `prove`, `undo`, `status`, `pstatus`, `fingerprint`, `at_base`, `ping`.
- **Phase 1** (prepare) runs in the host thread for the core and as one `collective_rpc` from
  the loop for the workers. **Phase 2** (validate, commit or abort, undo of a failed commit) is
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
  it off). `collective_rpc` gets the time left plus `FASTRSI_KERNEL_RPC_GRACE` (2 s) as its
  timeout. A timeout leaves a late reply in the executor's queue, so the worker group then
  refuses every call (`deadline_missed`) and the kernel poisons the host: undo only, then
  respawn. The same for a call from outside the loop (prepare, status) whose job the loop
  started but did not finish while the caller waited: it may still apply, so the group
  refuses every later call and poisons the host itself.
- **Import order.** The host's import finder asks the other finders on `sys.meta_path` for
  `vllm.v1.engine.core` (an overlay first), so the core runs the same text the plan's base is.

| File | What it owns |
|---|---|
| `host.py` | `Host` (coordinator, control channel, step timings), `WorkerGroup`, `parallel`, the `EngineCoreProc` patch, the import finder. |
| `worker_ext.py` | `Endpoint` (the participant protocol in one process: role filter, abstain, never raise) and `KernelWorkerExtension` (`--worker-extension-cls`). |
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

From the repo root (CPU, under a second; CI runs the same command):

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

## Not here yet

- Prepare runs inside one `collective_rpc`, so the loop stalls one step for as long as the
  ranks take to stage. Per-rank lanes, background prepare and graph detection come next.
- R1 and R6 (rebuilds after the commit) and their routes; DP>1; pipeline parallel.
