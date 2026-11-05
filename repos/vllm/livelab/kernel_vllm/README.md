# kernel_vllm: the transaction kernel inside vLLM (prototype)

The engine side of `template/livelab/kernel`: one transaction swaps code on the workers and in
the engine core together, between the same two engine steps, with receipts. Prototype: TP 1
only so far.

| File | What it does |
|---|---|
| `host.py` | The engine-core host. Patches `EngineCoreProc` when imported with `FASTRSI_KERNEL_DIR` set: custom input-queue items run jobs on the busy-loop thread; a phase-2 job runs only when no batch is in flight (pause keep: `scheduler.has_requests` is shadowed to False while `batch_queue` drains). Runs the kernel's `StepGapCoordinator` over two participants (the workers, the engine core) and serves a file control channel. Records every step's duration. |
| `worker_ext.py` | `KernelWorkerExtension` (`--worker-extension-cls`): the participant protocol on each worker rank, reached by `collective_rpc("fastrsi_kernel", args=(method, payload))` issued from the busy loop. |
| `side.py` | One participant side (worker or core): a kernel `LocalParticipant`, prepare in a background thread (`prepare_start` / `prepare_poll`), per-role plan filtering (`"on": ["worker"]`), a fault hook for tests, and replies that never raise (collective_rpc reply desync, kb/lessons.md). |
| `client.py` | Sends one op over the file channel (`ctl.json` -> `reply/<seq>.json`): `txn`, `prove`, `undo`, `status`, `ping`. Replies carry the kernel's receipt records; receipts also go to `receipts.jsonl` next to the channel. |
| `live_test.py` | The live golden tests below, against a running engine. |

## How a transaction runs

```
client ── ctl.json ──► host thread (engine-core process)
  phase 1, engine keeps stepping:
    workers: collective_rpc(prepare_start) from the loop; each rank stages in a thread;
             collective_rpc(prepare_poll) every 20 ms until done
    core:    prepare in the host thread
  phase 2, one job on the busy loop, after in-flight batches land (pause keep):
    validate (workers + core) -> commit (workers + core), or abort; undo of a failed commit
```

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

## Findings from the live run

- `apply_all_penalties` (the dashboard demo's target) never runs on vLLM 0.33: the default V2
  model runner applies penalties in `vllm.v1.worker.gpu.sample.penalties.PenaltiesState.apply`.
  The first run swapped `apply_all_penalties`; the kernel reported `zero_witness` on the
  workers, and "Paris" was still generated. The swap was never counted as a pass.
- Phase 1 is not free on the engine: a heap scan in the engine-core or worker process slows
  that process's steps while it runs (one 221 ms step here). A locate that does not scan the
  whole heap is the next step.
- TP > 1 is not verified. With several ranks, the same code path sends every call with one
  collective_rpc from the busy loop, and the replies are merged (`ranks_agree` for hashes);
  this needs a run on a multi-GPU node.
- The receipt writer (`kernel/receipt.py`) is still a proposal that needs a human owner
  (AGENTS.md, Trust).
