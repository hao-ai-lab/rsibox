# Walkthrough: three one-line edits on a live `vllm serve` (Qwen3-8B, 1 GB200)

Three small edits to vLLM 0.31.0, each checked two ways on the same GPU: by restarting
`vllm serve` with the edit, and by installing it into a running server through the kernel adapter
([`../../livelab/adapter`](../../livelab/adapter/README.md)). From our 2026-11-10 run on one
GB200 (aarch64, 4 GPUs in the node, one used), Qwen3-8B, TP 1, CUDA graphs on (vLLM defaults),
warm on-disk caches, weights on a memory disk.

| Edit | Restart, edit to result | Live, edit to result | Edited code ran (witness) | Live output = fresh process |
|---|---:|---:|---|---|
| scheduler: cap the step's token budget | 62.98 s | 1.175 s | yes, 816 calls in the engine core | yes (same as base) |
| sampler: pick the runner-up token | 63.37 s | 1.072 s | yes, 768 calls on the worker | yes (differs from base) |
| old sampler module: the same idea | 63.08 s | 1.067 s | **no, 0 calls** (`zero_witness`) | yes (same as base) |

Of the live time, 1.03 s is the check itself (8 requests); the install is 0.03 to 0.14 s and the
engine loop stalls for about 1 ms.

Everything below is the output of one command, [`walkthrough.py`](walkthrough.py) (standard
library only), run with the Python of a vLLM 0.31.0 environment:

```
$ CUDA_VISIBLE_DEVICES=0 python repos/vllm/examples/live-edit-qwen3/walkthrough.py \
    --model <Qwen3-8B directory or HF id> --work <scratch dir>
```

It copies the installed `vllm` package to `<work>/src` and serves from the copy, so the installed
package is never edited. JSON lines are wrapped here for reading; nothing else is changed.

The current CLI compares the full completion text across repeats and fresh and live engines.
It verifies the edited method on each owning rank, the inactive sampler’s zero witness,
and the restored code identity and text after undo. With `--out FILE`, the JSON also
includes client-observed time to first token and elapsed time for each streamed request.
The output below comes from the dated run above.

## 1. The edits

```diff
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -582 +582 @@
-        token_budget = self.max_num_scheduled_tokens
+        token_budget = min(self.max_num_scheduled_tokens, 4096)

--- a/vllm/v1/worker/gpu/sample/sampler.py
+++ b/vllm/v1/worker/gpu/sample/sampler.py
@@ -342,0 +343 @@
+            processed_logits = processed_logits.scatter(-1, processed_logits.argmax(-1, keepdim=True), float("-inf"))

--- a/vllm/v1/sample/sampler.py
+++ b/vllm/v1/sample/sampler.py
@@ -242 +242 @@
-        return logits.argmax(dim=-1).view(-1)
+        return logits.topk(2, dim=-1).indices[:, 1].view(-1)
```

- **scheduler** runs in the engine-core process. It should not change this workload's output.
- **sampler** runs in the worker process. It masks the top token, so greedy decoding picks the
  runner-up: a test edit, chosen because its effect is impossible to miss.
- **old sampler** does the same in `vllm/v1/sample/sampler.py`, the module a reader finds first.

## 2. The workload, and the restart it costs today

The check is 8 fixed prompts, one request at a time, greedy, 32 tokens each; `output` is a hash
of the 8 texts. The server command for every start:

```
python -m vllm.entrypoints.cli.main serve <model> --served-model-name model --port 8000 \
  --distributed-executor-backend mp --no-enable-prefix-caching
```

A first start warms the caches and gives the base output, three times the same:

```
{"step": "warm", "healthy_after_s": 64.55,
 "check": {"seconds": [1.043, 1.033, 1.035],
  "outputs": ["db291a3a797916ec", "db291a3a797916ec", "db291a3a797916ec"],
  "first_text": " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the Netherlands is Amsterdam. The"},
 "stop_s": 0.41}
```

Then the way a developer checks an edit without the lab: write it into the package, start a new
server, run the check. Once with no edit, then once per edit:

```
{"step": "restart_check", "edit": null, "healthy_after_s": 63.54, "check_s": 1.042,
 "edit_to_result_s": 64.58, "output": "db291a3a797916ec", "second_check_output": "db291a3a797916ec",
 "first_text": " Paris. The capital of Italy is Rome. ...", "stop_s": 0.41}
{"step": "restart_check", "edit": "scheduler", "healthy_after_s": 61.94, "check_s": 1.042,
 "edit_to_result_s": 62.98, "output": "db291a3a797916ec", "second_check_output": "db291a3a797916ec",
 "first_text": " Paris. The capital of Italy is Rome. ...", "stop_s": 0.41}
{"step": "restart_check", "edit": "sampler", "healthy_after_s": 62.32, "check_s": 1.045,
 "edit_to_result_s": 63.37, "output": "e8aa4a6b469cd55b", "second_check_output": "e8aa4a6b469cd55b",
 "first_text": " a country. True, false or can’t tell?\n\nFalse. \n\nStep 2 Explanation:\nStep: 2 of the reasoning is the evaluation. \n\nStep",
 "stop_s": 0.41}
{"step": "restart_check", "edit": "old_sampler", "healthy_after_s": 62.03, "check_s": 1.042,
 "edit_to_result_s": 63.08, "output": "db291a3a797916ec", "second_check_output": "db291a3a797916ec",
 "first_text": " Paris. The capital of Italy is Rome. ...", "stop_s": 0.41}
```

(`first_text` is cut here where it repeats the base text above.) Each check costs 62 to 65 s, of
which the check is 1 s. The old-sampler edit gives the base output even in a fresh process: with
only this evidence it reads as "the edit is harmless".

## 3. One server with the adapter, and `install`

The same command, plus the adapter: `RSIBOX_KERNEL_DIR=<work>/ctl` in the environment, a
`sitecustomize.py` that imports `adapter.host`, and
`--worker-extension-cls adapter.worker_ext.KernelWorkerExtension`.

```
{"step": "live_start", "healthy_after_s": 63.03,
 "ping": {"world": 1, "dp": 1, "dp_rank": 0, "expected": [0], "note": null}}
{"step": "live_base_check", "check": {"seconds": [1.042, 1.032, 1.032],
  "outputs": ["db291a3a797916ec", "db291a3a797916ec", "db291a3a797916ec"],
  "first_text": " Paris. The capital of Italy is Rome. ..."}}
```

The adapter does not change the base output or the check time. `install` is one transaction
(`adapter.client`, op `txn`): the plan carries the edited module text, the hash of the base text,
and the role that runs it. The files the server was started from stay at base.

```
{"step": "install", "edit": "scheduler", "ok": true, "outcome": "committed", "txn": "txn-f8f23b2c",
 "mechanisms": ["R0 swap"], "targets": {"vllm.v1.core.sched.scheduler": "auto"},
 "prepare_s": 0.133509, "gap_s": 0.001242, "edit_to_commit_s": 0.140634,
 "client_round_trip_s": 0.1421, "findings": [], "error": null}
```

Staging the 3,359-line scheduler module takes 0.13 s while the engine keeps stepping; the commit
holds the engine loop for 1.2 ms (`gap_s`).

## 4. `check`, with its witness

Three checks, then op `prove`, which reads the call counter the swap put on each edited function:

```
{"step": "check", "edit": "scheduler", "edit_to_result_s": 1.175,
 "seconds": [1.033, 1.032, 1.035],
 "outputs": ["db291a3a797916ec", "db291a3a797916ec", "db291a3a797916ec"],
 "first_text": " Paris. The capital of Italy is Rome. ...",
 "proven": true,
 "witness": {"workers/rank0": {}, "core": {"vllm.v1.core.sched.scheduler:Scheduler.schedule:562": 816}},
 "edit_to_run_s": null, "findings": []}
```

The output equals base, as expected for this edit, and the witness shows why that is still a
result: the new `Scheduler.schedule` ran 816 times in the engine core during the three checks.
The proof record has the first call 0.143 s after the edit (`first_run_s`, core). `edit_to_run_s`
is null because the kernel reports it only when every participant shows a call, and the worker
has no part in this edit.

## 5. `uninstall`

Op `undo`, then two checks, op `at_base`, and the code fingerprint (content hash and code object
id of every function in the three modules, on the worker and the core) against the one taken
before the first install:

```
{"step": "uninstall", "edit": "scheduler", "ok": true, "status": "undone", "undo_s": 0.000689,
 "client_round_trip_s": 0.0102,
 "check": {"seconds": [1.032, 1.031], "outputs": ["db291a3a797916ec", "db291a3a797916ec"],
  "first_text": " Paris. The capital of Italy is Rome. ..."},
 "output_is_base": true, "at_base": true, "fingerprint_is_base": true}
```

## 6. The sampler edit: visible, and equal to a fresh process

```
{"step": "install", "edit": "sampler", "ok": true, "outcome": "committed", "txn": "txn-bb34411d",
 "mechanisms": ["R0 swap"], "targets": {"vllm.v1.worker.gpu.sample.sampler": "auto"},
 "prepare_s": 0.02097, "gap_s": 0.000532, "edit_to_commit_s": 0.029886,
 "client_round_trip_s": 0.0356, "findings": [], "error": null}
{"step": "check", "edit": "sampler", "edit_to_result_s": 1.072,
 "seconds": [1.037, 1.036, 1.038],
 "outputs": ["e8aa4a6b469cd55b", "e8aa4a6b469cd55b", "e8aa4a6b469cd55b"],
 "first_text": " a country. True, false or can’t tell?\n\nFalse. \n\nStep 2 Explanation:\nStep: 2 of the reasoning is the evaluation. \n\nStep",
 "proven": true,
 "witness": {"workers/rank0": {"vllm.v1.worker.gpu.sample.sampler:Sampler._sample_random:324": 768}, "core": {}},
 "edit_to_run_s": null, "findings": []}
{"step": "uninstall", "edit": "sampler", "ok": true, "status": "undone", "undo_s": 0.000326,
 "client_round_trip_s": 0.0052,
 "check": {"seconds": [1.032, 1.031], "outputs": ["db291a3a797916ec", "db291a3a797916ec"],
  "first_text": " Paris. The capital of Italy is Rome. ..."},
 "output_is_base": true, "at_base": true, "fingerprint_is_base": true}
```

The live output `e8aa4a6b469cd55b` is the output of the fresh server started with this edit in
step 2. 768 calls = 3 checks × 8 requests × 32 tokens. After `uninstall` the output is base again.

## 7. The old-sampler edit: installed, not proven

```
{"step": "install", "edit": "old_sampler", "ok": true, "outcome": "committed", "txn": "txn-abd4f492",
 "mechanisms": ["R0 swap"], "targets": {"vllm.v1.sample.sampler": "auto"},
 "prepare_s": 0.02065, "gap_s": 0.000527, "edit_to_commit_s": 0.030163,
 "client_round_trip_s": 0.0356, "findings": [], "error": null}
{"step": "check", "edit": "old_sampler", "edit_to_result_s": 1.067,
 "seconds": [1.032, 1.031, 1.032],
 "outputs": ["db291a3a797916ec", "db291a3a797916ec", "db291a3a797916ec"],
 "first_text": " Paris. The capital of Italy is Rome. ...",
 "proven": false,
 "witness": {"workers/rank0": {"vllm.v1.sample.sampler:Sampler.greedy_sample:240": 0}, "core": {}},
 "edit_to_run_s": null, "findings": ["zero_witness"]}
{"step": "uninstall", "edit": "old_sampler", "ok": true, "status": "undone", "undo_s": 0.000332,
 "client_round_trip_s": 0.0102,
 "check": {"seconds": [1.034, 1.031], "outputs": ["db291a3a797916ec", "db291a3a797916ec"],
  "first_text": " Paris. The capital of Italy is Rome. ..."},
 "output_is_base": true, "at_base": true, "fingerprint_is_base": true}
```

The swap committed, the output did not move, and the witness says why: `Sampler.greedy_sample`
was called 0 times. vLLM 0.31.0 samples in `vllm/v1/worker/gpu/sample/sampler.py` by default (the
V2 model runner); the module edited here is not on that path. The restart in step 2 gave the same
unchanged output after 63 s, with nothing to tell "harmless" from "never ran".

## 8. Agreement

```
{"step": "agreement", "base": "db291a3a797916ec",
 "fresh": {"base": "db291a3a797916ec", "scheduler": "db291a3a797916ec",
           "sampler": "e8aa4a6b469cd55b", "old_sampler": "db291a3a797916ec"},
 "live": {"scheduler": "db291a3a797916ec", "sampler": "e8aa4a6b469cd55b",
          "old_sampler": "db291a3a797916ec"},
 "fresh_base_is_base": true,
 "live_equals_fresh": {"scheduler": true, "sampler": true, "old_sampler": true}}
```

## Found on the way

The first version of the check sent the 8 prompts in one request. That output is not repeatable:
over 7 server starts the first check gave three different hashes, later checks in the same
process gave a fourth, and the first check after a live scheduler install a fifth. The texts
differed in 3 of the 8 prompts. Turning the prefix cache off did not fix it; sending one request
at a time did (24 base-output checks in 5 processes above, one hash). A later start with the
prefix cache on and one request at a time gave the same hash three times, so
`--no-enable-prefix-caching` is part of the run above but not what made it repeatable.
Recorded in [`../../kb/lessons.md`](../../kb/lessons.md).

## What this does not show

- **Conformance.** `core/livelab/conformance` drives a lab through the `LabUnderTest` class
  (`install`, `check`, `reset`, `uninstall`, `status` and a test harness). vLLM has no such class
  yet: this walkthrough calls the adapter's ops (`txn`, `prove`, `undo`, `at_base`,
  `fingerprint`) directly. So 0 of the 11 cases were run on vLLM, and by this repo's own rule
  (AGENTS.md, Trust) the kernel records of this run do not count as receipts yet.
- One run, one node. Each restart was measured once per edit (4 starts, 62.98 to 64.58 s).
- No `reset`, no fault (failed install, crash, hang), no chained edits.
- TP 1 only: one worker rank and the engine core. No edit inside a CUDA graph, no Triton kernel.
- The restart here is 63 s, not the 228 s of the ladder in [`../../kb/costs.md`](../../kb/costs.md);
  that is a different measurement, which this run does not reproduce.
