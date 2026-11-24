# Workload F(x)

## Target Workload

Deterministic greedy batched causal-LM generation (`do_sample=False`, `temperature=0.0`,
`use_cache=True`, `seed=0`, left-padded batch of 2 prompts, `max_new_tokens=16`, `bfloat16`):

```json
{
  "prompts": [
    "Write one sentence about lighthouses.",
    "Write one sentence about rivers."
  ],
  "max_new_tokens": 16,
  "temperature": 0.0
}
```

Models verified on 1× NVIDIA B200 (PyTorch 2.14.0+cu130, Transformers 5.20.2):

- **Qwen3-0.6B** (`bfloat16`, ~1.2 GiB weights): base output digest `180bc838b63802ce`
- **Qwen3-8B** (`bfloat16`, ~15.4 GiB weights): base output digest `2301fe01c380e740`

## CPU Toy Conformance Workload (`ToyTransformersLab`)

Pure-Python `toy_hf` fixture (`vocab_size=64`, `seed=0`, 3 prompts, `max_new_tokens=6`,
`temperature=0.0`), exercising `logits_process`, `generation`, and `modeling` modules across all
11 RSIBox conformance cases (`c0–c10`).

## Correctness Gate

1. **Repeatability (`c0`):** Repeated base checks and post-`reset("state")` checks produce bit-exact
   generated token IDs and identical SHA-256 digests.
2. **Execution witness (`c1`, `c2`, `c10`):** Every live edit reports `proven=True` with non-zero
   witness counts when reached (`1` call for `GenerationMixin._sample`, `16` calls for
   `LogitsProcessorList.__call__`, `1,808` / `2,320` calls for `Qwen3RMSNorm.forward`), and
   `proven=False` with `zero_witness` when unreached.
3. **Fresh parity (`c3`):** Live output after an edit equals an isolated `PackageOverlay` fresh
   subprocess started from the edited files.
4. **Exact rollback (`c4`):** `uninstall()` restores the exact base code fingerprint and bit-exact
   base token IDs.
5. **Snapshot continuity:** After `DynamoK8sSnapshotDriver.checkpoint_and_wait(lab.quiesce_hooks())`,
   generation remains bit-exact against base and subsequent RSIBox live edits commit, prove, and
   roll back cleanly.
