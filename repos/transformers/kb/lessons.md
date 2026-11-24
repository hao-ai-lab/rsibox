# Lessons Learned

1. **Post-commit and post-undo state resets prevent stale KV-cache masking.**
   Any cached forward/KV state keyed by input tokens will silently mask edits to model forward
   layers unless flushed on both commit and undo. Wiring `RecaptureCoordinator(...,
   post_commit_reset="state")` and `undo_resets=["state"]` ensures every edit and rollback sees
   fresh model evaluation without manual caller resets.

2. **Weight-tie-safe CPU pinning unlocks ~50 GiB/s snapshot resume without breaking tied embeddings.**
   Models such as `Qwen3-0.6B` tie `embed_tokens.weight` and `lm_head.weight` to the same
   `nn.Parameter` storage. Calling `model.to("cpu")` first and then deduplicating tensors via
   `dict.fromkeys([*model.parameters(), *model.buffers()])` while only pinning unaliased contiguous
   full-storage tensors preserves weight tying and achieves `0.032 s` resume on `Qwen3-0.6B` and
   `0.310 s` resume on `Qwen3-8B` (~51 GiB/s effective H2D bandwidth).

3. **First-time pinned CPU allocation dominates quiesce latency, not restore latency.**
   On `Qwen3-8B` (~15.4 GiB weights), the first `quiesce_for_snapshot()` took `15.64 s` due to cold
   `cudaHostAlloc` page pinning, dropping to `8.76–8.98 s` on repeats 2–3, whereas
   `resume_after_restore()` stayed consistently fast at `0.304–0.322 s`. For checkpoint-once /
   restore-many workflows (`ai-dynamo/snapshot`), paying pinning cost once at checkpoint time yields
   sub-second GPU weight restoration on every restored pod.

4. **Transformers 5.x `from_pretrained` prefers `dtype=` over `torch_dtype=`, while 4.x uses `torch_dtype=`.**
   Trying `dtype=` and falling back to `torch_dtype=` on `TypeError` keeps the worker compatible
   across both Transformers 4.x and 5.x releases without version branching.

5. **Anchors in large `transformers` modules must include surrounding context.**
   `transformers.generation.logits_process` contains 10 occurrences of `return scores`; qualifying
   the anchor with the enclosing `LogitsProcessorList.__call__` branch guarantees `edit_once()`
   uniqueness across `transformers` releases.

6. **Keep `torch` and `transformers` imports strictly inside worker/runtime functions.**
   Isolating heavy imports to `adapter.agent` functions lets `rsibox doctor` verify both
   `ToyTransformersLab` and `TransformersLab` on CPU-only hosts with zero GPU/runtime imports.
