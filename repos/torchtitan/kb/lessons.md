# Lessons Learned

1. **Snapshot and restore FSDP2 `DTensor` shards via `.to_local()` rather than full-tensor copies.**
   Parameters wrapped by `torch.distributed.fsdp.fully_shard` are `DTensor` instances whose
   sharded local storage is accessed via `param.to_local()`. Cloning `param.to_local().detach().clone()`
   at startup and copying back in-place under `torch.no_grad()` alongside `optimizer.state.clear()`
   yields `0.0` `max_abs_loss_diff` and `0.0` `max_abs_grad_norm_diff` rollback without breaking
   FSDP2 parameter hooks or triggering collective all-gathers during `reset("state")`.

2. **Deterministic CUDA GEMM flags are required for bit-exact `c3_fresh_equivalence`.**
   Disabling TF32 (`allow_tf32 = False`), enabling `torch.use_deterministic_algorithms(True,
   warn_only=True)`, and setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` in both warm worker and fresh
   `PackageOverlay` subprocess environments ensures live FSDP2 training steps match fresh restarts
   to 8 decimal places across 3-step chains.

3. **Keep `S0` reset automatic on commit, undo, and step probes.**
   Because every training step mutates weights and `AdamW` moment buffers, leaving state unreset
   after a probe or refused install causes subsequent base checks to diverge. Resetting to `S0` at
   the start of non-continuation probes (and via `post_commit_reset="state"` /
   `undo_resets=["state"]`) guarantees repeatable comparisons while `continue_state=True` remains
   available when explicitly testing multi-step weight progression.

4. **Scale initial weights by `1 / sqrt(fan_in)` for numerically well-conditioned FSDP2 tests.**
   Unscaled random initialization across 4 residual MLP blocks compounds activation variance into
   `1e10+` losses where floating-point rounding masks small optimizer edits. Xavier-scaled
   initialization (`weight_scale / sqrt(fan_in)`) keeps base losses near `~1.0` so both loss and
   gradient-clip edits produce clean, sensitive, reproducible shifts.

5. **Keep `torch` imports strictly lazy inside worker functions and test methods.**
   Importing `torch` at module top level (including inside test-module skip decorators) violates
   `rsibox doctor`'s zero-GPU-import check and slows CPU CI. Deferring `import torch` into runtime
   functions keeps `rsibox list`, `rsibox doctor`, and `rsibox conformance torchtitan --real -- --help`
   instant and CPU-only.
