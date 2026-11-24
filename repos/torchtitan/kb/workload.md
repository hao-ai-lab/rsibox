# Workload F(x)

## Target Workload (Real GPU FSDP2, `world_size=1`)

Deterministic multi-step PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`) training loop running
`forward -> loss -> backward -> clip_grad_norm_ -> AdamW.step()` over a self-contained synthetic
4-layer SwiGLU-style `TitanMLPNet` (`hidden=256`, `ffn=512`, `layers=4`, `batch_size=8`, `steps=3`,
`seed=0`, `float32`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled) with an `nccl` process group
(upstream `pytorch/torchtitan` model/config integration is not yet wired):

```json
{
  "steps": 3
}
```

Measured outputs on 1× NVIDIA B200 (PyTorch 2.14.0+cu130, `batch_size=8`, `steps=3`, identical
across both CLI conformance and the 3× benchmark):
- Base `losses`: `[1.08151245, 0.21941125, 8.59863853]`
- Edited `compute_loss` (`* 2.0`) `losses`: `[2.1630249, 0.43885228, 16.2074585]`

## CPU Multi-Rank Conformance Workload (`ToyTorchTitanLab`)

Pure-Python 2-rank fake-distributed training loop (`world_size=2`, `hidden=8`, `ffn=16`,
`layers=1`, `steps=2`, `seed=0`) executing `adapter.model_ops` (`mlp_forward`, `compute_loss`)
and `adapter.optim_ops` (`clip_and_scale_grad`, `optimizer_step_scale`) across both worker
processes with zero external dependencies.

## Correctness Gate

1. **Repeatability (`c0`):** Repeated `check()` calls with `reset("state")` in between produce
   bit-exact `losses`, `grad_norms`, and per-rank weight checksum `digest` — including for
   stochastic ops using `random` via post-init saved RNG restoration.
2. **Per-rank execution proof (`c1`, `c2`, `c10`):** Every live edit reports `proven=True` with
   witness count `>= steps` on every rank when executed, and `proven=False` with `zero_witness`
   when a startup-only function (`init_weight_scale`) is edited.
3. **Fresh parity (`c3`):** Live output after visible and 3-edit chain swaps equals a fresh
   `PackageOverlay` subprocess started from the edited files.
4. **Bit-exact rollback (`c4`, `c7`, `c9`):** `uninstall()` restores base code identity and
   restores `S0` weights + `AdamW` state + RNG with `max_abs_loss_diff == 0.0` and
   `max_abs_grad_norm_diff == 0.0`.
5. **Fail-closed safety (`c5`, `c6`, `c8`, golden tests):** Partial-apply/commit faults roll back
   atomically across ranks; syntax, closure, module-state, empty, `__init__` constructor edits,
   and unsupported `mode="fsdp2"` with `world_size > 1` are refused before mutating state; worker
   crash/hang/lost/tamper faults poison the session until `reset("respawn")`.
