# Walkthrough: Live Step-Gap Hot-Swap on PyTorch FSDP2 (`TorchTitanLab`)

This runnable walkthrough demonstrates a live step-gap edit to `adapter.model_ops:compute_loss`
inside a warm PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard` + `nccl`, `world_size=1`) worker
running the self-contained `TitanMLPNet` training loop, with per-rank execution witness proof,
fresh-subprocess parity, and bit-exact `S0` rollback.

## 1. Run CPU 2-Rank Conformance (`ToyTorchTitanLab`, No GPU Required)

```bash
PYTHONPATH=core/livelab python3 -m rsibox conformance torchtitan
```

Observed output (2026-11-24 CPU run, 2 worker ranks):

```text
c0_repeatable         pass  0.044  3 checks (after resets ['weights_s0', 'optimizer_state', 'rng', 'step_counter'])
c1_noop               pass  0.065  installed by ['R0 swap']; calls {'adapter.model_ops:compute_loss:35': [2, 2]}
c2_visible            pass  0.066  calls per participant {'adapter.model_ops:compute_loss:35': [2, 2]}
c3_fresh_equivalence  pass  0.304  visible & chain of 3 bit-exact vs fresh subprocess
c4_uninstall          pass  0.063  uninstall in 0.0037 s; base identity restored
c5_failed_install     pass  0.103  fail_apply & fail_commit rolled back; 3-target/2-module control committed
c6_refusal            pass  0.080  syntax, closure, module_state, empty refused cleanly
c7_chain              pass  0.116  3 stacked edits undone one by one to base
c8_crash_and_hang     pass  2.957  crash/hang/lost/tamper poisoned -> respawn 0.097-0.320 s
c9_soak               pass  0.549  20 cycles; install median 0.0142 s, uninstall median 0.0045 s
c10_not_proven        pass  0.117  proven False, findings ['zero_witness', 'zero_witness']; fresh diverges

11/11 as expected
```

## 2. Run Real GPU FSDP2 Conformance (`TorchTitanLab` on 1× NVIDIA B200)

```bash
PYTHONPATH=core/livelab python3 -m rsibox conformance torchtitan --real -- \
  --mode fsdp2 --world-size 1 --hidden 256 --ffn 512 --layers 4 --steps 3 --check-timeout 15.0
```

Observed output (2026-11-24 1× NVIDIA B200, PyTorch 2.14.0+cu130, FSDP2 `fully_shard` + `nccl`,
default `--batch-size 8`):

```text
c0_repeatable         pass   1.051  3 checks identical after ['weights_s0', 'optimizer_state', 'rng', 'cuda_cache', 'step_counter']
c1_noop               pass   1.102  installed by ['R0 swap']; calls {'adapter.model_ops:compute_loss:35': [3]}
c2_visible            pass   1.175  losses [1.08151245, 0.21941125, 8.59863853] -> [2.1630249, 0.43885228, 16.2074585]
c3_fresh_equivalence  pass  11.573  visible & chain of 3 live == fresh FSDP2 subprocess
c4_uninstall          pass   1.196  uninstall in 0.0442 s; base identity restored
c5_failed_install     pass   1.447  fail_apply & fail_commit rolled back; 3 targets in 2 modules committed
c6_refusal            pass   1.241  patch_invalid, closure_changed, outside_changes, empty_change refused
c7_chain              pass   1.618  3 stacked edits undone step-by-step to base
c8_crash_and_hang     pass  28.242  crash/hang/lost/tamper poisoned -> respawn 2.58-2.80 s
c9_soak               pass   3.887  20 cycles; install median 0.0541 s, uninstall median 0.0458 s
c10_not_proven        pass   4.595  proven False (zero_witness) on startup-only init_weight_scale; fresh diverges

11/11 as expected
```

## 3. Interactive Python Snippet (Live Swap + Bit-Exact Rollback)

```python
import sys
sys.path.insert(0, "repos/torchtitan/livelab")
import adapter  # bootstraps core/livelab
from adapter.lab import TorchTitanLab

lab = TorchTitanLab({"mode": "fsdp2", "world_size": 1, "hidden": 256, "ffn": 512, "layers": 4, "batch_size": 8, "steps": 3, "seed": 0})
try:
    base = lab.check()
    res = lab.install(lab.patches()["visible"])        # 0.059 s install vs 2.689 s cold boot (32.5x)
    edited = lab.check()                               # 0.024 s check, proven=True, witness=3
    lab.uninstall()                                    # 0.048 s undo + S0 restore
    restored = lab.check()
    max_diff = max(abs(a - b) for a, b in zip(base["output"]["losses"], restored["output"]["losses"]))
    assert max_diff == 0.0
finally:
    lab.close()
```

Observed values on 1× NVIDIA B200 (`batch_size=8`, `steps=3`):
- `base["output"]["losses"]`: `[1.08151245, 0.21941125, 8.59863853]`
- `edited["output"]["losses"]`: `[2.1630249, 0.43885228, 16.2074585]` (`witness={'rank0': {'adapter.model_ops:compute_loss:35': 3}}`)
- `max_diff`: `0.0` (bit-exact `S0` weight, `AdamW`, and RNG rollback)
