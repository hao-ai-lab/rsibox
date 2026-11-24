# Walkthrough: Live Step-Gap Hot-Swap on PyTorch FSDP2 (`TorchTitanLab`)

This runnable walkthrough demonstrates a live step-gap edit to `adapter.model_ops:compute_loss`
inside a warm PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard` + `nccl`) worker, with per-rank
execution witness proof, fresh-subprocess parity, and bit-exact `S0` rollback.

## 1. Run CPU 2-Rank Conformance (`ToyTorchTitanLab`, No GPU Required)

```bash
PYTHONPATH=core/livelab python3 -m rsibox conformance torchtitan
```

Observed output (2026-11-24 CPU run, 2 worker ranks):

```text
c0_repeatable         pass  0.205  3 checks (after resets ['weights_s0', 'optimizer_state', 'rng', 'step_counter'])
c1_noop               pass  0.208  installed by ['R0 swap']; calls {'adapter.model_ops:compute_loss:41': [2, 2]}
c2_visible            pass  0.208  calls per participant {'adapter.model_ops:compute_loss:41': [2, 2]}
c3_fresh_equivalence  pass  0.457  visible & chain of 3 bit-exact vs fresh subprocess
c4_uninstall          pass  0.213  uninstall in 0.0046 s; base identity restored
c5_failed_install     pass  0.258  fail_apply & fail_commit rolled back; 3-target/2-module control committed
c6_refusal            pass  0.236  syntax, closure, module_state, empty refused cleanly
c7_chain              pass  0.268  3 stacked edits undone one by one to base
c8_crash_and_hang     pass  3.172  crash/hang/lost/tamper poisoned -> respawn 0.177-0.319 s
c9_soak               pass  0.806  20 cycles; install median 0.0155 s, uninstall median 0.0052 s
c10_not_proven        pass  0.296  proven False, findings ['zero_witness', 'zero_witness']; fresh diverges

11/11 as expected
```

## 2. Run Real GPU FSDP2 Conformance (`TorchTitanLab` on 1× NVIDIA B200)

```bash
PYTHONPATH=core/livelab python3 -m rsibox conformance torchtitan --real -- \
  --mode fsdp2 --world-size 1 --hidden 256 --ffn 512 --layers 4 --steps 3 --check-timeout 15.0
```

Observed output (2026-11-24 1× NVIDIA B200, PyTorch 2.14.0+cu130, FSDP2 `fully_shard` + `nccl`):

```text
c0_repeatable         pass   0.996  3 checks identical after ['weights_s0', 'optimizer_state', 'cuda_cache', 'step_counter']
c1_noop               pass   0.960  installed by ['R0 swap']; calls {'adapter.model_ops:compute_loss:37': [3]}
c2_visible            pass   0.936  losses [1.02023852, 0.33246392, 4.36541176] -> [2.04047704, 0.6649279, 8.73082352]
c3_fresh_equivalence  pass  11.496  visible & chain of 3 live == fresh FSDP2 subprocess
c4_uninstall          pass   0.982  uninstall in 0.0445 s; base identity restored
c5_failed_install     pass   1.098  fail_apply & fail_commit rolled back; 3 targets in 2 modules committed
c6_refusal            pass   1.097  patch_invalid, closure_changed, outside_changes, empty_change refused
c7_chain              pass   1.294  3 stacked edits undone step-by-step to base
c8_crash_and_hang     pass  28.452  crash/hang/lost/tamper poisoned -> respawn 2.65-2.78 s
c9_soak               pass   3.874  20 cycles; install median 0.0555 s, uninstall median 0.0468 s
c10_not_proven        pass   4.317  proven False (zero_witness) on startup-only init_weight_scale; fresh diverges

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
    res = lab.install(lab.patches()["visible"])        # 0.054 s install vs 2.650 s cold boot (35.4x)
    edited = lab.check()                               # 0.020 s check, proven=True, witness=3
    lab.uninstall()                                    # 0.046 s undo + S0 restore
    restored = lab.check()
    max_diff = max(abs(a - b) for a, b in zip(base["output"]["losses"], restored["output"]["losses"]))
    assert max_diff == 0.0
finally:
    lab.close()
```

Observed values on 1× NVIDIA B200:
- `base["output"]["losses"]`: `[1.08151245, 0.21941125, 8.59863853]`
- `edited["output"]["losses"]`: `[2.1630249, 0.43885228, 16.2074585]` (`witness={'rank0': {'adapter.model_ops:compute_loss:37': 3}}`)
- `max_diff`: `0.0` (bit-exact `S0` weight and `AdamW` rollback)
