---
name: rsibox-gpu-receipt
description: Enforce honest CPU vs GPU verification claims, multi-shape fresh runs, non-zero witness checks, cold/warm separation, and status:needs-gpu issue tracking.
---

# GPU Verification & Receipt Honesty

1. Treat CPU `ToyLab 11/11` and mocks strictly as shape checks; mark GPU execution under `Not verified` unless measured on real hardware.
2. When running on GPU: acquire dedicated idle GPUs without `pkill -f`, test `>= 2` shapes in alternating fresh processes, and report cold compile separately from warm runs.
3. Confirm witness outputs are non-zero (`witness.py`) and replay paths exercised rather than capture-only.
4. For any `Not verified` GPU gap in a merged PR, link or open a `status:needs-gpu` issue in the `GPU Receipts Backlog` milestone.
