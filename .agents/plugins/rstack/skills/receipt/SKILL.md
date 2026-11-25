---
name: receipt
description: Enforce honest CPU vs GPU verification claims, multi-shape fresh runs, non-zero witness checks, cold/warm separation, and status:needs-gpu issue tracking.
---

# GPU Verification & Receipt Honesty

1. Treat CPU runs of the sandbox verification suite (`rsibox conformance`, e.g., `ToyLab`) and mocks strictly as shape checks; mark GPU execution under `Not verified` unless measured on real hardware.
2. When running on GPU: acquire dedicated idle GPUs without `pkill -f`, test `>= 2` shapes in alternating fresh processes, and report cold compile separately from warm runs.
3. Validate structured receipts before referencing them: `python3 .agents/plugins/rstack/scripts/gpu_receipt_close.py validate --receipt <receipt.json>`.
4. For any `Not verified` GPU gap in a merged PR, link or open a `status:needs-gpu` issue via `python3 .agents/plugins/rstack/scripts/gpu_receipt_close.py backlog --title "<title>" --pr <N> --engine <engine>`.
