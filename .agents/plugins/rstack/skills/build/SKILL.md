---
name: build
description: Implement a single-boundary RSIBox change under <= 600 non-test LOC with formal TLA+ updates, ResourceWarning cleanliness, and doctor checks.
---

# Controlled Implementation

1. Keep edits inside one boundary (`<= 600` non-test LOC, zero cross-engine imports); check continuously with `python3 .agents/plugins/rstack/scripts/preflight.py --base origin/main`.
2. Update `.tla` specs and conformance tests whenever state-machine transitions (`SandboxPool`, `Coordinator`, `Swap`, `StandbyShadow`, `PoolAutoScaler`) change.
3. Close sockets, HTTP responses, files, and subprocesses cleanly so tests pass under `python3 -W error::ResourceWarning`.
4. Verify zero eager GPU imports via `PYTHONPATH=core/livelab python3 -m rsibox doctor`.
