# FastVideo knowledge base

What agents read before an experiment on FastVideo. Every number names its source and setup.
Architecture (process layout, restart classes, compile sites) is in `../docs/architecture.md`.
Current workload: MiniMax H3 / FastH3 inference (FastH3-QAD-v2). Pinned source: `376e445f`.

| File | Holds |
|---|---|
| [priors.md](priors.md) | The pinned source, earlier runs, and the real H3 performance PRs the lab replays |
| [workload.md](workload.md) | The target workload F(x) and its correctness gate |
| [costs.md](costs.md) | Measured startup, restart and check costs: where the time goes |
| [lessons.md](lessons.md) | Generic lessons learned while running the live lab |
| [prices.md](prices.md) | The restart action, predicted cost, noise and cheapest probe of each kind of experiment the loop runs |
