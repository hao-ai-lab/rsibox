# Prices of FastVideo experiment kinds

What each kind of experiment the loop runs costs before it runs: the restart action it needs,
its predicted cost from `costs.md`, the base-to-base noise of its metric, and the cheapest
probe that answers its question (`core/agents/toolsmith_online.md`). A kind not in this table is not
priced, and the loop does not run it.

| Kind | Restart action | Predicted cost | Base-to-base noise | Cheapest probe | Source |
|---|---|---|---|---|---|
| Replay of an upstream PR | live swap (install 0.2 s, no restart) for function-body changes in loaded modules; module constants (`envs.py`), new files and constructors are refused by the live swap and need an in-place rebuild (1.0-2.8 s) or a warm restart in a new process | live swap: install 0.05-0.3 s + 8 checks of one video, about 50 s; warm restart in a new process: 86-107 s from the restart to the end of its first video | lab, medians of 3 checks: +-0.5% on one engine; new process against new process, both at base: 1.1% (6.09 vs 6.16 s) | a live swap and one check: shows in seconds whether the PR's code is reached (witness) and whether the output is exact; a new process only for what the live swap refuses | our 2026-11-07 replays of four H3 PRs, 1 GPU, measured |
| Hill-climb edit (agent proposes, harness judges) | live swap (install 0.2 s, no restart); an edit the live swap refuses takes an in-place rebuild (1.0-2.8 s) or a worker respawn (40-80 s); a crash takes a worker respawn and a reinstall of the kept edits (41-79 s) | baseline 3 checks + agent proposal 30-120 s + install about 1 s + 3 checks: 60-150 s per iteration at 1 GPU | +-1% between 3-check baselines on one engine at 1 GPU; one baseline at 2 GPUs moved 8%; a baseline taken right after a restart or an undo can read slow and make an edit look faster, so a keep is measured again against a fresh baseline | one check after the install (hashes and witness) before measuring | our 2026-11-07 hill-climb runs, 1 and 2 GPUs, measured |

Not priced: sweeps of engine settings read once per process (one warm restart in a new
process each), and any kind at 4 GPUs.
