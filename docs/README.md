# Docs

Docs about how this repo and its process work. Nothing engine-specific lives here: an engine's
architecture goes in `repos/<engine>/docs/`, its measurements in `repos/<engine>/kb/`.

```
docs/
  README.md               this index
  lessons.md              framework lessons from building a live lab
  kernels-and-graphs.md   bytecode swap, Triton @jit rebuild, CUDA graphs, and launch witnesses
  core/                   how the skill and the agent prompts work
  livelab/                the interface, receipts, conformance
```

Rules:

- Make a folder only when its first doc arrives. No empty placeholders.
- Docs describe how things are now. Update them in place; no decision logs or changelogs.
- Add every new doc to the index below.

## Index

- [Lessons from building a live lab](lessons.md): framework lessons and rules for applying them.
- [Bytecode swap, kernel rebuild, and CUDA graph replay](kernels-and-graphs.md): why Python bytecode swaps stop at `@triton.jit` and `CUDAGraph` boundaries, how `template/livelab/kernel/jit.py` rebuilds kernels in place with zero-recompile undo, how the two-ledger witness (`live` vs. `capture`) prevents false positives, how three-level selective and lazy graph recapture avoids full-sweep stalls, and how the whole-sweep update hierarchy (`cuGraphExecKernelNodeSetParams`, `cuGraphExecUpdate`, and lazy recapture) updates all ~50 shape buckets at once.
