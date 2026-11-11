# FastVideo live lab

A warm FastVideo engine (today: MiniMax H3 / FastH3, `kb/workload.md`) that takes code edits
live and proves they ran. Implements `core/livelab/INTERFACE.md` (the v1 draft verbs) and
the conformance harness of `core/livelab/conformance/`.

The transaction kernel and the pre-fork guards are the framework's: `core/livelab/kernel/`
and `core/livelab/forksafe.py`. Importing `adapter` (or `services.zygote`) puts
`core/livelab` on `sys.path` (`FASTRSI_LIVELAB` overrides the folder).

- `adapter/`: the kernel inside FastVideo: `agent.py` runs in every worker process,
  `lab.py` is the lab host (the verbs), `patches.py` the conformance edits on real H3 code,
  `fresh.py` the fresh process `fresh_check` starts.
- `services/`: the actions cheaper than a plain restart (`kb/costs.md`, "Restart ladder"):
  - `rebuild.py`: re-import edited modules in a warm worker and rebuild what was built from them
    (in-place rebuild, the `rebuild` verb; 0.2-1.3 s, measured);
  - `jitcache.py`: path-stable JIT cache roots and a `torch_key` preset (warm restart with
    path-stable caches; first-video JIT 6.3 -> 2.2 s, measured);
  - `zygote.py`: fork workers from a preloaded forkserver, refreshing modules edited since it
    imported them, with Triton's GPU target and `CUTE_DSL_ARCH` pinned while it imports
    (`forksafe.pin_triton_target`, `pin_cute_dsl_arch`); when the preload initialized CUDA anyway
    (driver or torch runtime), forked workers refuse to start (zygote respawn; 14.1 s to ready,
    measured);
  - `holder.py`: keep the loaded DiT, text encoder and VAE in a holder process that respawned
    workers map by CUDA IPC, and refuse to serve when loader, quantization or model code changed
    (zygote respawn with weights holder; 4.8 s to ready, measured).

## Process layout

```
lab host (driver process: VideoGenerator, Session, Coordinator, receipts)
  └─ MultiprocExecutor
       ├─ worker rank 0 (GPU 0): pipeline + agent.LocalParticipant "rank0"
       ├─ worker rank 1 (GPU 1): pipeline + agent.LocalParticipant "rank1"
       └─ ...                         (N = sp_size, sequence parallel)
```

- **The engine always runs in spawned workers**, also on one GPU, where FastVideo would build
  its worker in the caller's process (`worker/executor.py:63-71` at the pin). The lab patches
  `Executor.get_class` for N = 1. Why: a respawn must re-import base code (a REPL edit or a
  failed undo in the host process could not be taken back), and every rank is reached the same
  way. Cost: the frames cross a process boundary (measured in `kb/costs.md`).
- **Ranks are reached through each worker's own pipe**, with `agent.rpc` as the callable
  (FastVideo's worker busy loop runs a callable as `method(worker, *args)`). One participant
  per rank, so a reply missing from one rank is never hidden in a merged reply.
- **Deadline:** FastVideo's `collective_rpc` ignores its `timeout` (its `recv` is unbounded).
  The lab polls each pipe with its own deadline; a reply that misses it leaves a stale reply
  in that pipe, so the lab marks itself dirty and refuses every call until a respawn.

## Where the step gap is

A generation is one `execute_forward` RPC per worker. The denoise loop (8 DiT forwards,
`minimax_h3_denoising.py:298-362` at the pin) runs inside it, and a worker reads its pipe only
between requests. So the lab's quiesce point is **the gap between two requests**: phase 2 of
every transaction (validate, commit or abort) runs there, on every rank, before the next video
starts. An edit never switches in the middle of a video, and ranks never run one video on
different code. The loop top (`self.interrupt` check) is the only finer quiesce point inside a
request; it is not a hook today, and edit-to-run does not need it (a video takes seconds).

## Verbs

| Verb | What it does on FastVideo |
|---|---|
| `install(patch)` | `patch = {"files": {module: full new text}, "accept": [...], "edit_time": t}`. Live swap of every changed function on every rank, in one request gap. Refusals as the kernel's findings |
| `check(workload)` | One generation of the workload in the warm engine. `output` = hashes of the uint8 frames and the audio; `seconds`; `psnr_vs_golden`; per-rank witness of every live edit; `proven` |
| `reset("state")` | Empties each rank's CUDA cache. Request state (`ForwardBatch`, latents, RNG) is built per request by FastVideo |
| `rebuild(patch)` | In-place rebuild, no new process: every worker re-imports the edited modules and their importers, live objects get the new classes, constructor edits are merged into the loaded DiT against meta builds (weights stay), stages are built again. For the edits the live swap refuses (constructors, new fields, module constants). Needs no live edits installed; the rebuilt text is the new base (a later `install` is made against it, and a later rebuild re-imports an importer from it, not from its file). A rank that fails after it began to change runs its rollback steps (class swaps, field merges, the pipeline's attributes and their dict/list/set contents, the modules); a rank that does not report every step ran poisons the lab (outcome `failed`; `partial` when only some ranks rebuilt). Changes deeper inside objects the pipeline holds are not rolled back. `{"revert": True}` goes back to the files |
| `reset("respawn")` | Shuts the worker group down and builds a new one from the same config; clears poison |
| `uninstall(id)` | Undo `id` and every later edit, newest first; checks the code fingerprint |
| `status()` | base / edited / poisoned, the installed transactions, the code fingerprint |
| `repl(code, rank)` | Runs a cell on every rank with `worker`, `pipeline`, `participant` in scope; `out` is returned. The notebook verb (AGENTS.md) |

The session fingerprints the H3 stage modules, the H3 pipeline and the H3 DiT module
(`lab.MODULES`). An edit to another module installs and is witnessed, but a hand edit there is
not caught by the tamper check.

## Not covered

- Code inside `torch.compile` regions (the VAE is compiled in the workload; the DiT is eager),
  Triton kernel sources (`@triton.jit`: the live swap replaces the Python wrapper, not the compiled kernel),
  CUDA graphs of the NVFP4 VAE decoder, and native `fastvideo-kernel` code. See
  `docs/architecture.md` for their restart classes.
- Constructors and fields (`constructor_edit`, `new_fields` block in `install`, as in the
  kernel): `rebuild` runs them in the warm workers unless they need new weights.
