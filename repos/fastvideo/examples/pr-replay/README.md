# Standing replay of real upstream H3 PRs

The FastVideo version of the replay gate we keep for vLLM: real upstream PRs that touch the
MiniMax H3 inference path, replayed through the live lab with a record each. Our 2026-11-07
runs, 1 GPU, lab at FastVideo `376e445f` (`kb/workload.md`).

How a PR is replayed:

- A PR already in the pin is replayed backwards: its diff is applied in reverse (the pre-PR
  code) and installed live, measured, and undone back to the pin. A PR after the pin is applied
  forwards. Upstream test files are left out; a diff that does not apply cleanly is applied
  with fuzz, and a file it breaks is recorded as "does not apply".
- Live swap (install 0.2 s, no restart, measured): 3 checks of one video before, 3 after, 2 after
  the undo; the witness says whether the PR's code ran. Edit-to-run is from the edit's
  timestamp to the commit: the install plus the agent's round trip to the lab.
- Warm restart in a new process (a new FastVideo process from the patched files, page and JIT
  caches warm; 86-107 s from the restart to the end of its first video, measured), for every
  PR, so a refused live swap still gets a measurement.

| PR | Direction | Files (non-test) | Live swap | Live-swap measurement | Warm restart in a new process |
|---|---|---|---|---|---|
| hao-ai-lab/FastVideo#1882 (reuse cached VAE offload in H3 conditioning) | reverse | latent preparation stage | committed, edit-to-run 0.81 s | 6.10 s base, 6.00 s edited, 6.01 s after undo; **witness 0**: the changed code (reference-image VAE encoding) is not reached by text-to-video, so this workload cannot measure it | 6.04 s warm, bit-exact; 107 s from the restart to the end of the first video (model load 13.2 s, the first video itself 14.8 s) |
| hao-ai-lab/FastVideo#1900 (fused VSA64 tile layout for SM100) | reverse | generic VSA backend, `envs.py`, generic denoising stage | refused: `removed_definition` (two helpers the reverse deletes), `outside_changes` (module-level code in `envs.py`) | none | 6.19 s warm, bit-exact: the generic VSA path is not used by H3; 107 s from the restart to the end of the first video |
| hao-ai-lab/FastVideo#2040 (decode eight VAE tiles per call by default) | forward | `envs.py`, two H3 VAE modules | refused: `outside_changes` (an `envs.py` default and module-level code) | none | 6.13 s warm, bit-exact: the workload already sets the tile batch to 12, so the default does not matter here; 86 s from the restart to the end of the first video |
| hao-ai-lab/FastVideo#2041 (fused VAE decoder QK RMSNorm and partial RoPE) | forward | `envs.py`, H3 VAE, one new file | `patch_invalid`: the fuzzy apply broke `envs.py` (the PR does not apply to the pin) | none | none |

What the records teach is in `kb/lessons.md`; the prices of a replay are in `kb/prices.md`.
