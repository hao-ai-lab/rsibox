# Walkthrough: replaying `hao-ai-lab/FastVideo#1900` in the H3 live lab

Replay of upstream PR `hao-ai-lab/FastVideo#1900` (fused VSA64 tile layout for SM100) on 1 GB200
GPU at the FastVideo `376e445f` pin (`../../kb/workload.md`), from our 2026-11-07 run. Because
`#1900` is already in `376e445f`, it is replayed in reverse (`-R`): applying the reverse diff,
probing a live `install` into the warm engine, and running a warm restart in a new process when
the live swap refuses the module-level and deleted-helper changes.

The live lab hosting this session passes the template conformance suite
([`../h3-conformance/README.md`](../h3-conformance/README.md)).

## 1. Apply the upstream diff in reverse

`#1900` touches three non-test modules (`fastvideo/attention/backends/video_sparse_attn.py`,
`fastvideo/envs.py`, and a comment in `fastvideo/pipelines/stages/denoising.py`). Applying it in
reverse deletes the two module-level helpers `_get_tile_to_bhsd` and `_can_fuse_vsa64_layout`,
two dataclass fields on `VideoSparseAttentionMetadata`, and `FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT`
in `fastvideo/envs.py`:

```
$ patch -p1 -F3 -R -i pr1900.src.diff
patching file fastvideo/attention/backends/video_sparse_attn.py
patching file fastvideo/envs.py
Hunk #1 succeeded at 379 (offset 1 line).
patching file fastvideo/pipelines/stages/denoising.py
Hunk #1 succeeded at 1207 (offset -24 lines).
Hunk #2 succeeded at 1331 (offset -24 lines).
```

## 2. Base checks in the warm lab

```
$ fvlab status
{"ok": true, "state": "base"}
$ fvlab check   # x3
{"tag": "base",
 "seconds": [6.034, 6.0147, 6.0172],
 "median": 6.0172,
 "outputs": [
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}
 ],
 "proven": [null, null, null],
 "proven_by_txn": [{}, {}, {}],
 "errors": []}
```

## 3. Live swap (`install`): refused in 0.20 s

```
$ fvlab install \
    fastvideo/attention/backends/video_sparse_attn.py \
    fastvideo/envs.py \
    fastvideo/pipelines/stages/denoising.py
{"ok": false,
 "outcome": "aborted",
 "codes": ["outside_changes", "removed_definition"],
 "mechanisms": [],
 "edit_to_run_s": null,
 "cost_s": 0.20199,
 "id": "txn-425ecdc7",
 "findings": [
  {"check": "removed_definition",
   "observed": "_can_fuse_vsa64_layout is deleted by the edit but defined in the live module",
   "subject": "fastvideo.attention.backends.video_sparse_attn:_can_fuse_vsa64_layout",
   "participant": "rank0"},
  {"check": "removed_definition",
   "observed": "_get_tile_to_bhsd is deleted by the edit but defined in the live module",
   "subject": "fastvideo.attention.backends.video_sparse_attn:_get_tile_to_bhsd",
   "participant": "rank0"},
  {"check": "outside_changes",
   "observed": "the module differs outside the target bodies (beyond added definitions and imports)",
   "subject": "fastvideo.attention.backends.video_sparse_attn",
   "participant": "rank0"},
  {"check": "outside_changes",
   "observed": "the module differs outside the target bodies (beyond added definitions and imports)",
   "subject": "fastvideo.envs",
   "participant": "rank0"}
 ]}
```

The live function-body swap aborts cleanly in `0.202 s` without touching the running worker:
deleting `_can_fuse_vsa64_layout` and `_get_tile_to_bhsd`, changing dataclass attributes on
`VideoSparseAttentionMetadata`, and editing module-level environment definitions in
`fastvideo.envs` require a restart.

## 4. Warm restart in a new process

Starting a fresh FastVideo process from the reverse-patched tree with warm page and JIT caches (4
generations: 1 warmup + 3 measured):

```json
{"status": "completed",
 "imports_s": 27.0837,
 "load_s": 15.3239,
 "first_video_s": 12.9439,
 "warm": [6.1893, 6.0492, 6.6072],
 "warm_median": 6.1893,
 "hashes": [["d0e55e892690e260", "29b690c8ac7ae3be"]],
 "bit_exact_vs_pin_golden": true,
 "psnr": [Infinity, Infinity, Infinity, Infinity],
 "restart_to_first_video_s": 107.4,
 "error": ""}
```

The output hashes are bit-exact with the pin (`d0e55e892690e260` / `29b690c8ac7ae3be`) and warm
generation median is `6.189 s`: MiniMax H3 uses its own VSA path (`minimax_h3.py`), not the
generic `VideoSparseAttentionImpl` in `video_sparse_attn.py`.

## Conformance receipt and what this does not show

- Lab conformance receipt: [`../h3-conformance/README.md`](../h3-conformance/README.md) (`c0`-`c10`
  on the same `FastVideoLab` adapter and H3 engine).
- Not shown: a model pipeline that routes through `VideoSparseAttentionImpl.preprocess_qkv` in
  `fastvideo/attention/backends/video_sparse_attn.py`, or a per-function witness inside the
  fresh-process run.
