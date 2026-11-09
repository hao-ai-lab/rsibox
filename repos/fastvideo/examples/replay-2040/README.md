# Walkthrough: replaying `hao-ai-lab/FastVideo#2040` in the H3 live lab

Replay of upstream PR `hao-ai-lab/FastVideo#2040` (decode eight VAE tiles per call by default) on
1 GB200 GPU at the FastVideo `376e445f` pin (`../../kb/workload.md`), from our 2026-11-07 run.
Because `#2040` merged after `376e445f`, it is replayed forward: applying the PR diff onto the
pin, probing a live `install` in the warm engine, and running a warm restart in a new process
when the live swap refuses the `fastvideo/envs.py` default change.

The live lab hosting this session passes the template conformance suite
([`../h3-conformance/README.md`](../h3-conformance/README.md)).

## 1. Apply the upstream diff forward

`#2040` changes `FASTVIDEO_H3_VAE_TILE_BATCH`'s default from `1` to `8` in `fastvideo/envs.py`
and adds a dynamic NVFP4 activation-scale guard `_has_dynamic_nvfp4_act_scale` in
`fastvideo/models/vaes/minimax_h3_video.py` (plus passes `vae.decoder` to `_tile_batch_size` in
`fastvideo/models/vaes/minimax_h3_parallel.py`):

```
$ patch -p1 -F3 -i pr2040.src.diff
patching file fastvideo/envs.py
Hunk #1 succeeded at 708 (offset -6 lines).
Hunk #2 succeeded at 725 (offset -6 lines).
patching file fastvideo/models/vaes/minimax_h3_parallel.py
patching file fastvideo/models/vaes/minimax_h3_video.py
```

## 2. Base checks in the warm lab

```
$ fvlab status
{"ok": true, "state": "base"}
$ fvlab check   # x3
{"tag": "base",
 "seconds": [6.0101, 6.0467, 6.0154],
 "median": 6.0154,
 "outputs": [
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}
 ],
 "proven": [null, null, null],
 "proven_by_txn": [{}, {}, {}],
 "errors": []}
```

## 3. Live swap (`install`): refused in 0.18 s

```
$ fvlab install \
    fastvideo/envs.py \
    fastvideo/models/vaes/minimax_h3_parallel.py \
    fastvideo/models/vaes/minimax_h3_video.py
{"ok": false,
 "outcome": "aborted",
 "codes": ["outside_changes"],
 "mechanisms": [],
 "edit_to_run_s": null,
 "cost_s": 0.179458,
 "id": "txn-733b8225",
 "findings": [
  {"check": "outside_changes",
   "observed": "the module differs outside the target bodies (beyond added definitions and imports)",
   "subject": "fastvideo.envs",
   "participant": "rank0"}
 ]}
```

The two VAE modules (`minimax_h3_parallel.py` and `minimax_h3_video.py`) only add a helper and
update function bodies, which the live swap accepts, but `fastvideo/envs.py` rebinds the module-level
`FASTVIDEO_H3_VAE_TILE_BATCH = EnvInt(8, ...)` object, so the transaction aborts with
`outside_changes` on `fastvideo.envs` in `0.179 s`.

## 4. Warm restart in a new process

Starting a fresh FastVideo process from the patched tree with warm page and JIT caches (4
generations: 1 warmup + 3 measured):

```json
{"status": "completed",
 "imports_s": 23.5525,
 "load_s": 13.035,
 "first_video_s": 15.3781,
 "warm": [6.0548, 6.1273, 6.5962],
 "warm_median": 6.1273,
 "hashes": [["d0e55e892690e260", "29b690c8ac7ae3be"]],
 "bit_exact_vs_pin_golden": true,
 "psnr": [Infinity, Infinity, Infinity, Infinity],
 "restart_to_first_video_s": 85.9,
 "error": ""}
```

The output hashes (`d0e55e892690e260` / `29b690c8ac7ae3be`) are bit-exact with the pin and warm
generation median is `6.127 s` (`85.9 s` from restart to the end of the first video): our H3
workload environment (`kb/workload.md`) already sets `FASTVIDEO_H3_VAE_TILE_BATCH=12`, so
changing the unset default in `fastvideo/envs.py` from `1` to `8` does not change the tile batch
size on this run.

## Conformance receipt and what this does not show

- Lab conformance receipt: [`../h3-conformance/README.md`](../h3-conformance/README.md) (`c0`-`c10`
  on the same `FastVideoLab` adapter and H3 engine).
- Not shown: a run with `FASTVIDEO_H3_VAE_TILE_BATCH` unset in the environment (where changing
  the default from `1` to `8` changes the VAE decode batch size), or a live swap of only the two
  VAE modules without `fastvideo/envs.py`.
