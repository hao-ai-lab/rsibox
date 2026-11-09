# Walkthrough: replaying `hao-ai-lab/FastVideo#1882` in the H3 live lab

Replay of upstream PR `hao-ai-lab/FastVideo#1882` (reuse cached VAE offload in MiniMax H3
conditioning) on 1 GB200 GPU at the FastVideo `376e445f` pin (`../../kb/workload.md`), from our
2026-11-07 run. Because `#1882` is already in `376e445f`, it is replayed in reverse (`-R`):
installing the pre-PR code into the warm engine, running 3 workload checks, and uninstalling back
to the pin, followed by a warm restart in a new process.

The live lab hosting this session passes the template conformance suite
([`../h3-conformance/README.md`](../h3-conformance/README.md)).

## 1. Apply the upstream diff in reverse

`#1882` touches one non-test module, swapping `pinned_offload.load` / `pinned_offload.unload`
for `self.vae.to(...)` and `self.audio_vae.to(...)` in `_encode_fl2va_conditions` and
`_encode_ref2va_conditions`:

```
$ patch -p1 -F3 -R -i pr1882.src.diff
patching file fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py
Hunk #1 succeeded at 12 with fuzz 2.
Hunk #2 succeeded at 191 (offset 9 lines).
Hunk #3 succeeded at 201 (offset 9 lines).
Hunk #4 succeeded at 210 (offset 9 lines).
Hunk #5 succeeded at 236 (offset 9 lines).
```

```diff
--- a/fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py
+++ b/fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py
@@ -194,7 +193,7 @@
         vae_device = get_local_torch_device()
-        pinned_offload.load(self.vae, vae_device, pin=fastvideo_args.pin_cpu_memory)
+        self.vae.to(vae_device)
         try:
             clean_rows: list[torch.Tensor] = []
             for image in keyframes:
                 clean_rows.append(
                     encode_pixels_parallel(self.vae, image.to(device=vae_device, dtype=torch.bfloat16),
                                            h3_dit_patch_size(fastvideo_args)))
         finally:
             if fastvideo_args.vae_cpu_offload:
-                pinned_offload.unload(self.vae)
+                self.vae.to("cpu")
```

## 2. Base checks in the warm lab

```
$ fvlab status
{"ok": true, "state": "base"}
$ fvlab check   # x3
{"tag": "base",
 "seconds": [15.8828, 6.0375, 6.0968],
 "median": 6.0968,
 "outputs": [
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}
 ],
 "proven": [null, null, null],
 "proven_by_txn": [{}, {}, {}],
 "errors": []}
```

The first check after engine start takes `15.8828 s` (first-request warmup); the next two take
`6.0375 s` and `6.0968 s` (median `6.0968 s`), all three bit-exact with the golden output hashes.

## 3. Live swap (`install`) and edited checks

```
$ fvlab install fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py
{"ok": true,
 "outcome": "committed",
 "codes": [],
 "mechanisms": ["R0 swap"],
 "edit_to_run_s": 0.808512,
 "cost_s": 0.086041,
 "id": "txn-5d011902",
 "findings": []}
```

The kernel swaps `_encode_fl2va_conditions` and `_encode_ref2va_conditions` on `rank0` in
`0.086 s` (`0.809 s` edit-to-run including the client hop). Running 3 checks with `txn-5d011902`
installed:

```
$ fvlab check   # x3
{"tag": "edited",
 "seconds": [6.0035, 6.0004, 6.0008],
 "median": 6.0008,
 "outputs": [
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}
 ],
 "proven": [false, false, false],
 "proven_by_txn": [
  {"txn-5d011902": false},
  {"txn-5d011902": false},
  {"txn-5d011902": false}
 ],
 "errors": []}
```

`proven: false` on all three checks (witness count `0`): the text-to-video workload in
`kb/workload.md` has no keyframe or reference-image conditioning, so neither edited method is
called. Without the per-function witness, the `6.0008 s` median and matching hashes would look
like a verified replay.

## 4. Undo (`uninstall`) and base confirmation

```
$ fvlab uninstall
{"ok": true,
 "undone": ["txn-5d011902"],
 "state": "base",
 "cost_s": 0.030073}
$ fvlab check   # x2
{"tag": "base again",
 "seconds": [6.0016, 6.0084],
 "median": 6.005,
 "outputs": [
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
  {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}
 ],
 "proven": [null, null],
 "proven_by_txn": [{}, {}],
 "errors": []}
```

## 5. Warm restart in a new process

Starting a fresh FastVideo process from the patched source tree with warm page and JIT caches (4
generations: 1 warmup + 3 measured):

```json
{"status": "completed",
 "imports_s": 31.4341,
 "load_s": 13.2073,
 "first_video_s": 14.8325,
 "warm": [6.0276, 6.045, 6.7013],
 "warm_median": 6.045,
 "hashes": [["d0e55e892690e260", "29b690c8ac7ae3be"]],
 "bit_exact_vs_pin_golden": true,
 "psnr": [Infinity, Infinity, Infinity, Infinity],
 "restart_to_first_video_s": 106.9,
 "error": ""}
```

## Conformance receipt and what this does not show

- Lab conformance receipt: [`../h3-conformance/README.md`](../h3-conformance/README.md) (`c0`-`c10`
  on the same `FastVideoLab` adapter and H3 engine).
- Not shown: a keyframe (`FL2VA`) or reference-media (`Ref2VA`) request that exercises
  `_encode_fl2va_conditions` or `_encode_ref2va_conditions` with `vae_cpu_offload` enabled.
