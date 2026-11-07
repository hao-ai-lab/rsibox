# Walkthrough: a live edit of MiniMax H3's video decode stage

One edit, installed into a warm FastH3 engine (1 GB200 GPU, the workload of `kb/workload.md`),
checked, and undone. Real output from our 2026-11-07 run, trimmed to the fields that matter.
The agent edits its copy of the FastVideo source and calls the lab's verbs; the lab answers
from the warm engine. The edit adds 1 to every pixel before the uint8 cast: visible in the
hashes, small in PSNR.

## 1. The lab is warm, at base

```
$ fvlab status
{"ok": true, "state": "base", "txns": []}
$ fvlab check
{"ok": true, "seconds": 6.013, "output": {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"},
 "psnr_vs_golden": Infinity, "proven": null}
```

`proven: null`: no edit is live. The hashes equal those of a plain `VideoGenerator` run of the
same request in FastVideo's in-process executor (`kb/costs.md`).

## 2. Edit and install

```diff
--- fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_decoding.py
-                output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
+                output = output.mul_(255).add_(1).clamp_(0, 255).to(torch.uint8)
```

```
$ fvlab install fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_decoding.py
{"ok": true, "outcome": "committed", "mechanisms": ["R0 swap"], "edit_to_run_s": 0.72258,
 "cost_s": 0.059647}
```

The kernel found the one changed method (`MiniMaxH3VideoDecodingStage.forward`, decorated with
`@torch.no_grad()`), swapped its code on the worker rank in the gap between two requests, and
wrote a `txn` receipt. 0.06 s in the lab; 0.72 s from the edit's timestamp to commit, including
the agent's hop to the lab.

## 3. Check: the new code ran

```
$ fvlab check
{"ok": true, "seconds": 6.0226, "output": {"video": "cd32f50b70d9479f", "audio": "29b690c8ac7ae3be"},
 "psnr_vs_golden": 48.133, "proven": true,
 "witness": {"rank0": {"fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding:MiniMaxH3VideoDecodingStage.forward:109": 1}}}
```

The video hash changed, the audio did not (the edit is in the video stage only), and the witness
counted one call of the new code on rank 0: one video, one decode.

## 4. Undo

```
$ fvlab uninstall
{"ok": true, "undone": ["txn-89aae11a"], "state": "base", "cost_s": 0.014432}
$ fvlab check
{"ok": true, "seconds": 6.013, "output": {"video": "d0e55e892690e260", "audio": "29b690c8ac7ae3be"}, "proven": null}
```

Back to the base hashes, bit for bit. A restart to see the same edit costs about 72 s warm and
about 500 s cold (`kb/costs.md`).

## What this does not show

Two or more ranks (the conformance run on 2 GPUs does), edits inside compiled or captured code,
or a speedup: this edit only proves the path.
