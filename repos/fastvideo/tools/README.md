# FastVideo probes

Proven probes that drive a warm H3 lab (`livelab/adapter/h3.py`; its environment variables
name the weights) and print one JSON line per step. Run them with the Python of a FastVideo
environment, the FastVideo tree to test first on `PYTHONPATH`, on free GPUs.

| Probe | What it answers | Example |
|---|---|---|
| `h3_shapes.py` | Do several request lengths run in one process, bit-exact to a reference, and what GPU memory and CUDA-graph cache does each leave? | `--frames 124,243,124 --reference base.jsonl` (the GPU test of `patches/h3-hillclimb` patch 20) |
| `h3_climb.py` | Do the hill-climb patches go live in a warm base engine, bit-exact after each one, and how fast is the result against base? Also in-place rebuild (`--rebuild --only 15,16,17,20`) and respawns (`--respawns N --zygote --holder`) | `--upto 20 --frames 243,124 --reference base.jsonl` |

Both exit 1 when a video fails or differs from what it must equal. A fresh-process A/B is
`h3_shapes.py --frames 124 --repeat 6` once per tree, trees alternating (1 cold + 5 warm
videos each).
