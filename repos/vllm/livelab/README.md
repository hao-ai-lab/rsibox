# vLLM live lab

Run from the repository root with the vLLM environment's Python and the GPUs assigned
to the run. Each command starts its own servers and keeps scratch files under `--work`.

```bash
CUDA_VISIBLE_DEVICES=0 python repos/vllm/livelab live-edit --model "$MODEL" --work "$WORK/edit"
CUDA_VISIBLE_DEVICES=0 python repos/vllm/livelab conformance --model "$MODEL" --work "$WORK/eager"
CUDA_VISIBLE_DEVICES=0 python repos/vllm/livelab conformance --model "$MODEL" --work "$WORK/graphs" --graphs
CUDA_VISIBLE_DEVICES=0,1 python repos/vllm/livelab conformance --model "$MODEL" --work "$WORK/tp2" --tp 2
```

Conformance defaults to eager execution. `RSIBOX_VLLM_EXTRA` replaces the default serving
flags, including the CUDA graph capture sizes.
Use `--receipts FILE` to keep transaction receipts and `--only c2_visible` for a single case.

vLLM live-lab integration built on the shared transaction kernel in `core/livelab/`.

- The transaction kernel (`core/livelab/kernel/`, engine-agnostic, standard library only) handles
  staging, checks, receipts, and undo journals; importing `adapter` puts `core/livelab` on
  `sys.path` (`RSIBOX_LIVELAB` overrides the folder).
- `adapter/`: the kernel inside a live vLLM engine: the engine-core host (`host.py`), the
  worker/core transaction side (`side.py`), the worker extension (`worker_ext.py`), graph
  reachability (`graphs.py`), selective CUDA graph recapture (`recapture.py`), and the control
  client (`client.py`).
- `services/`: warm restart and process services around `vllm serve`: the server lifecycle and
  reconfiguration CLI (`engine.py`), argument delta classification (`vllm_args.py`), the
  pre-import fork zygote (`zygote.py`, `zygote_refresh.py`), the IPC weight cache daemon wrapper
  (`weights.py`), and the KV-cache plan reuse hook (`kvplan.py`).
