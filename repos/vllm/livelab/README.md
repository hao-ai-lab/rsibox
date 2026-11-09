# vLLM live lab

vLLM live-lab integration built on the shared transaction kernel in `core/livelab/`.

- The transaction kernel (`core/livelab/kernel/`, engine-agnostic, standard library only) handles
  staging, checks, receipts, and undo journals; importing `adapter` puts `core/livelab` on
  `sys.path` (`FASTRSI_LIVELAB` overrides the folder).
- `adapter/`: the kernel inside a live vLLM engine: the engine-core host (`host.py`), the
  worker/core transaction side (`side.py`), the worker extension (`worker_ext.py`), graph
  reachability (`graphs.py`), selective CUDA graph recapture (`recapture.py`), and the control
  client (`client.py`).
- `services/`: warm restart and process services around `vllm serve`: the server lifecycle and
  reconfiguration CLI (`engine.py`), argument delta classification (`vllm_args.py`), the
  pre-import fork zygote (`zygote.py`, `zygote_refresh.py`), the IPC weight cache daemon wrapper
  (`weights.py`), and the KV-cache plan reuse hook (`kvplan.py`).
