# TensorRT-LLM (PyTorch backend) live lab

RSIBox live-lab adapter for NVIDIA TensorRT-LLM's PyTorch backend (`tensorrt_llm.LLM(..., backend="pytorch")`), completing the NVIDIA Dynamo serving triad (`vLLM`, `SGLang`, `TensorRT-LLM`).

## Layout

- `docs/architecture.md` — process tree, single-process worker mode (`TLLM_WORKER_USE_SINGLE_PROCESS=1`), step-gap transaction boundary, and snapshot quiesce path.
- `kb/workload.md` — deterministic workload `F(x)` and correctness gate.
- `kb/costs.md` — real NVIDIA B200 measurements (Cold-JIT vs Warm-JIT boot, live edit/undo latency, fresh parity, and `TrtllmQuiesceHooks` KV pool release/resume VRAM trajectory).
- `kb/lessons.md` — verified pitfalls and rules for TensorRT-LLM's PyTorch backend.
- `livelab/` — `livelab.json` manifest and `adapter/` (`TrtllmLab`, `ToyTrtllmLab`, `agent.py`, `TrtllmQuiesceHooks`, CPU fixture, and golden tests).

## Quick commands

```bash
# CPU doctor and 11/11 conformance on ToyTrtllmLab
PYTHONPATH=core/livelab python3 -m rsibox doctor
PYTHONPATH=core/livelab python3 -m rsibox conformance trtllm

# Adapter unit tests
python3 -m unittest discover -s repos/trtllm/livelab/adapter -t repos/trtllm/livelab -v
```
