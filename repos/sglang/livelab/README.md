# SGLang live lab

Two boundaries:

- `adapter/`: scheduler step-gap host (`lab.py`), per-rank `LocalParticipant` and overlap drain
  (`agent.py`), selective `CudaGraphRunner` decode-graph recapture (`recapture.py`), the real
  `sglang.Engine` conformance wrapper (`conformance_lab.py`), and the CPU `toysgl` harness
  (`toy_lab.py`, `fixture/`).
- `services/`: `forksafe`-guarded pre-import fork server (`zygote.py`).

## CPU verification (no GPU or `sglang` install required)

From the repository root:

```sh
PYTHONWARNINGS=error python3 -m unittest discover -s repos/sglang/livelab/adapter -t repos/sglang/livelab -v
PYTHONWARNINGS=error python3 -m unittest discover -s repos/sglang/livelab/services -t repos/sglang/livelab -v
PYTHONPATH=repos/sglang/livelab python3 -m adapter.test_bug_replays
(cd core/livelab && PYTHONPATH=../../repos/sglang/livelab python3 -m conformance --lab adapter.toy_lab:ToySglangLab)
```

## Real-GPU conformance (`sglang.Engine`)

```sh
export FASTRSI_SGLANG_MODEL=<model-path-or-hf-id>
export FASTRSI_SGLANG_TP=1
cd core/livelab
PYTHONPATH=../../repos/sglang/livelab python3 -m conformance --lab adapter.conformance_lab:SglangConformanceLab
```
