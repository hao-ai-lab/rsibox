# HuggingFace Transformers

The HuggingFace Transformers live-lab adapter: its single-worker process architecture, restart and
quiesce boundaries, measured B200 timings in its knowledge base, and its RSIBox live-edit runtime.

| Folder | Holds |
|---|---|
| [docs/](docs/README.md) | Process tree, restart boundaries, snapshot quiesce/resume lifecycle, and how RSIBox maps onto them |
| [kb/](kb/README.md) | Measured cold-boot, live-edit, and snapshot quiesce/resume costs, target workload F(x), and lessons |
| `livelab/` | Step-gap adapter (`adapter/`), `TransformersQuiesceHooks`, CPU `toy_hf` conformance fixture, and unit tests |
