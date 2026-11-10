# vLLM knowledge base

What agents read before an experiment on vLLM. Every number names its source and setup.
Prior measurements are copied in here first; new ones are added as they are taken.

| File | Holds |
|---|---|
| [costs.md](costs.md) | Measured startup, restart and check costs: where the time goes |
| [workload.md](workload.md) | The target workload F(x) and its correctness gate |
| [mechanisms.md](mechanisms.md) | Which edits install live, how, and what needs a restart |
| [reconfiguration_matrix.md](reconfiguration_matrix.md) | Every `vllm serve` flag's reconfiguration tier (`<0.1s` in-process vs `1s` graph vs `7s` warm IPC vs reshard), mutated objects, and effort |
| [lessons.md](lessons.md) | vLLM pitfalls, the rules that prevent them and their evidence |

Raw measurement files go in `measurements/`, one file per run, named `<date>-<what>.json`.
