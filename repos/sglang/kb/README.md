# SGLang knowledge base

What agents read before an experiment on SGLang. Every number names its source and setup.
Prior measurements are copied in here first; new ones are added as they are taken.

| File | Holds |
|---|---|
| [costs.md](costs.md) | Measured startup, swap, reset and recovery costs, plus static-survey coverage |
| [workload.md](workload.md) | The target workload F(x) and its correctness gate |
| [lessons.md](lessons.md) | SGLang pitfalls, the rules that prevent them and their evidence |
| [survey_rules.json](survey_rules.json) | Path, decorator and capability declarations for `core/livelab/survey` |

Raw measurement files go in `measurements/`, one file per run, named `<date>-<what>.json`.
