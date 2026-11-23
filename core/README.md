# rsibox Core

Shared engine-agnostic runtime (`core/livelab/`) and agent role specifications (`core/agents/`).
No engine-specific code belongs under `core/`; per-engine adapters, restart services, knowledge
bases, and experiment logs live in `repos/<engine>/`, and the starter scaffold for onboarding a
new engine lives in `template/`.

```
core/
├── livelab/                   Shared engine-agnostic live-lab runtime:
│   ├── INTERFACE.md             Adapter contract & receipt format (v0).
│   ├── INTERFACE-v1.md          Two-phase transaction contract & recipes (v1).
│   ├── forksafe.py              Pre-fork CUDA guards & NVML stand-ins for engine zygotes.
│   ├── test_forksafe.py         CPU golden tests for forksafe.py.
│   ├── kernel/                  Transaction kernel (locate, journal, swap, jit, witness,
│   │                            findings, coordinator, session, receipts) + CPU toy lab.
│   ├── conformance/             Conformance test harness (cases c0–c10).
│   └── survey/                  Static live-editability survey of an engine's source tree.
└── agents/                    Agent personas, protocols, and role specifications:
    ├── orchestrator.md          Orchestrator prompt: live streamer and human interface.
    ├── developer.md             Developer prompt: edit, stage, verify, commit, and undo.
    ├── toolsmith_offline.md     Per-repo offline toolsmith: mine PRs/tests and bootstrap labs.
    ├── toolsmith_online.md      Per-repo online toolsmith: price checks and cut loop latency.
    ├── toolsmith_distill.md     Cross-repo distillation toolsmith: promote 2+ engine primitives.
    └── livelab_overview.md      Live lab overview: interface verbs, boundaries, function groups.
```
