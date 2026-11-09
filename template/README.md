# Engine Onboarding Scaffold (`template/`)

Minimal starter layout for onboarding a new engine under `repos/<new-engine>/`.

The shared live-lab transaction kernel (`core/livelab/kernel/`), pre-fork CUDA guards
(`core/livelab/forksafe.py`), and conformance test suite (`core/livelab/conformance/`) live in
`core/livelab/` and must **never** be copied into `repos/<new-engine>/`. Instead, each engine's
live lab (`repos/<new-engine>/livelab/`) imports them directly from `core/livelab`.

## Target Layout for `repos/<new-engine>/`

```
repos/<new-engine>/
├── README.md                  Engine overview, hardware receipts, and replay summary.
├── docs/
│   └── architecture.md        Process tree, state holders, and priced restart boundaries.
├── kb/
│   ├── lessons.md             Engine-specific lessons and failure modes.
│   ├── prices.md              Measured cold/warm start and restart ladder costs.
│   └── workload.md            Representative benchmark workload and invariants.
└── livelab/
    ├── README.md              How to start the lab and run conformance cases c0–c10.
    ├── adapter/
    │   └── __init__.py        Puts <repo>/core/livelab on sys.path (see template/livelab/adapter/).
    └── services/
        └── zygote.py          Optional pre-import forkserver using core/livelab/forksafe.py.
```

## Bootstrapping a New Engine

1. Copy `template/livelab/` into `repos/<new-engine>/livelab/` as your starting package skeleton.
2. Implement the step-gap participant hook and host coordinator wiring using `from kernel import ...`
   and `import forksafe` (resolved from `core/livelab`).
3. Verify the adapter against the conformance suite from `core/livelab`:
   ```sh
   cd core/livelab && PYTHONPATH=<repo>/repos/<new-engine>/livelab \
       python3 -m conformance --lab adapter.toy_lab:ToyEngineLab
   ```
4. Follow `core/agents/toolsmith_offline.md` for full onboarding steps.
