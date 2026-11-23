# Engine Onboarding Scaffold (`template/`)

Minimal starter layout and runnable `LabUnderTest` skeleton for onboarding a new engine under
`repos/<new-engine>/`.

The shared live-lab transaction kernel (`core/livelab/kernel/`), stdlib runtime kit
(`core/livelab/rsibox/`: `BaseLab`, `RecaptureCoordinator`, `RoleEndpoint`, IPC, process-tree,
import hooks, package overlays, fault injectors, and the `rsibox` CLI), Ray actor participant
helpers (`core/livelab/ray_participants.py`), pre-fork CUDA guards (`core/livelab/forksafe.py`),
static coverage survey (`core/livelab/survey/`), and conformance suite (`core/livelab/conformance/`)
live in `core/livelab/` and must **never** be copied into `repos/<new-engine>/`. Instead, each
engine's live lab (`repos/<new-engine>/livelab/`) imports them directly from `core/livelab/`.

## Target Layout for `repos/<new-engine>/`

```
repos/<new-engine>/
├── README.md                  Engine overview, scorecard summary, and replay highlights.
├── docs/
│   └── architecture.md        Process tree, state holders, and priced restart boundaries.
├── kb/
│   ├── costs.md               Measured cold/warm start and restart ladder costs.
│   ├── lessons.md             Engine-specific lessons and failure modes.
│   ├── survey_rules.json      Static coverage declarations (`rsibox-survey-rules/1`).
│   └── workload.md            Representative benchmark workload and invariants.
└── livelab/
    ├── README.md              How to start the lab and run conformance cases c0–c10.
    ├── adapter/
    │   ├── __init__.py        Puts `<repo>/core/livelab` on `sys.path`.
    │   ├── lab.py             `LabUnderTest` driver (`<Engine>Lab`) over the live engine.
    │   ├── toy_lab.py         CPU stand-in (`Toy<Engine>Lab`) exercising the real adapter.
    │   └── test_adapter.py    Golden unit tests (`<= 600` non-test LOC per boundary).
    └── services/
        └── zygote.py          Optional pre-import forkserver using `core/livelab/forksafe.py`.
```

## Step-by-Step Bootstrapping (`0/11` to `11/11` Conformance)

1. Copy `template/livelab/` into `repos/<new-engine>/livelab/` as your starting package skeleton.
2. Run the conformance suite out of the box (either via the unified `rsibox` CLI or directly):
   ```sh
   PYTHONPATH=core/livelab python3 -m rsibox conformance template
   # Or, once copied to repos/<new-engine>/livelab/, discovery picks it up automatically:
   PYTHONPATH=core/livelab python3 -m rsibox list
   PYTHONPATH=core/livelab python3 -m rsibox conformance <new-engine>
   ```
   `TemplateLab` instantiates and tears down cleanly, failing all `11/11` cases (`c0`–`c10`) with
   numbered `NotImplementedError` instructions. Subclassing `rsibox.BaseLab` implements `install`,
   `check`, `reset`, `uninstall`, `status`, `live_text` stack tracking, witness merging, and
   poison-on-fault handling out of the box (~60–100 LOC for a complete ToyLab):

   | Step | Method(s) to implement | `core/livelab/` primitive | Conformance cases unlocked |
   |---|---|---|---|
   | **1/6** | `_start_engine()` (`status`) | `rsibox.BaseLab`, `kernel.coordinator`, `kernel.session.Session` | Base state & fingerprint entry across `c0`–`c10` |
   | **2/6** | `_run_workload(workload)` (`check`) | `rsibox.BaseLab` + `coordinator.prove(txn)` | Repeatable check path for `c0` |
   | **3/6** | `_reset_state()`, `_stop_engine()` (`reset`) | `rsibox.BaseLab` + `rsibox.proc.stop_tree` | `c0_repeatable` (`1/11`), respawn half of `c8` |
   | **4/6** | `patches()` (`install` & `uninstall` inherited from `BaseLab`) | `rsibox.BaseLab`, `rsibox.edit_once` | `c1_noop`, `c2_visible`, `c4_uninstall`, `c6_refusal`, `c7_chain`, `c9_soak` (`7/11`) |
   | **5/6** | `fresh_check(patch, workload)` | `rsibox.PackageOverlay` + `rsibox.run_fresh_subprocess` | `c3_fresh_equivalence`, `c10_not_proven` (`9/11`) |
   | **6/6** | `inject(fault, **kwargs)` | `rsibox.inject_participant_fault` (`fail_apply`, `fail_commit`, `tamper`) + engine crash/hang/lost hooks | `c5_failed_install`, `c8_crash_and_hang` (`11/11`) |

3. Follow `core/agents/toolsmith_offline.md` for hotspot mining, offline fail-closed checks, and
   `kb/` documentation.

## Onboarding LOC Scorecard Template

Copy this table into `repos/<new-engine>/README.md` (or PR description) when onboarding a new
engine to verify that engine-specific code stays within the `<= 600` non-test Python LOC boundary
budget and reuses `core/livelab/` rather than duplicating runtime machinery:

| Layer / File | Owner | Non-Test Python LOC | Test Python LOC | Reused `core/livelab/` Primitives |
|---|---|---:|---:|---|
| `livelab/adapter/__init__.py` | `repos/<engine>` | `0`–`25` | `0` | `RSIBOX_LIVELAB` path bootstrap |
| `livelab/adapter/<gap_or_gate>.py` | `repos/<engine>` | `<fill>` | `<fill>` | `Coordinator` / `StepGapCoordinator`, `LocalParticipant` / `RayActorParticipant` |
| `livelab/adapter/<state_reset>.py` | `repos/<engine>` | `<fill>` | `<fill>` | Coupled weight/cache snapshot & epoch drain |
| `livelab/adapter/lab.py` | `repos/<engine>` | `<fill>` | `<fill>` | `Session`, `ReceiptWriter`, `LabUnderTest` (`c0`–`c10`) |
| `livelab/adapter/toy_lab.py` | `repos/<engine>` | `<fill>` | `<fill>` | CPU stand-in driving the real adapter (`11/11`) |
| **Adapter boundary total** | **`repos/<engine>/livelab/adapter/`** | **`<= 600`** | **`<fill>`** | **`0` cross-`repos/` imports; `0` duplicated kernel files** |
| Shared runtime (reused, 0 new LOC) | `core/livelab/` | `0` (shared) | `0` (shared) | `kernel/*`, `ray_participants.py`, `forksafe.py`, `conformance/*`, `survey/*` |
