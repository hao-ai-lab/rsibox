---
name: rsibox
description: Guide for navigating the RSIBox codebase, developing and probing on a live sandbox via the rsibox CLI, distinguishing normal fail-closed boundary refusals from real framework bugs, and escalating across restart boundaries.
---

# RSIBox Codebase & Live Sandbox Guide

RSIBox keeps an ML engine's startup state, model weights, and compiled state warm in memory so code edits swap in milliseconds to seconds instead of paying a cold restart.

## 1. Four-Layer Architecture

```
  [ Agent / Conformance / SandboxPool ]  <-- INTERFACE v0 five verbs (install, check, reset, uninstall, status)
                   │
  Layer 1: Control Plane          core/livelab/rsibox/  (BaseLab, Box HTTP daemon, CLI, launchers, SandboxPool)
                   │
  Layer 2: Hot-Reload Kernel      core/livelab/kernel/  (pure-Python stdlib: swap, jit, reinit, journal, witness, Session)
                   │
  Layer 3: Warm & Process Substrate  core/livelab/rsibox/warm/ + core/livelab/forksafe.py
                   │                 (WeightStore, RecoveryDriver, Dynamo GMS/Snapshot, pre-fork CUDA guards)
  Layer 4: Thin Engine Adapters   repos/<engine>/livelab/  (engine-specific hooks + services; zero cross-engine imports)
```

- **Layer 1 — Control Plane ([`core/livelab/rsibox/`](../../../core/livelab/rsibox/README.md)):** Engine-agnostic `BaseLab`, `Box` HTTP daemon (`POST /v1/{install,check,reset,uninstall,close}`, `GET /v1/{status,source,info,events,receipts}`), `rsibox` CLI, container/Slurm/gVisor launchers, `doctor.py`, `SandboxPool` (`pool.py`), and `PoolAutoScaler` (`autoscaler.py`).
- **Layer 2 — Hot-Reload Kernel ([`core/livelab/kernel/`](../../../core/livelab/kernel/README.md)):** Pure-Python stdlib transaction kernel importing zero ML frameworks:
  - AST target analysis & `R0` bytecode swap (`swap.py`), `@triton.jit` in-place rebuild & caller invalidation (`jit.py`), literal constructor reinit (`reinit.py`), GC heap locator (`locate.py`).
  - Reversible undo log (`journal.py`), two-ledger `live` vs. `capture` execution proof (`witness.py`), policy & findings (`findings.py`).
  - Two-phase step-gap coordinator (`coordinator.py`), `Session` state machine (`session.py`), deterministic SHA-256 identity (`identity.py`), and `ReceiptWriter` (`receipt.py`).
- **Layer 3 — Warm Recovery & Process Substrate ([`core/livelab/rsibox/warm/`](../../../core/livelab/rsibox/warm/README.md) & [`core/livelab/forksafe.py`](../../../core/livelab/forksafe.py)):** Separates sub-second live edits from weight residency and crash recovery via `WeightStore` (`DiskStore`, `DynamoGmsStore`), `RecoveryDriver` (`DefaultRespawnDriver`, `StandbyShadowDriver`, `DynamoK8sSnapshotDriver`), and pre-fork NVML, Triton, and CuTe DSL CUDA guards (`forksafe.py`).
- **Layer 4 — Thin Engine Adapters ([`repos/`](../../../repos/README.md)):** Per-engine `LabUnderTest` adapters (`repos/<engine>/livelab/adapter/` or `rsibox_megatron/`) and optional restart/weight services (`services/`). Engine code must never import from another `repos/<other>/`.
- **Cross-Cutting Verification & Gates:**
  - Formal TLA+ state-machine specs & spec-to-code conformance tests in [`core/livelab/formal/`](../../../core/livelab/formal/) and [`core/livelab/kernel/formal/`](../../../core/livelab/kernel/formal/README.md).
  - **The sandbox verification suite** in [`core/livelab/conformance/`](../../../core/livelab/conformance/README.md) (`c0`–`c10` cases driving `LabUnderTest`).
  - Static AST live-editability survey in [`core/livelab/survey/`](../../../core/livelab/survey/README.md).
  - Deterministic workflow, LOC, leak, and doc gates in [`.agents/plugins/rstack/`](../../../.agents/plugins/rstack/README.md).

### Fixed Interface Verbs ([`core/livelab/INTERFACE.md`](../../../core/livelab/INTERFACE.md))

| Verb | Purpose | Key Reply Fields |
|---|---|---|
| `install(patch)` | Two-phase atomic stage + bind across all ranks (`files`, optional `accept`, `undo_resets`; plus optional `base` enforced by Box HTTP). | `ok`, `id`, `outcome` (`committed \| aborted \| rolled_back \| refused \| in-doubt`), `mechanisms`, `findings`, `cost_s` |
| `check(workload)` | Run probe workload inside the warm engine. | `ok`, `output`, `proven` (`True \| False \| None`), `witness`, `findings`, `cost_s` |
| `reset(kind)` | `kind="state"` clears volatile caches/RNG/buffers; `kind="respawn"` restarts workers from warm caches and clears poison. | `ok`, `reset`, `cost_s` |
| `uninstall(id)` | Journaled revert of `id` (default newest) and any later stacked transactions. | `ok`, `undone`, `state` (`base \| edited \| poisoned`), `cost_s` |
| `status()` | Report live session state, active txns, code identity digest, and poison/recovery flags. | `ok`, `state`, `txns`, `fingerprint`, `poison` (`Session`/`BaseLab` also include `incarnation`, `recovery_required`, `identity`) |

## 2. Complete Repository Directory Map

| Path | Contents & Ownership |
|---|---|
| [`core/livelab/kernel/`](../../../core/livelab/kernel/README.md) | Pure-Python transaction kernel (`swap`, `jit`, `reinit`, `locate`, `journal`, `witness`, `findings`, `coordinator`, `session`, `receipt`, `identity`, `toy_lab.py`) and `formal/` TLA+ protocol specs (`Transaction.tla`, `Participant.tla`, `Incarnation.tla`, `SequentialTx.tla`). |
| [`core/livelab/rsibox/`](../../../core/livelab/rsibox/README.md) | Stdlib runtime kit (`BaseLab`, `RecaptureCoordinator`, `RoleEndpoint`), HTTP `Box` server/client, `rsibox` CLI, launchers (`local`, `docker`, `gvisor`, `slurm`), `doctor.py`, `pool.py`, `autoscaler.py`, and `promote.py`. |
| [`core/livelab/rsibox/warm/`](../../../core/livelab/rsibox/warm/README.md) | Pluggable `WeightStore` (`dynamo_gms.py`, `weight_store.py`) and `RecoveryDriver` (`recovery.py`, `snapshot_contract.py`, `types.py`). |
| [`core/livelab/conformance/`](../../../core/livelab/conformance/README.md) | **The sandbox verification suite** (`c0`–`c10` cases in `cases.py`) and `LabUnderTest` contract (`lab.py`). |
| [`core/livelab/formal/`](../../../core/livelab/formal/) | TLA+ models (`autoscaler/`, `coordinator/`, `pool/`, `shadow/`, `swap/`) and `test_tla_conformance.py`. |
| [`core/livelab/survey/`](../../../core/livelab/survey/README.md) | Static AST live-editability classifier (`scan.py`, `rules.py`, `history.py`) against upstream engine source trees. |
| [`core/livelab/forksafe.py`](../../../core/livelab/forksafe.py) | Pre-fork CUDA/NVML/Triton/CuTe DSL guards and bad-fork poison checks for engine zygote servers. |
| [`repos/vllm/`](../../../repos/vllm/livelab/README.md) | vLLM live lab (`livelab/adapter/`, `livelab/services/`, `docs/`, `kb/`, `examples/`, `tools/`). |
| [`repos/sglang/`](../../../repos/sglang/livelab/README.md) | SGLang live lab (`livelab/adapter/`, `livelab/services/`, `docs/`, `kb/`). |
| [`repos/trtllm/`](../../../repos/trtllm/livelab/README.md) | TensorRT-LLM PyTorch live lab (`livelab/adapter/`, `docs/`, `kb/`). |
| [`repos/transformers/`](../../../repos/transformers/) | HuggingFace Transformers live lab (`livelab/adapter/`, `docs/`, `kb/`). |
| [`repos/megatron/`](../../../repos/megatron/livelab/README.md) | Megatron-LM live lab (`livelab/rsibox_megatron/`, `kb/`, `examples/`, `patches/`, `tools/`). |
| [`repos/fastvideo/`](../../../repos/fastvideo/livelab/README.md) | FastVideo live lab (`livelab/adapter/`, `livelab/services/`, `docs/`, `kb/`, `examples/`, `patches/`, `tools/`). |
| [`repos/verl/`](../../../repos/verl/livelab/README.md) | veRL Ray/FSDP live lab (`livelab/adapter/`, `docs/`, `kb/`, `examples/`, `tools/`). |
| [`repos/miles/`](../../../repos/miles/livelab/README.md) | miles live lab (`livelab/adapter/`, `docs/`, `kb/`). |
| [`repos/torchtitan/`](../../../repos/torchtitan/) | TorchTitan FSDP2 live lab (`livelab/adapter/`, `docs/`, `kb/`, `examples/`). |
| [`repos/nemo-rl/`](../../../repos/nemo-rl/) | NeMo-RL multi-participant live lab (`livelab/adapter/`, `docs/`, `kb/`). |
| [`docs/`](../../../docs/README.md) | Framework design docs (`concepts.md`, `engines.md`, `mechanisms.md`, `kernels-and-graphs.md`, `lessons.md`, `agent-teams-and-contention.md`, `process/`). |
| [`site/`](../../../site/README.md) | Static landing page (`landing/`), verified JSON facts (`facts/`), and `build.py`. |
| [`scripts/`](../../../scripts/) | Fact and packaging validators (`verify_facts.py`, `plot_rsibox_dynamo_4bar.py`, `test_*.py`). |
| [`template/`](../../../template/README.md) | Onboarding scaffold (`template/livelab/`) for adding a new engine under `repos/<new-engine>/`. |
| [`.agents/plugins/rstack/`](../../../.agents/plugins/rstack/README.md) | Deterministic workflow gates (`claim_board.py`, `preflight.py`, `pre_push.py`, `pr_gate.py`, `docs_lint.py`, `facts_sync.py`, `trajectory_audit.py`), guardrails, and review skills. |
| [`.github/`](../../../.github/) | CI workflows (`workflows/`), automated PR review (`review/`), GPU Modal runner (`modal-ci/`), and leak/CI helpers (`scripts/`). |

## 3. Developing & Probing on a Live Sandbox

Run commands from `core/livelab` with `python3 -m rsibox ...` (or `rsibox ...` when installed).

### Preflight, Discovery & CPU Tests

```sh
python3 -m rsibox doctor                            # lazy-import lab probe + environment preflight across labs
python3 -m rsibox doctor --engine <engine>          # engine-specific environment preflight only
python3 -m rsibox list                              # discover toy, template, and repos/<engine>/ labs
python3 -m rsibox conformance <engine> [--real]     # run the sandbox verification suite (--only CASE to filter)
python3 -m unittest discover -s kernel -t . -v      # kernel golden suite on CPU (stdlib only)
python3 -m unittest discover -s rsibox -t . -v      # runtime kit, BaseLab, HTTP daemon, pool, CLI tests
```

### Live Edit–Check–Undo Loop

```sh
python3 -m rsibox up --launcher local --lab <engine> [--real] --state /tmp/box1
export RSIBOX=/tmp/box1/box.json                    # or RSIBOX_URL + RSIBOX_TOKEN for remote boxes

python3 -m rsibox wait                              # block until engine is warm; prints boot_s
python3 -m rsibox source <module> > mod.py          # fetch live module source text (add --json for hash)
# ... edit mod.py ...
python3 -m rsibox install --file <module>=mod.py    # atomic swap across ranks (exit 2 if refused/aborted)
python3 -m rsibox check                             # run workload; check output, proven, witness, cost_s
python3 -m rsibox status                            # verify state (base|edited|poisoned), poison (plus incarnation, recovery_required on Session/BaseLab)
python3 -m rsibox receipts                          # inspect read-only JSONL txn/proof/undo receipts
python3 -m rsibox events                            # inspect per-call outcomes and wall_s timings
python3 -m rsibox uninstall                         # journaled revert to base
python3 -m rsibox reset --kind state                # clear volatile caches/RNG (or --kind respawn after poison)
python3 -m rsibox down --state /tmp/box1            # close sandbox and stop daemon
```

### Probing Checklist

1. **Read engine priors first:** Check `repos/<engine>/kb/` (e.g., `costs.md`, and where present `workload.md` / `lessons.md`) for probe costs, reset rules, and unsupported paths.
2. **Send full module text (or `--diff`):** `install` accepts full module texts (`--file MODULE=PATH` or JSON `{"files": {...}}`) or unified diffs (`--diff`), computes changed AST targets automatically, and never requires manual line numbers.
3. **Prevent concurrent clobber:** Pass `"base": {"<module>": "<hash>"}` in `--patch` JSON when multiple writers share a box; `install` fails closed with `base_mismatch` if live code changed.
4. **Verify execution proof:** A `check` result counts only when `ok: true`, `proven: true`, and `witness` shows non-zero live calls on every participant (`proven: null` means the lab is at `base`; `proven: false` with `zero_witness` means the edited path never ran).
5. **Verify sandbox health & pool traces:** Confirm `state` (`base` or `edited`, never `poisoned`), `poison: []`, and (on `Session` / `BaseLab`) `recovery_required: false` in `status`. When sharing slots through `SandboxPool`, validate the SHA-256 hash-chained trace via `python3 -m rsibox verify-trace <trace.jsonl>`.

### Function Groups & What `install` Refreshes

| Group | What Changed | How `install` Handles It |
|---|---|---|
| `OP` | Per-call Python function or `@triton.jit` body | In-place bytecode swap (`R0`) or Triton rebuild + caller cache clear |
| `GRAPH` | Code inside CUDA graphs or `torch.compile` | Swap + selective `R1` recapture of affected graphs (`RecaptureCoordinator`) |
| `STATE` | Objects built once at startup (buffers, runners) | Rebuild affected objects in place or refuse with `route: "T1"` (component/constructor restart) |
| `RESOURCE` | Load-time memory layout and allocation | Allocate new, copy, switch, free (or refuse when fixed at boot) |
| `RANKS` | Multi-GPU / multi-participant collective paths | Two-phase commit across all ranks or none (`StepGapCoordinator`) |
| `NATIVE` | C++ / CUDA `.cu` kernel source | Versioned shared-library compile, rebind, and graph recapture |

## 4. Normal Fail-Closed Refusal vs. Real Bug Triage

A live sandbox fails closed by design (`ok: false`, `outcome: "refused"` or `"aborted"`, CLI exit `2`). Never bypass a boundary refusal or edit receipts by hand. Read `findings` (`check`, `observed`, `expected`, `why`, `actions`, `route`, `overridable`) to separate expected boundary refusals from real framework bugs:

| Symptom / Signal | Expected Fail-Closed Refusal (Normal) | Real Framework Bug (Escalate & Fix) |
|---|---|---|
| **Weight, allocator, or load-time layout edit** | Refused with `new_fields`, `constructor_edit` (`route: "T1"` for component/constructor restart), `outside_changes`, or `removed_definition`. | Mutates module text while live objects/weights keep old layout, or crashes mid-step without refusing. |
| **Static CUDA graph or Triton signature change** | Refused with `captured_target` (blocking unless accepted or `R1` recaptured; note `captured_partly` is advisory and commits for prefill/mixed steps) or `jit_target` (signature/decorator changed). | Swaps Python wrapper only, reports `proven: true` from graph-capture hits (`capture` vs. `live` ledger confusion), or leaves stale device code active. |
| **Startup logic, launch flags, or env vars** | Refused as outside live-swappable definitions; requires warm respawn / restart boundary. | Reports `committed` even though the edited startup path cannot execute until process restart. |
| **Duplicate `txn` ID, stale base, or no-op** | Refused before mutating state: `patch_invalid` (`duplicate transaction id`), `base_mismatch`, or `empty_change`. | Re-applies a prior `txn` ID, silently overwrites concurrent edits without `base_mismatch`, or commits a no-op patch as `ok: true`. |
| **Post-commit reset & `undo_resets`** | `RecaptureCoordinator(..., post_commit_reset="state")` runs `reset("state")` after commit (without checking its return value during `install`); direct `reset("state")` calls or `undo_resets` during `Coordinator.undo` surface `reset_failed` when state cannot be cleared cleanly. | Stale KV/RNG/buffer state leaks into subsequent `check` calls, breaking repeatability (`c0_repeatable`), or `reset_failed` is swallowed during undo. |
| **Fault, poison, or failed rollback** | Refuses `install` (`session_poisoned`) and `uninstall` (`recovery_required: true`), blocking edits until `reset --kind respawn`. | **Missing witness execution** on a pass, **split-rank commit** across phase 2, **silent swallow** on partial rollback (failing to latch `recovery_required`), or **stale incarnation acceptance** from a superseded epoch. |

### Restart-Boundary Escalation

When `install` refuses an edit, step up the boundary ladder using the cheapest mechanism named in `findings` (`route` / `actions`; see [`docs/mechanisms.md`](../../../docs/mechanisms.md)):

1. **In-place Python / Triton (`R0`):** Function bodies, `@triton.jit` kernel bodies, and literal `constructor_reinit` (`swap.py`, `jit.py`, `reinit.py`).
2. **In-place CUDA graph / native (`R1`):** Selective graph recapture (`RecaptureCoordinator`) or versioned `.cu` side-library reload (`native.py`).
3. **In-process rebuild / reshard:** Re-import edited modules and rebuild live holders (`rebuild.py`), or CUDA-IPC TP/EP reshard.
4. **Warm respawn / snapshot:** `forksafe` zygote + IPC weight holder, Dynamo GMS RO or Standby Shadow promotion (`rsibox reset --kind respawn`), or CRIU + `cuda-checkpoint` snapshot restore.

Before pushing any branch or opening a PR, run the deterministic `rstack` gates (`preflight.py`, `pre_push.py --strict`, `docs_lint.py`, and `scripts/verify_facts.py --strict`).
