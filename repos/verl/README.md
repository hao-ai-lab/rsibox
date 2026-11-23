# verl (`repos/verl/`)

Live-edit laboratory and offline replay probes for **verl** synchronous RL post-training
(`PPOTrainerSync`, GRPO / PPO over Ray), built on `core/livelab/kernel/` and
`core/livelab/ray_participants.py`.

Restarting a 4x GB200 verl GRPO job (`Qwen3-0.6B`, Megatron-Core trainer + 4 colocated vLLM
rollout replicas, 13 warm processes) takes **187-204 s** on fast parallel storage (and
**1,022-1,113 s** when the Python environment sits on shared NFS). Keeping the Ray actor tree
alive and installing edits at the synchronous step gap (`PPOTrainer.on_step_begin`) delivers a
live RL step verdict in **28-39 s** (~6x faster on fast storage, ~26x on NFS), while offline
replay probes check reward edits (`rescore`) in **0.5 ms** and advantage edits (`replay_adv`)
in **12 ms**.

## Layout

```text
repos/verl/
  README.md                  Overview, hardware receipts, and conformance quickstart
  docs/
    architecture.md          Ray process tree, step-gap barrier, S0 state reset, blast radius
  kb/
    costs.md                 4x GB200 startup anatomy, live step-gap timings, offline probes
    workload.md              Fixed GRPO GSM8K workload F(x) and correctness gate
    lessons.md               Ray @ray.remote class copy trap, file_namespace, coupled S0 sync
    survey_rules.json        Static coverage rules (rsibox-survey-rules/1: 71.8% functions live)
  examples/
    grpo-gsm8k/
      reward_gsm8k.py        The custom reward the GPU runs load; also the conformance harness's edit target
  livelab/
    README.md                Gate protocol, VerlLab, ToyVerlLab (CPU) and VerlConformanceLab (GPU) conformance
    adapter/
      __init__.py            Adds core/livelab to sys.path
      gate.py                GateMixin + PPOTrainerRsiboxSync step-gap Coordinator and the gate verbs
      conformance_lab.py     VerlConformanceLab: LabUnderTest over a real gated job (owns relaunch and fresh jobs)
      determinism.py         Per-sample vLLM seeds so a step after S0 reset resamples the same rollouts
      faults.py              Conformance fault injection inside one participant process
      hook.py                Ray worker_process_setup_hook import finder
      lab.py                 VerlLab (LabUnderTest) over RayActorParticipant + coupled S0
      settings.py            Live --set micro-batch and dynamic-batch validators
      state.py               Coupled trainer + rollout S0 weight/optimizer/RNG snapshot
      toy_lab.py             ToyVerlLab passing 11/11 conformance cases on CPU
      test_lab.py            Golden unit tests for coupled S0 reset, --set, setup hook, and Ray swap
      test_gate.py           Golden unit tests for actor discovery, FSDP sizes, dataloader S0, inline sources, seeds, harness patches
      toyverl/               Spawned multi-actor synchronous RL toy (rollout, trainer, reward)
  tools/
    cli.py                   Gate client, verl main_ppo launcher, and CLI (train, request, stop, conformance, rescore, replay-adv)
    rescore.py               0.5 ms CPU offline reward re-scorer over recorded rollouts
    replay_adv.py            12 ms CPU offline advantage re-evaluator over recorded steps
    test_tools.py            Golden unit tests for rescore, replay_adv, and dead-PID gate client
```

## Quick Verification (CPU, No GPU or Ray Required)

```sh
# 1. Adapter golden tests
PYTHONPATH=repos/verl/livelab \
  python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v

# 2. Full c0-c10 conformance suite on spawned multi-actor ToyVerlLab (11/11 pass)
cd core/livelab && PYTHONPATH=../../repos/verl/livelab \
  python3 -m conformance --lab adapter.toy_lab:ToyVerlLab
```

On GPUs the same eleven cases run against a real gated job (`livelab/README.md`, "Running Conformance"):
`python3 -m tools.cli conformance --gate <dir> --reward examples/grpo-gsm8k/reward_gsm8k.py --launch '<job command>'`.
