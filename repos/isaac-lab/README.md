# Isaac Lab (`isaac-lab` Robotics Simulation + RL)

RSIBox live-lab adapter for Isaac Lab (`Isaac-Cartpole-Direct`) with resident multi-env GPU physics state, sub-second live swaps of `compute_reward` / `compute_observations` / `policy_step` / `summarize_rollout`, and seeded `S0` environment rollback.

| Path | Holds |
|---|---|
| [kb/costs.md](kb/costs.md) | Measured CPU (`ToyIsaacLab`) and NVIDIA B200 (`newton_mjwarp`) timings |
| [kb/workload.md](kb/workload.md) | Deterministic `4096`-env / `1024`-env rollout workload and correctness gate |
| `livelab/` | `livelab.json` and `adapter/{lab,toy_lab,env_ops,worker,test_isaac_lab}.py` |
