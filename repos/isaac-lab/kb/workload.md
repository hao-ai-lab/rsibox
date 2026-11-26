# Workload F(x)

Resident `Isaac Lab 3.0` (`isaaclab`, `isaaclab_tasks`, `isaaclab_newton`) `Isaac-Cartpole-Direct` rollout on **NVIDIA B200** with the **`newton_mjwarp`** (Newton + MuJoCo-Warp + Warp + PyTorch CUDA) GPU physics backend across two `(num_envs, horizon_steps)` shapes (`(4096, 10)` and `(1024, 20)`, `seed=42`).

## Correctness Gate

1. **Repeatability (`c0`):** `reset("state")` (`sim.reset()` + zeroed episode buffers + restored Python/PyTorch/CUDA RNG) yields bit-exact rollouts (`max_abs_diff == 0.0`).
2. **Execution proof (`c1`, `c2`, `c10`):** AST witness fires on every executed step (`proven=True`); startup-only edits (`init_action_scale`) fail with `proven=False` (`zero_witness`).
3. **Fresh parity (`c3`) & rollback (`c4`, `c7`, `c9`):** Live outputs match `PackageOverlay` fresh subprocesses, and `uninstall()` restores exact base code and `S0` trajectories.
