# Measured Costs (`Isaac Lab 3.0`, `newton_mjwarp` & CPU Toy)

## Real GPU (`1× NVIDIA B200`, `Isaac-Cartpole-Direct`, `physics=newton_mjwarp`)

Runtime: Isaac Lab 3.0 (`isaaclab`, `isaaclab_tasks`, `isaaclab_newton`) with Newton + MuJoCo-Warp + Warp + PyTorch CUDA (not `isaacsim` PhysX).

| Shape | Cold Boot + 1st Check | Live Install + Check | Speedup | Undo (`S0`) | Fresh Parity |
|---|---:|---:|---:|---:|---|
| `num_envs=4096, h=10` | `9.6616 s` (`8.7428 s` + `0.9188 s`) | **`0.3968 s`** (`0.1570 s` + `0.2398 s`) | **`24.3×`** | `0.1478 s` | `max_abs_diff=0.0` |
| `num_envs=1024, h=20` | `6.7864 s` (`6.0697 s` + `0.7167 s`) | **`0.3017 s`** (`0.0834 s` + `0.2183 s`) | **`22.5×`** | `0.0799 s` | `max_abs_diff=0.0` |

- **B200 Conformance (`4096` envs, `h=10`):** **11/11 PASS** in `154.493 s` (`c0` `3.665 s`, `c1` `3.216 s`, `c2` `3.276 s`, `c3` `33.219 s`, `c4` `3.808 s`, `c5` `4.143 s`, `c6` `5.293 s`, `c7` `5.508 s`, `c8` `60.923 s`, `c9` `18.573 s` [install median `0.1513 s`, undo `0.1445 s`], `c10` `12.869 s`).

## CPU Toy Conformance (`ToyIsaacLab`, `num_envs=64`, `h=4`)

- **11/11 PASS** in `3.52 s` (`c9_soak` install median `0.0056 s`, uninstall median `0.0009 s`; `5/5` unit tests pass with `-W error::ResourceWarning`).
