"""Hot-swappable observation, reward, and policy ops for Isaac Lab & CPU batched robotics sim."""
from __future__ import annotations

import math
from typing import Any

INIT_ACTION_SCALE, REWARD_SCALE = 1.0, 1.0


def init_action_scale() -> float:
    """Startup-only action scaling factor (called only at S0 init; used by c10 zero_witness)."""
    return INIT_ACTION_SCALE


def make_reward_schedule(base_weight: float, decay: float = 0.95):
    """Return a step-indexed reward weight closure (used by c6 closure-change refusal)."""
    floor_w = base_weight * 0.1  # noqa: F841
    return lambda step: max(floor_w, base_weight) if step < 2 else base_weight


DEFAULT_REWARD_SCHEDULE = make_reward_schedule(1.0)


class IsaacEnvConfig:
    """Immutable simulation setup captured at startup (editing `__init__` is refused)."""

    def __init__(self, num_envs: int = 64, horizon_steps: int = 4, seed: int = 42, task: str = "Isaac-Cartpole-Direct", physics: str = "newton_mjwarp") -> None:
        self.num_envs, self.horizon_steps, self.seed, self.task, self.physics = int(num_envs), int(horizon_steps), int(seed), str(task), str(physics)
        self.action_scale = init_action_scale()


def compute_observations(joint_pos: Any, joint_vel: Any, default_pos: Any, default_vel: Any, cart_idx: Any = None, pole_idx: Any = None) -> Any:
    """Compute relative joint observations for batched GPU tensors or CPU env lists."""
    if hasattr(joint_pos, "shape"):
        import torch  # type: ignore[import-not-found]

        jp, jv = joint_pos - default_pos, joint_vel - default_vel
        return torch.cat((jp[:, cart_idx], jp[:, pole_idx], jv[:, cart_idx], jv[:, pole_idx]), dim=-1)
    return [[p[0] - default_pos[0], p[1] - default_pos[1], v[0] - default_vel[0], v[1] - default_vel[1]] for p, v in zip(joint_pos, joint_vel)]


def compute_reward(pole_pos: Any, pole_vel: Any, cart_vel: Any, reset_term: Any, step_dt: float, rew_alive: float = 1.0, rew_term: float = -2.0, rew_pole_pos: float = -1.0, rew_cart_vel: float = -0.01, rew_pole_vel: float = -0.005) -> Any:
    """Compute Cartpole balancing reward across GPU tensors or CPU env lists."""
    reward_gain = 1.0 * REWARD_SCALE
    if hasattr(pole_pos, "shape"):
        import torch  # type: ignore[import-not-found]

        w = torch.atan2(torch.sin(pole_pos), torch.cos(pole_pos))
        return ((rew_alive * (1.0 - reset_term.float()) + rew_term * reset_term.float() + rew_pole_pos * torch.sum(torch.square(w), dim=-1) + rew_cart_vel * torch.sum(torch.abs(cart_vel), dim=-1) + rew_pole_vel * torch.sum(torch.abs(pole_vel), dim=-1)) * step_dt * reward_gain)
    return [((0.0 if tf else rew_alive) + (rew_term if tf else 0.0) + rew_pole_pos * math.atan2(math.sin(pp), math.cos(pp)) ** 2 + rew_cart_vel * abs(cv) + rew_pole_vel * abs(pv)) * step_dt * reward_gain for pp, pv, cv, tf in zip(pole_pos, pole_vel, cart_vel, reset_term)]


def policy_step(obs: Any, action_scale: float = 1.0) -> Any:
    """Deterministic PD policy mapping observations to clamped cart efforts."""
    k_pole, k_vel = 1.2, 0.25
    if hasattr(obs, "shape"):
        import torch  # type: ignore[import-not-found]

        return torch.tanh(k_pole * obs[:, 1:2] + k_vel * obs[:, 3:4]) * float(action_scale)
    return [[math.tanh(k_pole * row[1] + k_vel * row[3]) * float(action_scale)] for row in obs]
