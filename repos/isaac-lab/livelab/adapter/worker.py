"""Step-gap worker subprocess for Isaac Lab GPU (`newton_mjwarp`) and CPU batched simulation."""
from __future__ import annotations

import contextlib, hashlib, importlib, importlib.util, json, linecache, math, os, queue, random, sys, time
from typing import Any

from adapter import env_ops
from kernel import coordinator, findings
from rsibox import harness, ipc
from rsibox.lab import ENDPOINT_METHODS

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)


def validate_positive_int(value: Any, name: str) -> int:
    """Validate strict positive integer configuration fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1 (got {value!r})")
    return value


def require_isaac_lab_deps(device: str = "cuda:0") -> dict[str, str]:
    """Verify Isaac Lab and CUDA runtime availability lazily; fail closed when absent."""
    missing = [p for p in ("torch", "isaaclab", "isaaclab_tasks", "gymnasium") if importlib.util.find_spec(p) is None]
    if missing:
        raise RuntimeError(f"IsaacLabLiveLab real mode missing required package(s): {', '.join(missing)}")
    import torch  # type: ignore[import-not-found]

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Isaac Lab real mode but torch.cuda.is_available() is False")
    return {"torch": getattr(torch, "__version__", "unknown"), "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"}


def summarize_rollout(step_rewards: list[float], obs_checksums: list[float]) -> dict[str, Any]:
    """Aggregate per-step rollout rewards and observation checksums into a deterministic digest."""
    summary_scale = 1.0
    scaled, obs_sums = [round(r * summary_scale, 8) for r in step_rewards], [round(o, 8) for o in obs_checksums]
    return {"step_rewards": scaled, "obs_checksums": obs_sums, "reward_total": round(sum(scaled), 8), "digest": hashlib.sha256(json.dumps([scaled, obs_sums]).encode()).hexdigest()[:16]}


def _sync_linecache(part: Any, modules: Any) -> None:
    for mod in modules:
        with contextlib.suppress(Exception):
            if fpath := (getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)):
                if isinstance(src := part._live_source(mod), str):
                    linecache.cache[fpath] = (len(src), None, src.splitlines(True), fpath)
                else:
                    linecache.cache.pop(fpath, None)


class IsaacEnvWorker:
    """Holds resident GPU (`isaaclab` + `newton_mjwarp`) or CPU batched environment state with `S0` reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.mode = str(config.get("mode", "cpu"))
        self.num_envs = validate_positive_int(int(config.get("num_envs", 64)), "num_envs")
        self.horizon_steps = validate_positive_int(int(config.get("horizon_steps", 4)), "horizon_steps")
        self.seed, self.task, self.physics = int(config.get("seed", 42)), str(config.get("task", "Isaac-Cartpole-Direct")), str(config.get("physics", "newton_mjwarp"))
        self.device = str(config.get("device", "cpu" if self.mode == "cpu" else "cuda:0"))
        self.cfg = env_ops.IsaacEnvConfig(self.num_envs, self.horizon_steps, self.seed, self.task, self.physics)
        self._stack, self.env, self.step_count, self._at_s0 = contextlib.ExitStack(), None, 0, False
        self._obs: Any = None
        self._rng_py: Any = None
        self._rng_torch: Any = None
        self._rng_cuda: Any = None
        try:
            self._init_backend()
            self._at_s0 = True
        except BaseException:
            self.close()
            raise

    def _init_backend(self) -> None:
        if self.mode == "cpu":
            rng = random.Random(self.seed)
            self._s0_pos = [[rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2)] for _ in range(self.num_envs)]
            self._s0_vel = [[rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)] for _ in range(self.num_envs)]
            self._joint_pos, self._joint_vel, self._term = [list(r) for r in self._s0_pos], [list(r) for r in self._s0_vel], [False] * self.num_envs
            random.seed(self.seed)
            self._rng_py = random.getstate()
            self._obs = env_ops.compute_observations(self._joint_pos, self._joint_vel, [0.0, 0.0], [0.0, 0.0])
            return
        if self.mode != "isaaclab":
            raise ValueError(f"unsupported mode {self.mode!r} (expected 'cpu' or 'isaaclab')")
        require_isaac_lab_deps(self.device)
        import argparse, gymnasium as gym, isaaclab_tasks, torch  # type: ignore[import-not-found] # noqa: F401
        from isaaclab.app import add_launcher_args, launch_simulation  # type: ignore[import-not-found]
        from isaaclab_tasks.utils import resolve_task_config  # type: ignore[import-not-found]

        p = argparse.ArgumentParser()
        add_launcher_args(p)
        cfg, _ = resolve_task_config(self.task, None, overrides=[f"physics={self.physics}"])
        cfg.scene.num_envs, cfg.sim.device = self.num_envs, self.device
        self._stack.enter_context(launch_simulation(cfg, p.parse_args([])))
        self.env = gym.make(self.task, cfg=cfg)
        u = self.env.unwrapped
        u._get_observations = lambda: {"policy": env_ops.compute_observations(u.joint_pos, u.joint_vel, u.cartpole.data.default_joint_pos.torch, u.cartpole.data.default_joint_vel.torch, u._cart_dof_idx, u._pole_dof_idx)}
        u._get_rewards = lambda: env_ops.compute_reward(u.joint_pos[:, u._pole_dof_idx], u.joint_vel[:, u._pole_dof_idx], u.joint_vel[:, u._cart_dof_idx], u.reset_terminated, u.step_dt, u.cfg.rew_scale_alive, u.cfg.rew_scale_terminated, u.cfg.rew_scale_pole_pos, u.cfg.rew_scale_cart_vel, u.cfg.rew_scale_pole_vel)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed(self.seed)
        self._rng_py, self._rng_torch = random.getstate(), torch.get_rng_state()
        self._rng_cuda = torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None
        self.reset_state()

    def reset_state(self) -> list[str]:
        """Restore exact `S0` physics state, episode buffers, and RNG without destroying sim context."""
        self.step_count = 0
        random.setstate(self._rng_py)
        if self.mode == "cpu":
            self._joint_pos, self._joint_vel, self._term = [list(r) for r in self._s0_pos], [list(r) for r in self._s0_vel], [False] * self.num_envs
            self._obs = env_ops.compute_observations(self._joint_pos, self._joint_vel, [0.0, 0.0], [0.0, 0.0])
            self._at_s0 = True
            return ["sim_s0", "joint_state", "rng", "step_counter"]
        import torch  # type: ignore[import-not-found]

        torch.set_rng_state(self._rng_torch)
        if self._rng_cuda is not None:
            torch.cuda.set_rng_state(self._rng_cuda, self.device)
        assert self.env is not None
        u = self.env.unwrapped
        u.sim.reset()
        for buf in (u.episode_length_buf, u.reset_terminated, u.reset_time_outs, u.reset_buf):
            buf.zero_()
        self._obs, self._at_s0 = self.env.reset(seed=self.seed)[0]["policy"], True
        return ["sim_s0", "phys_buffers", "rng", *(["cuda_cache"] if self._rng_cuda is not None else []), "step_counter"]

    def run_rollout(self, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        wl = workload or {}
        if (n_envs := validate_positive_int(int(wl.get("num_envs", self.num_envs)), "num_envs")) != self.num_envs:
            raise ValueError(f"num_envs mismatch ({n_envs} != {self.num_envs}); respawn to resize resident scene")
        steps = validate_positive_int(int(wl.get("horizon_steps", self.horizon_steps)), "horizon_steps")
        if not bool(wl.get("continue_state", False)) and (self.step_count != 0 or not self._at_s0):
            self.reset_state()
        self._at_s0, step_rewards, obs_sums, dt = False, [], [], 1.0 / 60.0
        for _ in range(steps):
            acts = env_ops.policy_step(self._obs, self.cfg.action_scale)
            if self.mode == "cpu":
                for i, act in enumerate(acts):
                    f = max(-1.0, min(1.0, float(act[0]))) * 10.0
                    x, th, xd, thd = self._joint_pos[i][0], self._joint_pos[i][1], self._joint_vel[i][0], self._joint_vel[i][1]
                    xdd = (f + 0.1 * thd * thd * math.sin(th) - 9.8 * math.sin(th) * math.cos(th)) / 1.1
                    thdd = (9.8 * math.sin(th) - math.cos(th) * xdd) / 0.5
                    self._joint_pos[i], self._joint_vel[i], self._term[i] = [x + dt * xd, th + dt * thd], [xd + dt * xdd, thd + dt * thdd], abs(x + dt * xd) > 3.0
                self._obs = env_ops.compute_observations(self._joint_pos, self._joint_vel, [0.0, 0.0], [0.0, 0.0])
                rew = env_ops.compute_reward([p[1] for p in self._joint_pos], [v[1] for v in self._joint_vel], [v[0] for v in self._joint_vel], self._term, dt)
                step_rewards.append(sum(rew))
                obs_sums.append(sum(abs(v) for row in self._obs for v in row))
            else:
                assert self.env is not None
                obs_d, rew_t, _, _, _ = self.env.step(acts)
                self._obs = obs_d["policy"]
                step_rewards.append(float(rew_t.sum().item()))
                obs_sums.append(float(self._obs.abs().sum().item()))
            self.step_count += 1
        return {"mode": self.mode, "task": self.task, "physics": self.physics, "num_envs": self.num_envs, "horizon_steps": steps, "step": self.step_count, **summarize_rollout(step_rewards, obs_sums)}

    def close(self) -> None:
        if self.env is not None:
            with contextlib.suppress(Exception):
                self.env.close()
            self.env = None
        with contextlib.suppress(Exception):
            self._stack.close()


def _dispatch(worker: IsaacEnvWorker, part: Any, planned: dict[str, set[str]], tracked: set[str], faults: dict[str, Any], method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method in ("shutdown", "ping"):
        return ({"ok": True}, True) if method == "shutdown" else ({"ok": True, "pid": os.getpid(), "mode": worker.mode, "num_envs": worker.num_envs}, False)
    if method == "step":
        if faults.pop("crash", False):
            os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0):
            time.sleep(float(hang_s))
        return {"ok": True, "output": worker.run_rollout(payload or {})}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault") or "")
        info.pop("fault", None)
        if fault in ("crash", "hang", "lost"):
            faults["crash" if fault == "crash" else "hang_s"] = True if fault == "crash" else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, fault == "lost"
        return harness.inject_participant_fault(part, fault, **info), False
    if method not in PARTICIPANT_METHODS:
        return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        mods = (payload or {}).get("modules", {})
        for mod, spec in mods.items():
            origin, src = getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod), spec.get("source") if isinstance(spec, dict) else None
            if origin and isinstance(src, str):
                linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        reply = part.prepare(payload)
        (planned.__setitem__(str(reply.get("txn")), set(mods)) if reply.get("ok") else _sync_linecache(part, mods))
        return reply, False
    reply = getattr(part, method)(payload)
    if method in ("commit", "abort", "undo"):
        affected = planned.pop(str(coordinator.txn_ref(payload)[0]), set())
        tracked.update(affected if method == "commit" else ())
        _sync_linecache(part, (tracked | affected) if method == "undo" else affected)
    return reply, False


def main(argv: list[str] | None = None) -> int:
    with open((argv or sys.argv[1:])[0], encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("fresh"):
        if sys.platform.startswith("linux") and spec.get("parent_pid") and os.getppid() != int(spec["parent_pid"]):
            return 0
        worker = IsaacEnvWorker(dict(spec["config"]))
        try:
            print("FRESH " + json.dumps({"output": worker.run_rollout(spec.get("workload") or {})}), flush=True)
        finally:
            worker.close()
        return 0
    cfg = spec["config"]
    parent_pid = int(cfg.get("parent_pid") or os.getppid())
    if parent_pid and os.getppid() != parent_pid:
        return 0
    worker = IsaacEnvWorker(cfg)
    for mod in cfg.get("modules") or ():
        importlib.import_module(mod)
    part, planned, tracked, faults, server = coordinator.LocalParticipant("sim0", scope={"worker": worker, "cfg": worker.cfg}, resets={"state": worker.reset_state}), {}, set(), {}, ipc.UnixJsonServer(spec["sock_path"], handler=None)
    try:
        while True:
            try:
                req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid:
                    return 0
                continue
            try:
                reply, should_exit = _dispatch(worker, part, planned, tracked, faults, str(req.get("method") or ""), req.get("payload"))
            except Exception as exc:
                reply, should_exit = findings.not_ok(f"{req.get('method')}: {type(exc).__name__}: {exc}"), False
            reply_fn(reply)
            if should_exit:
                return 0
    finally:
        server.close()
        worker.close()


if __name__ == "__main__":
    raise SystemExit(importlib.import_module("adapter.worker").main())
