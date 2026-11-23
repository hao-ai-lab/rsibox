"""Per-sample vLLM sampling seeds, for checks that must be bit-exact (`RSIBOX_RL_SEED_PER_SAMPLE=1`).

verl's `rollout.full_determinism` gives every request on a replica the *same* seed
(`vllm_async_server.generate`: `seed = replica_rank + rollout.seed`), so a group's `n` samples
collapse to one text per replica, and which replica serves a request (least-loaded routing) decides
its text. Seeding each request from its `(sample_index, session_id)` keeps sampling diverse within a
group and independent of routing, so a step after a coupled `S0` reset reproduces the same rollouts.
`seed_agent_loops` runs inside an `AgentLoopWorker` actor (through `__ray_call__`); it wraps the
class's `_run_agent_loop`, which both the TQ and the plain agent loop call once per session.
"""
import os

STRIDE = 1000   # sessions per sample: seed = base + STRIDE * sample_index + session_id


def seed_for(base, trajectory, session_id):
    return int(base) + STRIDE * int((trajectory or {}).get("sample_index", 0) or 0) + int(session_id or 0)


def seed_agent_loops(host, base_seed):
    cls = type(host)
    orig = cls._run_agent_loop
    if getattr(orig, "_rsibox_seeded", None) == int(base_seed):
        return {"ok": True, "pid": os.getpid(), "already": True}

    async def seeded(self, sampling_params, trajectory, *args, **kwargs):
        params = dict(sampling_params)
        params["seed"] = seed_for(base_seed, trajectory, kwargs.get("session_id", 0))
        return await orig(self, params, trajectory, *args, **kwargs)

    seeded._rsibox_seeded, seeded._rsibox_orig = int(base_seed), orig
    cls._run_agent_loop = seeded
    return {"ok": True, "pid": os.getpid(), "base_seed": int(base_seed)}
