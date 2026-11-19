"""Spawned multi-actor synchronous RL cluster for `ToyVerlLab`."""
import importlib.util, multiprocessing as mp, os, sys, time, types
from adapter import faults, state as S0

RPC_ERROR = "__verl_rpc_error__"
def _read_policy_weights(host):
    return list(host.weights["policy"])


DEFAULT_REWARD_SOURCE = """def format_bonus(token_val):
    return 0.25 if token_val % 2 == 0 else 0.0


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    token_val, target = int(solution_str), int(ground_truth)
    return (1.0 if token_val % 5 == target % 5 else 0.0) + format_bonus(token_val)
"""


class TrainerWorkerHost:
    def __init__(self, rank=0):
        from adapter.toyverl import algos
        self.rank, self.schedule_bias = rank, algos.init_schedule()
        self.weights = {"policy": [1.0, 0.5, -0.25]}
        self.optimizer, self.scheduler = {"step": 0, "momentum": [0.0, 0.0, 0.0]}, types.SimpleNamespace(num_steps=0)
        acfg = types.SimpleNamespace(micro_batch_size_per_gpu=2, infer_micro_batch_size_per_gpu=2, use_dynamic_bsz=False, max_token_len_per_gpu=1024, infer_max_token_len_per_gpu=1024)
        rcfg = types.SimpleNamespace(infer_micro_batch_size_per_gpu=2, use_dynamic_bsz=False, infer_max_token_len_per_gpu=1024)
        self.worker_dict = {"actor_rollout_ref": types.SimpleNamespace(
            actor=types.SimpleNamespace(engine=self, optimizer=self.optimizer, scheduler=self.scheduler, engine_config=acfg),
            ref=types.SimpleNamespace(engine_config=rcfg))}

    def train_step(self, tokens, rewards):
        from adapter.toyverl import algos
        advantages = algos.grpo_advantages(rewards)
        loss, deltas = algos.policy_gradient_step(self.weights["policy"], tokens, advantages)
        self.optimizer["step"] += 1
        self.scheduler.num_steps += 1
        for idx, val in enumerate(list(self.weights["policy"])):
            mom = round(0.5 * self.optimizer["momentum"][idx] + deltas[idx], 6)
            self.optimizer["momentum"][idx] = mom
            self.weights["policy"][idx] = round(val + mom + self.schedule_bias * 0.01, 6)
        return {"loss": loss, "advantages": advantages, "weights": list(self.weights["policy"]), "step": self.optimizer["step"]}


class RewardWorkerHost:
    def __init__(self, reward_file):
        spec = importlib.util.spec_from_file_location("custom_reward_fn", reward_file)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        self._score_fn = loaded.compute_score

    def score_batch(self, prompts, tokens):
        return [round(float(self._score_fn(data_source="toy", solution_str=str(t), ground_truth=str(p))), 6) for p, t in zip(prompts, tokens)]


def _actor_loop(pipe, role, actor_name, config):
    import adapter  # noqa: F401
    if role == "rollout":
        from adapter.toyverl import rollout
        host = rollout.instantiate_actor_copy(rollout.RolloutActor, rank=0)
    elif role == "trainer":
        host = TrainerWorkerHost(rank=0)
    else:
        host = RewardWorkerHost(config["reward_file"])
    fault_state = faults.STATE
    pipe.send("ready")
    while True:
        try:
            msg = pipe.recv()
        except EOFError:
            break
        kind = msg.get("kind")
        if kind == "shutdown":
            pipe.send({"ok": True})
            break
        if fault_state.get("lost"):
            pipe.send({RPC_ERROR: True, "error": f"{actor_name}: actor lost"})
            continue
        try:
            if kind == "ping":
                pipe.send({"ok": True, "pid": os.getpid()})
            elif kind == "inject":
                kw = dict(msg["kwargs"])
                kw.setdefault("module", "adapter.toyverl.algos"), kw.setdefault("function", "grpo_advantages")
                pipe.send(faults.inject(actor_name, kw.pop("fault"), **kw))
            elif kind == "ray_call":
                pipe.send(msg["fn"](host, *msg["args"]))
            elif kind == "method":
                if fault_state.pop("crash", False):
                    raise RuntimeError(f"{actor_name}: injected crash")
                if (hang_s := fault_state.pop("hang_s", None)) is not None:
                    time.sleep(hang_s)
                pipe.send(getattr(host, msg["method"])(*msg.get("args", ())))
        except Exception as err:
            pipe.send({RPC_ERROR: True, "error": f"{type(err).__name__}: {err}"})


class _SpawnedActor:
    def __init__(self, ctx, role, name, config):
        self.name, self.pipe, self._child = name, *ctx.Pipe()
        self.proc = ctx.Process(target=_actor_loop, args=(self._child, role, name, config), daemon=True)
        self.proc.start()
        assert self.pipe.recv() == "ready"
        self.__ray_call__ = types.SimpleNamespace(remote=lambda fn, *a: (self, {"kind": "ray_call", "fn": fn, "args": a}))

    def request(self, payload, timeout_s=10.0):
        self.pipe.send(payload)
        if not self.pipe.poll(timeout_s):
            raise TimeoutError(f"{self.name}: no reply within {timeout_s} s")
        reply = self.pipe.recv()
        if isinstance(reply, dict) and reply.get(RPC_ERROR):
            raise RuntimeError(reply["error"])
        return reply

    def call_method(self, method, *args, timeout_s=10.0):
        return self.request({"kind": "method", "method": method, "args": args}, timeout_s=timeout_s)

    def shutdown(self):
        try:
            self.pipe.send({"kind": "shutdown"})
        except Exception:
            pass
        self.proc.join(1.5)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(1.5)
        for conn in (self.pipe, self._child):
            conn.close()


class ToyVerlCluster:
    def __init__(self, config):
        self.config, ctx = dict(config), mp.get_context("spawn")
        self.rollouts = [_SpawnedActor(ctx, "rollout", "rollout0", self.config), _SpawnedActor(ctx, "rollout", "rollout1", self.config)]
        self.trainer, self.reward = _SpawnedActor(ctx, "trainer", "trainer0", self.config), _SpawnedActor(ctx, "reward", "reward0", self.config)
        self.actors = [*self.rollouts, self.trainer, self.reward]

    def actor_handles(self):
        return [(a.name, a) for a in self.actors]

    def resolve_ref(self, ref, timeout_s):
        return ref[0].request(ref[1], timeout_s=timeout_s)

    def ping_all(self, timeout_s=5.0):
        for a in self.actors:
            a.request({"kind": "ping"}, timeout_s=timeout_s)

    def _sync_rollout_from_trainer(self):
        weights = self.resolve_ref(self.trainer.__ray_call__.remote(_read_policy_weights), 10.0)
        for r_actor in self.rollouts:
            r_actor.call_method("sync_weights", weights)

    def capture_s0(self):
        res = self.resolve_ref(self.trainer.__ray_call__.remote(S0.worker_capture_s0), 10.0)
        self._sync_rollout_from_trainer()
        return res

    def restore_s0(self):
        res = self.resolve_ref(self.trainer.__ray_call__.remote(S0.worker_restore_s0), 10.0)
        if not res.get("ok"):
            return res
        self._sync_rollout_from_trainer()
        return {"ok": True, "reset": ["trainer_s0", "rollout_weights", "kv_cache", "optimizer", "rng"], "digest": res.get("digest")}

    def run_steps(self, workload):
        prompts, steps = list(workload.get("prompts", [1, 2, 3, 4])), int(workload.get("steps", 1))
        mid = max(len(prompts) // 2, 1)
        history = []
        for _ in range(steps):
            tokens = self.rollouts[0].call_method("generate", prompts[:mid], timeout_s=30.0) + self.rollouts[1].call_method("generate", prompts[mid:], timeout_s=30.0)
            scores = self.reward.call_method("score_batch", prompts, tokens, timeout_s=30.0)
            step_out = self.trainer.call_method("train_step", tokens, scores, timeout_s=30.0)
            for r_actor in self.rollouts:
                r_actor.call_method("sync_weights", step_out["weights"], timeout_s=30.0)
            history.append({"tokens": tokens, "scores": scores, "loss": step_out["loss"], "weights": step_out["weights"]})
        return {"steps": history}

    def inject_fault(self, fault, **kw):
        target = {a.name: a for a in self.actors}.get(kw.pop("participant", "trainer0"), self.trainer)
        return target.request({"kind": "inject", "kwargs": {"fault": fault, **kw}}, timeout_s=5.0)

    def shutdown(self):
        for a in self.actors:
            a.shutdown()
