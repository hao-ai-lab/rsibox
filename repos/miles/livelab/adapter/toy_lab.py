"""ToyMilesLab: CPU conformance driver over spawned Ray-shaped rollout and trainer actors."""
import json, multiprocessing as mp, os, subprocess, sys, threading, types
import adapter
from adapter.lab import MilesLab
from adapter.state import CoupledWeightReset, EpochDataBuffer
from fastrsi import harness as H
import ray_participants as RP

OPS, ROLLOUT, TRAINER = "adapter.toymiles.ops", "adapter.toymiles.rollout", "adapter.toymiles.trainer"
PROMPTS, RPC_ERR = [1, 2, 4], "__miles_rpc_error__"
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_edit = H.edit_once
FRESH_SCRIPT = """import json, sys
livelab_dir, overlay_root, prompts, names = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
sys.path[:0] = [overlay_root, livelab_dir]
from adapter.toymiles import instantiate_actor, rollout, trainer
r_actor = instantiate_actor(rollout.RolloutManager)
t_actors = {n: instantiate_actor(trainer.MegatronTrainRayActor, rank=i) for i, n in enumerate(names)}
r_actor.update_weights(next(iter(t_actors.values())).weight); groups = r_actor.generate_batch(prompts)
print(json.dumps({n: actor.train_step(groups) for n, actor in t_actors.items()}))
"""


def _actor_loop(pipe, livelab_dir, pkg_root, module_name, class_name, kwargs):
    sys.path[:0] = [p for p in (pkg_root, livelab_dir) if p not in sys.path]
    from adapter.toymiles import instantiate_actor
    instance = instantiate_actor(getattr(__import__(module_name, fromlist=[class_name]), class_name), **kwargs)
    pipe.send("ready")
    while True:
        try: kind, target, args, call_kw = pipe.recv()
        except EOFError: break
        if kind == "shutdown": pipe.send({"ok": True}); break
        try: pipe.send(target(instance, *args, **call_kw) if kind == "ray_call" else getattr(instance, target)(*args, **call_kw))
        except Exception as err: pipe.send({RPC_ERR: f"{type(err).__name__}: {err}"})


def _inject_fault(host, fault, name, after):
    if fault in ("fail_apply", "fail_commit", "tamper"):
        kw = {"after": after} if fault == "fail_apply" else ({"module": OPS, "function": "score_prompt"} if fault == "tamper" else {})
        return H.inject_participant_fault(RP._LOCAL[name], fault, **kw)["ok"]
    host.next_fault = fault
    return True


class SpawnedActorHandle:
    def __init__(self, ctx, module_name, class_name, **kwargs):
        parent_pipe, child_pipe = ctx.Pipe()
        self.pipe, self.lock, self.lost = parent_pipe, threading.Lock(), False
        self.proc = ctx.Process(target=_actor_loop, daemon=True, args=(
            child_pipe, adapter.LIVELAB, PACKAGE_ROOT, module_name, class_name, kwargs))
        self.proc.start(); assert self.pipe.recv() == "ready"
        self.__ray_call__ = types.SimpleNamespace(remote=lambda fn, *a: ("ray_call", self, fn, a, {}))

    def invoke(self, method_name, *args, timeout_s=5.0, **kwargs):
        return pipe_get(("method", self, method_name, args, kwargs), timeout_s)

    def shutdown(self):
        with self.lock:
            if self.proc.is_alive():
                try: self.pipe.send(("shutdown", None, (), {}))
                except Exception: pass
                self.proc.join(1.0)
            if self.proc.is_alive(): self.proc.kill(); self.proc.join(1.0)
            self.pipe.close()

def pipe_get(request_tuple, timeout_s):
    kind, handle, target, args, kwargs = request_tuple
    with handle.lock:
        if handle.lost or not handle.proc.is_alive(): raise ConnectionError("actor process lost")
        handle.pipe.send((kind, target, args, kwargs))
        if not handle.pipe.poll(timeout_s or 5.0): handle.lost = True; raise TimeoutError("actor timed out")
        reply = handle.pipe.recv()
    if isinstance(reply, dict) and RPC_ERR in reply: raise RuntimeError(reply[RPC_ERR])
    return reply


class ToyMilesLab(MilesLab):
    """CPU conformance lab running 1 `RolloutManager` + 2 `MegatronTrainRayActor` processes."""
    name = "miles-toy"

    def __init__(self, ranks=2, receipts=None, mode="sync", reward_file=None, **kwargs):
        self.num_trainers, self.reward_file = ranks, reward_file
        kwargs.setdefault("rpc_timeout", 5.0); kwargs.setdefault("check_timeout", 0.35)
        super().__init__(receipts=receipts, mode=mode, **kwargs)

    def make_runtime(self):
        ctx, step_t = mp.get_context("spawn"), self.check_timeout
        r_actor = SpawnedActorHandle(ctx, ROLLOUT, "RolloutManager", reward_file=self.reward_file)
        t_actors = {f"trainer{i}": SpawnedActorHandle(ctx, TRAINER, "MegatronTrainRayActor", rank=i)
                    for i in range(self.num_trainers)}
        buffer, pause_log = EpochDataBuffer(capacity=16), []
        sync_w = lambda: r_actor.invoke("update_weights", next(iter(t_actors.values())).invoke("get_state", timeout_s=step_t)["weight"], timeout_s=step_t)
        coupled = CoupledWeightReset(
            get_trainer_states=lambda: {k: a.invoke("get_state", timeout_s=step_t) for k, a in t_actors.items()},
            set_trainer_states=lambda s: [t_actors[k].invoke("set_state", v, timeout_s=step_t) for k, v in s.items()],
            sync_rollout_weights=sync_w, flush_rollout_cache=lambda: r_actor.invoke("flush_cache", timeout_s=step_t),
            data_buffer=buffer)
        coupled.capture_s0(); sync_w()
        ctl = types.SimpleNamespace(
            events=pause_log,
            pause_generation=lambda: (pause_log.append("pause"), r_actor.invoke("pause_generation", timeout_s=step_t)),
            continue_generation=lambda: (pause_log.append("resume"), r_actor.invoke("continue_generation", timeout_s=step_t)),
            flush_cache=lambda: (pause_log.append("flush"), r_actor.invoke("flush_cache", timeout_s=step_t)))
        modules = [OPS, ROLLOUT, TRAINER] + ([self.reward_file] if self.reward_file else [])
        base = {m: H.read_module(m) for m in modules}

        def execute_step(workload=None):
            prompts = (workload or {}).get("prompts", PROMPTS)
            for grp in r_actor.invoke("generate_batch", prompts, timeout_s=step_t): buffer.put_group(grp)
            batch = buffer.get_groups(len(prompts))
            out = {k: a.invoke("train_step", batch, timeout_s=step_t) for k, a in t_actors.items()}
            sync_w(); return out

        return {"rollout_actors": {"rollout0": r_actor}, "train_actors": t_actors, "modules": modules,
                "base_text": base, "data_buffer": buffer, "coupled_reset": coupled, "rollout_controller": ctl,
                "get": pipe_get, "run_step": execute_step,
                "shutdown": lambda: [a.shutdown() for a in [r_actor, *t_actors.values()]]}

    def patches(self):
        ops_src, train_src = self.base_text[OPS], self.base_text[TRAINER]
        vis1 = _edit(ops_src, "    return raw + 1", "    return raw + 2")
        vis2 = _edit(vis1, "    return raw + 2", "    return raw + 3")
        vis3 = _edit(train_src, "        step_gain = 1", "        step_gain = 2")
        multi_t = _edit(vis3, "    return [reward - anchor + 1 for reward in rewards]",
                        "    return [reward - anchor + 2 for reward in rewards]")
        return {
            "noop": {"files": {OPS: _edit(ops_src, "    return raw + 1", "    val = raw + 1\n    return val")}},
            "visible": {"files": {OPS: vis1}},
            "chain": [{"files": {OPS: vis1}}, {"files": {OPS: vis2}}, {"files": {TRAINER: vis3}}],
            "multi": {"files": {OPS: vis1, TRAINER: multi_t}},
            "unreached": {"files": {OPS: _edit(ops_src, "def initial_bias(): return 3", "def initial_bias(): return 5")}},
            "refused": {
                "syntax": {"files": {OPS: _edit(ops_src, "    return raw + 1", "    return raw +")}},
                "closure": {"files": {OPS: _edit(ops_src, "lambda value: value + offset", "lambda value: value + offset + extra")}},
                "module_state": {"files": {OPS: _edit(ops_src, "LOGIT_SCALE = 2", "LOGIT_SCALE = 4")}},
                "empty": {"files": {OPS: _edit(ops_src, "def score_prompt(", "def score_prompt(  # note\n")}},
            }}

    def fresh_check(self, patch, workload=None):
        prompts, names = (workload or {}).get("prompts", PROMPTS), [f"trainer{i}" for i in range(self.num_trainers)]
        with H.PackageOverlay("adapter", (patch or {}).get("files")) as overlay:
            proc = subprocess.run([sys.executable, "-I", "-c", FRESH_SCRIPT, adapter.LIVELAB,
                                   overlay.root, json.dumps(prompts), json.dumps(names)],
                                  capture_output=True, text=True, timeout=30)
        if proc.returncode:
            lines = proc.stderr.strip().splitlines()
            return {"ok": False, "error": lines[-1] if lines else "fresh_check failed", "output": None}
        return {"ok": True, "output": json.loads(proc.stdout)}

    def inject(self, fault, **kwargs):
        target = kwargs.get("participant", f"trainer{self.num_trainers - 1}")
        handle = self.gate.handles[target]
        if fault == "lost": handle.lost = True
        elif fault in ("fail_apply", "fail_commit", "tamper", "crash", "hang"):
            arg = kwargs.get("hang_s", self.check_timeout + 0.4) if fault == "hang" else fault
            pipe_get(handle.__ray_call__.remote(_inject_fault, arg, target, kwargs.get("after", 1)), 5.0)
        else: return {"ok": False, "error": f"unknown fault {fault}"}
        return {"ok": True, "fault": fault, "participant": target}
