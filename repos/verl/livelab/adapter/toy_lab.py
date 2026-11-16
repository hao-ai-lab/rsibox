"""ToyVerlLab: `VerlLab` (`LabUnderTest`) on the multi-actor synchronous RL toy (`toyverl/`)."""
import os, shutil, tempfile
import ray_participants as RP
from adapter.lab import VerlLab
from adapter.toyverl import cluster as TC

ALGOS, ROLLOUT = "adapter.toyverl.algos", "adapter.toyverl.rollout"


def _edit(src, old, new):
    assert src.count(old) == 1, f"anchor not unique: {old!r}"
    return src.replace(old, new)


class ToyVerlLab(VerlLab):
    name, package, fresh_module = "verl-toy", "adapter", "adapter.toyverl.fresh"

    def __init__(self, receipts=None, **kw):
        self._tmp_dir = tempfile.mkdtemp(prefix="fastrsi-toyverl-")
        self.reward_file = os.path.join(self._tmp_dir, "custom_reward.py")
        with open(self.reward_file, "w", encoding="utf-8") as handle:
            handle.write(TC.DEFAULT_REWARD_SOURCE)
        kw.setdefault("rpc_timeout", 30.0)
        kw.setdefault("check_timeout", 3.0)
        super().__init__({"reward_file": self.reward_file}, receipts=receipts,
                         modules=[ALGOS, ROLLOUT, self.reward_file], **kw)

    def make_cluster(self):
        return TC.ToyVerlCluster(self.config)

    def default_workload(self):
        return {"prompts": [1, 2, 3, 4], "steps": 2}

    def close(self):
        super().close()
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def inject(self, fault, **kw):
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 1.5)
        if fault == "tamper":
            kw.setdefault("module", ALGOS)
            kw.setdefault("function", "grpo_advantages")
        kw.setdefault("participant", "rollout0" if fault in ("fail_apply", "fail_commit") else "trainer0")
        return super().inject(fault, **kw)

    def patches(self):
        algos_src, rollout_src, reward_src = self._base(ALGOS), self._base(ROLLOUT), self._base(self.reward_file)
        vis = _edit(rollout_src, "tok = sample_token(prompt_id, weight_sum, self.bias)", "tok = sample_token(prompt_id + 1, weight_sum, self.bias)")
        ch2 = _edit(vis, "tok = sample_token(prompt_id + 1, weight_sum, self.bias)", "tok = sample_token(prompt_id + 2, weight_sum, self.bias)")
        ch3 = {"files": {self.reward_file: _edit(reward_src, "return 0.25 if token_val % 2 == 0 else 0.0", "return 0.75 if token_val % 2 == 0 else 0.1")}}
        m_algos = _edit(algos_src, "return [tok for tok in tokens if all(rule(tok) for rule in checks)]", "return [int(tok) for tok in tokens if all(rule(tok) for rule in checks)]")
        m_roll = _edit(vis, "raw = int(round((prompt_id * 7 + weight_sum * 100.0 + bias * 10.0))) % 29", "raw = int(round((prompt_id * 9 + weight_sum * 100.0 + bias * 10.0))) % 29")
        m_rew = _edit(reward_src, "return 0.25 if token_val % 2 == 0 else 0.0", "return 0.50 if token_val % 2 == 0 else 0.0")
        return {
            "noop": {"files": {ROLLOUT: _edit(rollout_src, "weight_sum = sum(self.weights)", "weight_sum = sum(self.weights) + 0.0")}},
            "visible": {"files": {ROLLOUT: vis}},
            "chain": [{"files": {ROLLOUT: vis}}, {"files": {ROLLOUT: ch2}}, ch3],
            "multi": {"files": {ALGOS: m_algos, ROLLOUT: m_roll, self.reward_file: m_rew}},
            "unreached": {"files": {ALGOS: _edit(algos_src, "return 0.0", "return 5.0")}},
            "refused": {
                "syntax": {"files": {ALGOS: _edit(algos_src, "mean = sum(rewards) / count", "mean = sum(rewards / count")}},
                "closure": {"files": {ALGOS: _edit(algos_src, "lambda value: isinstance(value, int)]", "lambda value: isinstance(value, int) and checks]")}},
                "module_state": {"files": {ALGOS: _edit(algos_src, "CLIP_SCALE = 1.0", "CLIP_SCALE = 2.0")}},
                "empty": {"files": {ALGOS: _edit(algos_src, "def grpo_advantages(rewards):", "def grpo_advantages(rewards):  # note")}},
            },
        }
