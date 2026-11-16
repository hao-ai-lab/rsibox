"""Toy `@remote` Megatron trainer actor (`MegatronTrainRayActor`)."""
import time
from adapter.toymiles import remote
from adapter.toymiles.ops import initial_bias, make_shaper, score_prompt


def compute_step_advantages(rewards):
    anchor = rewards[0]
    return [reward - anchor + 1 for reward in rewards]


@remote
class MegatronTrainRayActor:
    def __init__(self, rank=0):
        self.rank, self.weight, self.step_count = rank, initial_bias() + 1, 0
        self.shaper, self.next_fault = make_shaper(0), None

    def train_step(self, groups):
        if (fault := self.next_fault) is not None:
            self.next_fault = None
            if fault == "crash": raise RuntimeError(f"trainer{self.rank}: actor crashed in train_step")
            time.sleep(float(fault))
        step_gain = 1
        metrics = [self.shaper(score_prompt(g[0]["prompt"], self.weight)
                               + sum(compute_step_advantages([s["reward"] for s in g]))) * step_gain
                   for g in groups]
        self.weight += sum(metrics) % 5 + 1; self.step_count += 1
        return metrics + [self.weight]

    def get_state(self): return {"weight": self.weight, "step_count": self.step_count}
    def set_state(self, saved): self.weight, self.step_count = int(saved["weight"]), int(saved["step_count"]); return True
