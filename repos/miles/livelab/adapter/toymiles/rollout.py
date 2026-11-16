"""Toy `@remote` SGLang rollout actor (`RolloutManager`) with optional custom reward file."""
import importlib.util
from adapter.toymiles import remote
from adapter.toymiles.ops import initial_bias, score_prompt


def _load_reward_file(path):
    spec = importlib.util.spec_from_file_location("miles_custom_reward", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.compute_reward


@remote
class RolloutManager:
    def __init__(self, reward_file=None):
        self.weight, self.kv_cache, self.paused = initial_bias(), {}, False
        self.reward_fn = _load_reward_file(reward_file) if reward_file else (lambda tok: tok)

    def generate_batch(self, prompts):
        return [[{"prompt": p, "token": (t0 := self.kv_cache.setdefault(p, score_prompt(p, self.weight))),
                  "reward": self.reward_fn(t0)},
                 {"prompt": p, "token": (t1 := score_prompt(p + 1, self.weight)),
                  "reward": self.reward_fn(t1)}] for p in prompts]

    def update_weights(self, weight): self.weight = int(weight); return self.weight
    def flush_cache(self): self.kv_cache.clear(); return True
    def pause_generation(self): self.paused = True
    def continue_generation(self): self.paused = False
