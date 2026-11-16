"""Shared token scoring helpers imported by rollout and trainer actors."""
LOGIT_SCALE = 2
def initial_bias(): return 3
def make_shaper(offset):
    extra = 0
    return [extra, lambda value: value + offset][1]
def score_prompt(prompt_id, weight):
    raw = (prompt_id + weight) * LOGIT_SCALE
    return raw + 1
