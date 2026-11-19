"""GSM8K rule reward for the live lab, loaded by every `RewardLoopWorker` through
`reward.custom_reward_function.path` (verl keeps only `compute_score`, no `sys.modules` entry).

Self-contained so `tools.rescore` can load it without verl. Its shape doubles as the conformance
harness's edit targets (`adapter/conformance_lab.py`): `SCORE` is module state (an edit is refused),
`format_score` runs only at import (an edit installs but is never witnessed), `extract_solution`
holds a closure, and `compute_score` is the visible target.
"""
import re

SCORE = 1.0


def format_score():
    """Startup-only: evaluated once at import to set the reward of a well-formed wrong answer."""
    return 0.0


FORMAT_SCORE = format_score()


def extract_solution(solution_str):
    checks = [lambda m: m is not None]
    matches = [m for m in re.finditer(r"#### (\-?[0-9\.\,]+)", solution_str) if all(rule(m) for rule in checks)]
    if not matches:
        return None
    return matches[-1].group(1).replace(",", "").replace("$", "")


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return SCORE if answer == str(ground_truth) else FORMAT_SCORE
