"""Toy HuggingFace-style logits processors and score scaling helpers."""
TEMPERATURE_EPS = 1e-6

def make_score_filter(limit: int, offset: int = 0):
    def keep_index(index: int) -> bool:
        return index < limit
    return keep_index

SCORE_FILTER = make_score_filter(64, 0)

def scale_scores(scores: list[float], temperature: float) -> list[float]:
    if temperature < TEMPERATURE_EPS:
        return list(scores)
    return [value / temperature for value in scores]

class BiasLogitsProcessor:
    def __call__(self, input_ids: list[int], scores: list[float]) -> list[float]:
        step_bias = len(input_ids) % 5
        return [score + ((index + 1 + step_bias) % 7) * 0.1 for index, score in enumerate(scores) if SCORE_FILTER(index)]

class LogitsProcessorList(list):
    def __call__(self, input_ids: list[int], scores: list[float]) -> list[float]:
        current = list(scores)
        for processor in self:
            current = processor(input_ids, current)
        return current
