"""The sampler (sglang/srt/layers/sampler.py): runs eagerly on every TP rank after the logits
are gathered, outside the captured graphs. Greedy at temperature 0; seeded otherwise."""
TEMPERATURE_EPS = 1e-6


class SamplingBatchInfo:
    def __init__(self, temperatures, rng):
        self.temperatures, self.rng = temperatures, rng


class Sampler:
    def forward(self, logits, sampling_info):
        batch_next_token_ids = []
        for row, temperature in zip(logits, sampling_info.temperatures):
            if temperature < TEMPERATURE_EPS:
                batch_next_token_id = argmax(row)
            else:
                batch_next_token_id = sample(row, temperature, sampling_info.rng)
            batch_next_token_ids.append(batch_next_token_id)
        return batch_next_token_ids


def argmax(row):
    best = 0
    for index, value in enumerate(row):
        if value > row[best]:
            best = index
    return best


def sample(row, temperature, rng):
    weights = [max(value, 1) ** (1.0 / temperature) for value in row]
    return rng.choices(range(len(row)), weights=weights)[0]


def make_top_k_filter(k):
    limit = 0

    def keep(index):
        return index < k
    keep.limit = limit
    return keep
