"""Vocab-parallel logits: each TP rank computes its slice of the vocabulary; the model runner
all-gathers the slices. Runs eagerly, outside the captured graphs."""
LOGIT_MODULUS = 10007


class LogitsProcessor:
    def __init__(self, vocab_size, tp_rank, tp_size):
        per = vocab_size // tp_size
        self.vocab_range = range(tp_rank * per, (tp_rank + 1) * per)

    def forward(self, hidden_states, vocab_offset):
        return [shard_logits(h, self.vocab_range, vocab_offset) for h in hidden_states]


def shard_logits(h, vocab_range, vocab_offset):
    return [(h * (v + 1 + vocab_offset) * 7919 + v * v) % LOGIT_MODULUS for v in vocab_range]
