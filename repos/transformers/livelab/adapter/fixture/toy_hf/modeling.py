"""Toy causal language model with deterministic RMSNorm, KV cache, and CPU offload hooks."""
import random
from adapter.fixture.toy_hf.generation import GenerationMixin

MODULUS = 97

def rms_norm(hidden_values: list[int]) -> list[int]:
    return [(value * 7 + 1) % MODULUS for value in hidden_values]

class ToyCausalLM(GenerationMixin):
    def __init__(self, vocab_size: int = 64, seed: int = 0) -> None:
        self.vocab_size, self.seed, self.device, self.offloaded, self.vocab_offset = int(vocab_size), int(seed), "cpu", False, 0
        self.kv_cache: dict[tuple[int, ...], list[float]] = {}
        self.rng = random.Random(self.seed)
        self._init_weights()

    def _init_weights(self) -> None:
        self.vocab_offset = 0

    def forward_logits(self, sequence: list[int]) -> list[float]:
        cache_key = tuple(sequence)
        if cache_key in self.kv_cache:
            return list(self.kv_cache[cache_key])
        state_sum = sum((pos + 1) * val for pos, val in enumerate(rms_norm(sequence)))
        logits = [float((state_sum + token_id * 13 + self.vocab_offset * 19) % MODULUS) for token_id in range(self.vocab_size)]
        self.kv_cache[cache_key] = logits
        return list(logits)

    def clear_cache(self, seed: int | None = None) -> None:
        self.kv_cache.clear()
        self.rng.seed(self.seed if seed is None else int(seed))

    def offload_to_cpu(self) -> None:
        self.kv_cache.clear()
        self.offloaded = True

    def restore_to_device(self, target_device: str = "cpu") -> None:
        self.offloaded, self.device = False, str(target_device)
