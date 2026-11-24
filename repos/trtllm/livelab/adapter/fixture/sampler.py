"""Toy TensorRT-LLM PyTorch sampler module mirroring `tensorrt_llm._torch.pyexecutor.sampler`."""
LOGIT_OFFSET, VOCAB_SIZE = 1, 32

def initial_logit_bias() -> int:
    return 3

def make_bias_fn(offset: int):
    extra = 1
    return lambda score: score + offset

def score_logits(prompt_tokens: list[int], step_index: int, startup_bias: int) -> list[int]:
    seed_sum = sum(prompt_tokens) + step_index * 3 + startup_bias
    return [((seed_sum + token_id * 7) % 19) + LOGIT_OFFSET for token_id in range(VOCAB_SIZE)]

def greedy_search_sampling_batch(logits: list[int]) -> int:
    best_index = max(range(len(logits)), key=lambda idx: logits[idx])
    return best_index
