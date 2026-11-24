"""Toy TensorRT-LLM PyExecutor + KVCacheManager mirroring `tensorrt_llm._torch.pyexecutor`."""
from adapter.fixture import sampler

class ToyKVCacheImpl:
    def __init__(self) -> None:
        self.allocated, self.entries = True, {}
    def release_pools(self) -> None:
        self.allocated = False
        self.entries.clear()
    def allocate_pools(self, enable_block_reuse: bool = False) -> None:
        del enable_block_reuse
        self.allocated = True
    def get_block_pool_pointers(self) -> list[int]:
        return [1] if self.allocated else []
    def get_layer_to_pool_mapping(self) -> list[int]:
        return [0] if self.allocated else []

class ToyKVCacheManager:
    def __init__(self) -> None:
        self.impl = ToyKVCacheImpl()
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()

def format_token_step(token_id: int, step_index: int) -> int:
    return (token_id + step_index) % sampler.VOCAB_SIZE

class ToyPyExecutor:
    def __init__(self) -> None:
        self.startup_bias, self.kv_cache_manager = sampler.initial_logit_bias(), ToyKVCacheManager()
        self.active_requests, self.history = [], []
    def generate(self, prompts: list[list[int]], max_tokens: int = 6) -> list[list[int]]:
        if not self.kv_cache_manager.impl.allocated:
            raise RuntimeError("KV cache pools are released; call resume_memory before generate")
        outputs: list[list[int]] = []
        for prompt_tokens in prompts:
            generated = list(prompt_tokens)
            for step_index in range(max_tokens):
                next_token = sampler.greedy_search_sampling_batch(sampler.score_logits(generated, step_index, self.startup_bias))
                step_token = format_token_step(next_token, step_index)
                generated.append(step_token)
                self.history.append(step_token)
            outputs.append(generated[-max_tokens:])
        return outputs
