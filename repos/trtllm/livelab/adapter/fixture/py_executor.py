"""Toy TensorRT-LLM PyExecutor + KVCacheManager mirroring `tensorrt_llm._torch.pyexecutor`."""
from adapter.fixture import sampler


class ToyKVCacheImpl:
    def __init__(self, *, shift_pointers_on_alloc: bool = False) -> None:
        self.allocated = True
        self.entries: dict[tuple[int, ...], list[int]] = {}
        self.pointer_base = 1
        self.shift_pointers_on_alloc = bool(shift_pointers_on_alloc)
        self.last_use_uvm: bool | None = None

    def release_pools(self) -> None:
        self.allocated = False
        self.entries.clear()

    def allocate_pools(self, use_uvm: bool = False) -> None:
        self.last_use_uvm = bool(use_uvm)
        if self.shift_pointers_on_alloc:
            self.pointer_base += 1
        self.allocated = True

    def get_block_pool_pointers(self) -> list[int]:
        return [self.pointer_base] if self.allocated else []

    def get_layer_to_pool_mapping(self) -> list[int]:
        return [0] if self.allocated else []


class ToyKVCacheManager:
    def __init__(self, *, shift_pointers_on_alloc: bool = False) -> None:
        self.impl = ToyKVCacheImpl(shift_pointers_on_alloc=shift_pointers_on_alloc)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()


def format_token_step(token_id: int, step_index: int) -> int:
    return (token_id + step_index) % sampler.VOCAB_SIZE


class ToyPyExecutor:
    def __init__(self, *, shift_pointers_on_alloc: bool = False) -> None:
        self.startup_bias = sampler.initial_logit_bias()
        self.kv_cache_manager = ToyKVCacheManager(shift_pointers_on_alloc=shift_pointers_on_alloc)
        self.active_requests: list[int] = []
        self.history: list[int] = []

    def generate(self, prompts: list[list[int]], max_tokens: int = 6) -> list[list[int]]:
        kv_manager = self.kv_cache_manager
        if not kv_manager.impl.allocated:
            raise RuntimeError("KV cache pools are released; call resume_memory before generate")
        if not kv_manager.kv_cache_pool_pointers or not kv_manager.kv_cache_pool_mapping:
            raise RuntimeError("KV cache pool pointers or mapping were not refreshed after resume")
        outputs: list[list[int]] = []
        for prompt_tokens in prompts:
            generated = list(prompt_tokens)
            for step_index in range(max_tokens):
                logits = sampler.score_logits(generated, step_index, self.startup_bias)
                next_token = sampler.greedy_search_sampling_batch(logits)
                step_token = format_token_step(next_token, step_index)
                generated.append(step_token)
                self.history.append(step_token)
            outputs.append(generated[-max_tokens:])
        return outputs
