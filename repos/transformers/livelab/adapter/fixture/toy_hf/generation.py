"""Toy HuggingFace-style `GenerationMixin` greedy decode loop."""
from adapter.fixture.toy_hf.logits_process import BiasLogitsProcessor, LogitsProcessorList, scale_scores

def argmax_token(scores: list[float]) -> int:
    best_index = 0
    best_value = scores[0]
    for index, value in enumerate(scores):
        if value > best_value:
            best_value, best_index = value, index
    return best_index

class GenerationMixin:
    def _sample(self, input_ids: list[int], logits_processor: LogitsProcessorList, max_new_tokens: int, temperature: float = 0.0) -> list[int]:
        sequence, generated = list(input_ids), []
        for _ in range(int(max_new_tokens)):
            processed = logits_processor(sequence, self.forward_logits(sequence))
            next_token = argmax_token(scale_scores(processed, float(temperature)))
            sequence.append(next_token)
            generated.append(next_token)
        return generated

    def generate(self, input_ids_batch: list[list[int]], max_new_tokens: int = 6, temperature: float = 0.0) -> list[list[int]]:
        processors = LogitsProcessorList([BiasLogitsProcessor()])
        return [self._sample(prompt_ids, processors, max_new_tokens, temperature) for prompt_ids in input_ids_batch]
