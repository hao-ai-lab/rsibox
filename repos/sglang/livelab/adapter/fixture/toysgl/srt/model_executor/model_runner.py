"""ModelRunner (sglang/srt/model_executor/model_runner.py): the model, its pools, the decode
graphs, the vocab-parallel logits and the sampler of one TP rank."""
import random

from adapter.fixture.toysgl.srt.layers.logits_processor import LogitsProcessor
from adapter.fixture.toysgl.srt.layers.sampler import Sampler, SamplingBatchInfo
from adapter.fixture.toysgl.srt.mem_cache.memory_pool import (MambaPool, ReqToTokenPool,
                                                      TokenToKVPoolAllocator)
from adapter.fixture.toysgl.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from adapter.fixture.toysgl.srt.models.toy_hybrid import ToyHybridForCausalLM


class ModelRunner:
    def __init__(self, server_args, tp_rank, tp_group):
        self.server_args = server_args
        self.tp_rank, self.tp_group = tp_rank, tp_group
        self.vocab_size = server_args.get("vocab_size", 64)
        self.model = ToyHybridForCausalLM()
        self.logits_processor = LogitsProcessor(self.vocab_size, tp_rank, tp_group.size)
        self.sampler = Sampler()
        self.random_seed = server_args.get("random_seed", 0)
        self.rng = random.Random(self.random_seed)
        size = server_args.get("max_running_requests", 8)
        self.req_to_token_pool = ReqToTokenPool(size)
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(size * 64)
        self.mamba_pool = MambaPool(size)
        self.init_vocab_offset()
        self.graph_runner = None
        self.init_device_graphs()

    def init_vocab_offset(self):
        """Start-up only: never runs again while the engine serves."""
        self.vocab_offset = 0

    def init_device_graphs(self):
        if not self.server_args.get("disable_cuda_graph", False):
            self.graph_runner = CudaGraphRunner(self)

    # -- forward -------------------------------------------------------------------------
    def _step(self, req, token, h):
        """Attention over the request's KV, then the recurrent state of its slot."""
        model = self.model
        out = model.attend(h, req.kv_sum)
        req.kv.append(h)
        req.kv_sum += h
        state = model.mamba_step(self.mamba_pool.states[req.mamba_slot], out)
        self.mamba_pool.states[req.mamba_slot] = state
        req.states.append(state)
        return model.final(out, state)

    def forward_extend(self, reqs):
        """Prefill runs eagerly (no graphs), from the cached prefix on."""
        model, hidden = self.model, []
        for req in reqs:
            tokens = req.origin_input_ids
            if req.states:
                self.mamba_pool.states[req.mamba_slot] = req.states[-1]
            last = None
            for position in range(len(req.kv), len(tokens)):
                h = model.layers_forward(model.embed(tokens[position], position))
                last = self._step(req, tokens[position], h)
            hidden.append(last)
        return hidden

    def forward_decode(self, reqs):
        model = self.model
        positions = [len(req.origin_input_ids) + len(req.output_ids) - 1 for req in reqs]
        embedded = [model.embed(req.output_ids[-1], position)
                    for req, position in zip(reqs, positions)]
        if self.graph_runner is not None and self.graph_runner.can_run(len(reqs)):
            layer_out = self.graph_runner.replay(embedded)
        else:
            layer_out = [model.layers_forward(h) for h in embedded]
        return [self._step(req, req.output_ids[-1], h) for req, h in zip(reqs, layer_out)]

    def forward(self, batch):
        if batch.forward_mode == "extend":
            return self.forward_extend(batch.reqs)
        return self.forward_decode(batch.reqs)

    def sample(self, hidden, batch):
        shard = self.logits_processor.forward(hidden, self.vocab_offset)
        shards = self.tp_group.all_gather(shard)
        logits = [[value for rank_rows in shards for value in rank_rows[i]]
                  for i in range(len(hidden))]
        info = SamplingBatchInfo([req.temperature for req in batch.reqs], self.rng)
        return self.sampler.forward(logits, info)
