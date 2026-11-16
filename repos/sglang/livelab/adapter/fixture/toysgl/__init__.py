"""toysgl: an SGLang-shaped toy engine for the CPU live lab. Standard library only.

It keeps the parts of SGLang's process layout the live lab depends on, under the same module
paths and names (`srt/entrypoints/engine.py`, `srt/managers/scheduler.py`, ...):

- `Engine` (driver process) spawns one `Scheduler` process per TP rank with
  `run_scheduler_process_func`, a class attribute a subclass may override, as in SGLang.
- Rank 0 reads requests and RPCs from the driver and broadcasts them to every TP rank in
  `recv_requests`, so every rank processes the same control message in the same loop iteration.
- The vocab-parallel logits are all-gathered across ranks in every forward: a rank held at
  a gap while another runs a batch blocks both, as NCCL collectives do.
- `event_loop_overlap` defers `process_batch_result` by one iteration (`result_queue`);
  `pause_generation` drains it, as upstream does.
- `CudaGraphRunner` captures decode graphs per batch size. A captured graph is a lookup table
  of the decoder layers: replay runs no Python of the model, so an edit of layer code changes
  nothing a replay does until the graphs are captured again.
- `RadixCache` keeps KV and the hybrid (Mamba-like) recurrent state of finished prefixes;
  `flush_cache` clears it with the request, KV and state pools.

Not modeled: tokenizer and detokenizer processes, paging, chunked prefill, DP attention,
speculative decoding, PD disaggregation.
"""
