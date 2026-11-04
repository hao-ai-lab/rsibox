# Changed functions by group

Non-test functions changed by each patch, read from the upstream source at the merge commit.
Groups are from reading the code; nothing here is measured yet.

## #62240 · DeepSelect top-k (1 file)

| Function | Group | Why |
|---|---|---|
| `sparse_attn_indexer` (`model_executor/layers/sparse_attn_indexer.py`) | OP | Registered custom op `vllm::sparse_attn_indexer`, listed as a graph-splitting op, and decorated with `@eager_break_during_capture`: it runs eagerly between captured graph segments |
| module imports | OP | Two new names, `deep_select_topk` and `is_deep_select_supported`; both already exist in `indexer_topk` at the base, so install adds them to the module's globals |

**Install rule (first general rule):** the decorator uses `functools.wraps`, and breakable-graph
replay records `lambda: fn(...)` over the *inner* function. So install swaps
`sparse_attn_indexer.__wrapped__.__code__`. Swapping the module attribute or the wrapper's code
would change nothing: the v0 silent no-op. Exception: batches captured in `CUDAGraphMode.FULL`
call `fn` inside the graph, so for those the change is GRAPH and needs a recapture.

**Check:** witness that the inner function ran the new code; compare top-k indices with the old
path on captured 32K inputs (bf16 vs fp32 ranking: near-identical, not bit-exact); one 32K point.

## #62359 · Engram host tables (5 files, 1 new)

| Function | Group | Why |
|---|---|---|
| `nvidia/ops/engram_host_table.py` (new module: `share_host_table`, `_create_table`, `_map_table`, …) | RESOURCE | New allocator; nothing runs it until the tables are rebuilt |
| `DPSharedEngramStorage.__init__`, `_allocate_shm`, `_unregister`, `get_views` | RESOURCE, RANKS | Builds the shared tables across DP ranks at load time |
| `ParallelEngramEmbedding.__init__`, `_storage`, `_get_shard_info`, `collapse_huge_pages`, `_uses_small_host_pages` | STATE | Built once per layer at load |
| `ParallelEngramEmbedding.lookup` | OP | Per-call; small change |
| `gather_errors`, `describe_error`, `_shm_shortfall`, `_allow_any_numa_node`, `can_share_engram_tables` | OP | Helpers called by the above |
| `EngramConfig` (docstring), `weight_utils` (`_get_fs_type` and helpers removed) | none / restart | Config and loader code that only runs at startup |
| `cpu_resource_utils.get_available_memory` (new) | OP | Helper |

**Install needs:** add the new module, swap the functions, then rebuild every rank's Engram
storage (allocate new tables, copy 2 × 23.6 GiB per rank from the old ones, switch views in
each `ParallelEngramEmbedding`, free the old ones), all ranks or none.

## #59894 · Layer-20 replay (4 files)

| Function | Group | Why |
|---|---|---|
| `DeepseekV4Attention.forward_kv`, `DeepseekV4DecoderLayer.write_kv` (new methods) | GRAPH | Called from the replay layers' path |
| `DeepseekV4Attention.forward`, `DeepseekV4DecoderLayer.forward`, `DeepseekV4Model._run_replay_layers` | GRAPH | Model forward, inside captured graphs |
| `DeepseekV4Model.__init__`, `_decoder_replay_supported` | STATE | Sets `decoder_replay_start` and builds `DecoderReplayLayers` |
| `DecoderReplayLayers` (`__init__`, `__call__`, `uses_replay_batch`, `graph_uses_replay_batch`, `_run`, `_run_in_graph_break`) | STATE, GRAPH | Object built at init; its methods run in and around graphs |
| `DeepseekV41ModelState.__init__`, `add_request`, `_prepare_replay_batch`, `_copy_attn_groups` | STATE | Per-engine model state and batch preparation |
| `DeepseekV41ModelState._agree_across_dp` | RANKS | DP ranks agree on the replay batch |

**Install needs:** swap the methods, rebuild `DecoderReplayLayers` and the model state's replay
groups, then recapture the replay CUDA graphs, on all DP ranks.
