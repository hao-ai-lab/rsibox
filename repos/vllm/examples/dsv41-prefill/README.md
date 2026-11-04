# DeepSeek-V4.1-Flash prefill: three PRs, hot-reloaded

From vLLM's post "Serving DeepSeek-V4.1-Flash at Scale" (2026-10-23): three changes raise new-token
prefill throughput by 35.6–43.9% (4K–64K new tokens after a 128K cached prefix, DEP2 on GB300).

| PR | Change | Blog gain at 32K | Groups |
|---|---|---|---|
| vllm-project/vllm#59894 | Extend SWA bounded replay to layer 20 | +19.9% | STATE, GRAPH, RANKS |
| vllm-project/vllm#62359 | 2 MB pages for shared Engram host tables | +2.8% | RESOURCE, RANKS |
| vllm-project/vllm#62240 | DeepSelect top-k in the prefill indexer | +15.6% | OP |

## Plan

1. Base: `0124e16a6c06` (first parent of #62359's merge), with #62240 and #59894 reverted.
   Not yet checked that the reverts apply cleanly.
2. Tag every changed function by group (`groups.md`, to write).
3. Per PR: PR tests offline, install, quick check, one 32K point, undo/redo A/B.
4. Full 4K–64K sweep once at the end.

No receipts yet.
