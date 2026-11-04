# DeepSeek-V4.1-Flash prefill: three PRs, hot-reloaded

From vLLM's post "Serving DeepSeek-V4.1-Flash at Scale" (2026-10-23): three changes raise new-token
prefill throughput by 35.6–43.9% (4K–64K new tokens after a 128K cached prefix, DEP2 on GB300).

| PR | Change | Blog gain at 32K | Groups | Patch |
|---|---|---|---|---|
| vllm-project/vllm#59894 | Extend SWA bounded replay to layer 20 | +19.9% | STATE, GRAPH, RANKS | [patches/59894.patch](patches/59894.patch) |
| vllm-project/vllm#62359 | 2 MB pages for shared Engram host tables | +2.8% | RESOURCE, RANKS | [patches/62359.patch](patches/62359.patch) |
| vllm-project/vllm#62240 | DeepSelect top-k in the prefill indexer | +15.6% | OP | [patches/62240.patch](patches/62240.patch) |

Each patch is the PR's exact upstream diff (`git diff <merge>^ <merge>`).

## Bases

The three PRs merged three weeks apart (1,413 commits, 66 touching these files), so they do not
share one clean base. Checked on 2026-11-04:

| Stack | Base | Then | Checked |
|---|---|---|---|
| A | `0124e16a6c06` with #62240 reverted (`git revert c105af429241`) | +62359, +62240, in either order | Both patches apply; A + both equals upstream `4e0ad5720361` exactly |
| B | `8cbd5d03006` (parent of #59894) | +59894 | Applies |

Not possible without porting: reverting #59894 on the Oct 29 base conflicts in all 6 of its files;
cherry-picking #62240 or #62359 onto the Oct 9 base conflicts too.

## Plan

1. Tag every changed function by group, per patch.
2. Per PR: its own tests offline, `install`, a quick check, one 32K point, `uninstall`/`install`
   to A/B, then the full workload ([../../kb/workload.md](../../kb/workload.md)).
3. Start with #62240 on stack A (one function, OP).

No receipts yet.
