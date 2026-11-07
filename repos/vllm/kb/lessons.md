# vLLM lessons

What misled us on vLLM and the rule that now prevents it. Edit rows in place when a rule changes;
this is not a log. Framework lessons are in `docs/lessons.md`. "The replay" below is our
2026-11-06/07 replay of 19 upstream PRs through v1 on GB200 (Qwen3-8B and Qwen3.5-35B-A3B-FP8).

## Edits a swap does not reach

| Pitfall | Rule | Evidence |
|---|---|---|
| A Triton `@jit` kernel compiles from its source text and caches per device, so swapping its Python code object changes nothing. | Route `@jit` targets out of the plain swap: today to T1. The prototype's in-place rebuild (0.07 s) is not ported to v1 yet. | vllm-project/vllm#58231, vllm-project/vllm#61213 and vllm-project/vllm#62046 went T1 in the replay ("T0 cannot rebuild Triton kernels (v1)"). |
| An edited `__init__` swaps cleanly, but the objects built at start-up keep the old state. | A constructor target (`__init__`, `__new__`, `__post_init__`) blocks with `constructor_edit`, route T1, overridable. | vllm-project/vllm#61323 (norm layer constructors): T0 committed, witness 0 on every participant. Fixed in #53, CPU only; pending replay. |
| The graph-split decorator `eager_break_during_capture` runs once at import and during capture; captured graphs keep its old decision. | On a graph or compiled engine, refuse an edit of the decorator (or code nested in it) as `captured_target`, route T1. | vllm-project/vllm#59941 at TP2 with CUDA graphs: T0 committed, witness 0 on both ranks and the engine core. Fixed in #54, CPU only; pending replay. |
| A graph rebuild works idle but kills the engine under load: the rebuild empties the workers' request state. | Preempt requests before any rebuild; an idle rebuild proves nothing about serving. | Prototype GPU run, 2026-11-05 (`KeyError`, then `EngineDeadError`); listed under "Not built" in [architecture.md](../docs/architecture.md). |

## Zygote refresh

| Pitfall | Rule | Evidence |
|---|---|---|
| Re-importing an edited `layernorm` re-ran `@CustomOp.register("rms_norm")`, which refuses a known name: every forked server died ("Duplicate op name: rms_norm"). | Before the re-import, forget the refreshed modules' entries in vLLM's name registries (`NAME_REGISTRIES`). Registries outside that list still collide. | vllm-project/vllm#61323: two runs fell back to a plain restart (113 s, 195 s). Fixed in #52, CPU only; pending replay. |
| An edit to almost any module refreshes most of vLLM through the `vllm.config` / `vllm.platforms` import cycle. | Expect a core edit to cost a full re-import (+1.5–2.8 s); leaf edits refresh in milliseconds. | [costs.md](costs.md), zygote section. |

## Undo and `at_base`

| Pitfall | Rule | Evidence |
|---|---|---|
| vLLM's `LazyLoader` binds modules into a namespace the first time a backend is used, so an exact undo read as "not at base". | A name the transaction did not add, whose value is a loaded module, goes to `lazy_imports` and does not make `at_base` false; our own extra names still do (`extra_names`). | vllm-project/vllm#62872, eager and graph: the only extras were two `llguidance` names. Fixed in #55, CPU only; pending replay. |

## Starting and measuring

| Pitfall | Rule | Evidence |
|---|---|---|
| Two intended cold starts used `--load-format ipc_cache` with no weight daemon; each waited 300 s and died without writing caches, so the next "warm" runs that copied those caches were cold. | Check the start that seeds a cache finished; label each run by what its caches held. | Our 2026-11-06 start-up dissection: "warm" runs at 181.9 s (eager) and 223.6 s (graph) vs 54.9 s and 59.1 s truly warm. |
| Most of a cold start is FlashInfer's JIT build, not weights. | Seed new cache roots from a verified warm copy before timing anything. | Same dissection: about 121 s of JIT builds in a cold TP1 start; weights 2.6–12.9 s. |

## Sharing a node

| Pitfall | Rule | Evidence |
|---|---|---|
| Another server on the same port answered `/health`, so a launcher reported an engine that never bound as healthy after 0.01 s. | Check the port has no listener before launch; use a private port range on shared nodes. | Our 2026-11-05 multimodal run. |
| Engines were killed about 10.5–12 min after start, all processes in the same second. | On a cluster that reaps unallocated GPU processes, confirm the allocation before launch; read the reaper's log before suspecting the launcher. | Our 2026-11-05 runs. |
| A process-group kill on a launcher's job took down the node's scheduler daemon. | Kill only PIDs you started, or a group you created with `setsid` and checked. | Our 2026-11-05 runs; one node lost. |
