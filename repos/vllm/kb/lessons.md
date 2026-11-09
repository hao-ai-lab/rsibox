# vLLM lessons

What misled us on vLLM and the rule that now prevents it. Edit rows in place when a rule changes;
this is not a log. Framework lessons are in `docs/lessons.md`. "The replay" below is our
2026-11-06/07 replay of 19 upstream PRs through v1 on GB200 (Qwen3-8B and Qwen3.5-35B-A3B-FP8).

## Edits a swap does not reach

| Pitfall | Rule | Evidence |
|---|---|---|
| A Triton `@jit` kernel compiles from its source text and caches per device, so swapping its Python code object changes nothing. | Route `@jit` targets out of the plain swap to component restart (or in-place kernel rebuild via `jit.py`). | vllm-project/vllm#58231, vllm-project/vllm#61213 and vllm-project/vllm#62046 went to component restart in the replay ("plain live code swap cannot rebuild Triton kernels"). |
| An edited `__init__` swaps cleanly, but the objects built at start-up keep the old state. | A constructor target (`__init__`, `__new__`, `__post_init__`) blocks with `constructor_edit`, route component_restart, overridable. | vllm-project/vllm#61323 (norm layer constructors): live code swap committed, witness 0 on every participant. Fixed in #53: the rerun on main refused the swap with `constructor_edit` and restarted the component (9.2 s eager, 44.9 s graph). Trade-off: vllm-project/vllm#62872 (a `Scheduler.__init__` edit) went from 0.38 s in place to 16.6 s / 51.2 s. |
| The graph-split decorator `eager_break_during_capture` runs once at import and during capture; captured graphs keep its old decision. | On a graph or compiled engine, refuse an edit of the decorator (or code nested in it) as `captured_target`, route component_restart (or CUDA graph recapture). | vllm-project/vllm#59941 at TP2 with CUDA graphs: live code swap committed, witness 0 on both ranks and the engine core. Fixed in #54: the rerun refused it at prepare (`captured_target`) and restarted the component. |
| A graph rebuild works idle but kills the engine under load: the rebuild empties the workers' request state. | Preempt requests before any rebuild; an idle rebuild proves nothing about serving. | Prototype GPU run, 2026-11-05 (`KeyError`, then `EngineDeadError`); listed under "Not built" in [architecture.md](../docs/architecture.md). |

## Zygote refresh

| Pitfall | Rule | Evidence |
|---|---|---|
| Re-importing an edited `layernorm` re-ran `@CustomOp.register("rms_norm")`, which refuses a known name: every forked server died ("Duplicate op name: rms_norm"). | Before the re-import, forget the refreshed modules' entries in vLLM's name registries (`NAME_REGISTRIES`). Registries outside that list still collide. | vllm-project/vllm#61323: two runs fell back to a plain restart (113 s, 195 s). Fixed in #52: the rerun forked fine (layernorm and 26 dependents refreshed). |
| An edit to almost any module refreshes most of vLLM through the `vllm.config` / `vllm.platforms` import cycle. | Expect a core edit to cost a full re-import (+1.5–2.8 s); leaf edits refresh in milliseconds. | [costs.md](costs.md), zygote section. |

## Undo and `at_base`

| Pitfall | Rule | Evidence |
|---|---|---|
| vLLM's `LazyLoader` binds modules into a namespace the first time a backend is used, so an exact undo read as "not at base". | A name the transaction did not add, whose value is a loaded module, goes to `lazy_imports` and does not make `at_base` false; our own extra names still do (`extra_names`). | vllm-project/vllm#62872, eager and graph: the only extras were two `llguidance` names. Fixed in #55: the rerun reported `at_base` true after undo, eager and graph. |

## Starting and measuring

| Pitfall | Rule | Evidence |
|---|---|---|
| Two intended cold starts used `--load-format ipc_cache` with no weight daemon; each waited 300 s and died without writing caches, so the next "warm" runs that copied those caches were cold. | Check the start that seeds a cache finished; label each run by what its caches held. | Our 2026-11-06 start-up dissection: "warm" runs at 181.9 s (eager) and 223.6 s (graph) vs 54.9 s and 59.1 s truly warm. |
| Most of a cold start is FlashInfer's JIT build, not weights. | Reuse a warm cache root in place, and time nothing until its start shows no JIT build. | Same dissection: about 121 s of JIT builds in a cold TP1 start; weights 2.6–12.9 s. |
| A copied cache root rebuilt FlashInfer's JIT kernels: its build flags hold absolute paths, so the copy is cold. | Reuse the warm root in place (lock its GPU first); a copy is warm only if every tool in it is path-independent, checked per cache. | The replay: Qwen3.5-35B-A3B-FP8 starts from copied roots took 667–763 s vs 134–231 s warm, more than 5 min of it JIT rebuild. |

## Sharing a node

| Pitfall | Rule | Evidence |
|---|---|---|
| Another server on the same port answered `/health`, so a launcher reported an engine that never bound as healthy after 0.01 s. | Check the port has no listener before launch; use a private port range on shared nodes. | Our 2026-11-05 multimodal run. |
| Engines were killed about 10.5–12 min after start, all processes in the same second. | On a cluster that reaps unallocated GPU processes, confirm the allocation before launch; read the reaper's log before suspecting the launcher. | Our 2026-11-05 runs. |
| A process-group kill on a launcher's job took down the node's scheduler daemon. | Kill only PIDs you started, or a group you created with `setsid` and checked. | Our 2026-11-05 runs; one node lost. |
