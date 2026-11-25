# Bytecode swap, kernel rebuild, and CUDA graph replay

A live-edit transaction in `core/livelab/kernel/` must update what the hardware actually runs, prove that the updated code executed on a live workload step, and restore the exact prior state on undo. In a GPU inference or training worker, a single Python function edit can sit behind three distinct execution boundaries:

1. **CPython bytecode (`FunctionType.__code__`)**: evaluated by the Python interpreter on every call unless bypassed by a JIT compiler or a recorded graph.
2. **Triton JIT kernels (`@triton.jit`, `@triton.autotune`, `@triton.heuristics`)**: compiled from source text (`JITFunction._src`) into device binaries (`cubin` / `CUfunction`), cached in memory by argument specialization, and inlined into calling Triton kernels across modules at compile time.
3. **Captured CUDA graphs (`torch.cuda.CUDAGraph`)**: recorded once by the CUDA driver during stream capture into an opaque `cudaGraphExec_t` handle that binds raw `CUfunction` pointers, grid dimensions, and pooled device addresses, then replayed from C++ (`cudaGraphLaunch`) without entering Python.

This document explains why a bytecode swap stops at the JIT and CUDA graph boundaries, how `core/livelab/kernel/jit.py` rebuilds `@triton.jit` kernels in place with sub-millisecond zero-recompile undo, how `torch.compile` (Dynamo/Inductor) interacts with Triton kernels and CUDA graphs, how the two-ledger witness (`live` vs. `capture`) in `core/livelab/kernel/witness.py` prevents false-positive proofs (with generation-matched replay witness crediting now implemented for Tier-A pointer-swapped graphs in `repos/vllm/livelab/adapter/graph_guard.py` and `pointer_swap.py`, Section 4), what selective graph recapture does today in the vLLM lab while Levels 2/3 per-subgraph and lazy recapture remain target design (Section 5), how the whole-sweep update hierarchy updates all captured shape graphs (`cuGraphExecKernelNodeSetParams` $\to$ `cuGraphExecUpdate` $\to$ piecewise/lazy recapture, Section 6 — where Tier-A surgical in-place kernel-node patching and generation-matched replay witness crediting are now implemented in the vLLM lab in `repos/vllm/livelab/adapter/pointer_swap.py` and `graph_guard.py` with automatic fail-closed fallback to Tier-C recapture, while Tier B and Levels 2/3 lazy/per-subgraph recapture remain target design), and why warm-cache fresh-process restarts across vLLM, Megatron-LM, and FastVideo still pay `30–228+ s` unless disaggregated or kept inside the live process.

---

## 1. Why bytecode swap stops at the JIT and graph boundaries

When `swap.apply` installs a standard function edit, `locate.index_module` resolves the target's `(qualname, firstlineno)` to every live `types.FunctionType` holding that code object, and `journal.set_code` replaces `fn.__code__`, `fn.__defaults__`, and `fn.__kwdefaults__` in place. Every Python reference to `fn`—the defining module's `__dict__`, `from module import fn` copies in other modules, class attributes, decorator closure cells, and saved bound-method callbacks—immediately sees the new bytecode on the next Python call.

That mechanism stops working as soon as execution crosses into a Triton `JITFunction` or a captured `CUDAGraph`:

```
Python call site
  |
  +-- [Path A: Plain Python function]
  |     fn(*args)
  |       --> reads fn.__code__ (swapped in place by journal.set_code)
  |       --> runs __rsibox_hit__(key) at top of body
  |       --> dispatches eager ops to the GPU
  |
  +-- [Path B: Triton @jit / @autotune / @heuristics callable]
  |     kernel[grid](*args)  -->  Autotuner.run(...)  -->  JITFunction.run(...)
  |       --> 1. runs JITFunction.pre_run_hooks(*args, **kw)
  |       --> 2. looks up device_caches[device][0][specialization_key]
  |       |      (keyed by argument dtypes/alignments/constexprs, NEVER by fn.__code__ or _src)
  |       |      +-- HIT:  returns cached CompiledKernel (holds loaded CUfunction / cubin)
  |       |      +-- MISS: checks on-disk cache by JITFunction.cache_key (memoised in .hash);
  |       |                if absent, parses JITFunction._src (and inlines @triton.jit callees)
  |       |                to compile a new cubin
  |       --> 3. verifies JITFunction.used_global_vals against module __globals__
  |       --> 4. launches CompiledKernel (cuLaunchKernel) on current CUDA stream
  |       (JITFunction.fn.__code__ is NEVER executed on launch)
  |
  +-- [Path C: Captured torch.cuda.CUDAGraph]
        graph.replay()
          --> calls C++ cudaGraphLaunch(cudaGraphExec_t, stream)
          --> driver replays frozen kernel nodes (raw CUfunction*, grid, pooled tensor ptrs)
          (Python, Autotuner.run, JITFunction.run, pre_run_hooks, and device_caches are NEVER entered)
```

### Object and pointer state across the three layers

`locate.module_functions` follows `.fn` and `__wrapped__` chains on wrapper objects, so a plain bytecode swap *does* find `JITFunction.fn` and updates `JITFunction.fn.__code__`. However, nothing in Triton's launch path executes `JITFunction.fn.__code__`, and nothing in `CUDAGraph.replay()` enters Python:

| Runtime object | Attribute / structure | What a plain `fn.__code__` swap does | What actually governs GPU execution |
|---|---|---|---|
| `types.FunctionType` (`fn` or `JITFunction.fn`) | `fn.__code__`, `fn.__defaults__`, `fn.__kwdefaults__` | Updated in place via `journal.set_code`; records old tuple for undo. | Governs Path A (interpreter calls). Ignored by Triton launches (Path B) and graph replays (Path C). |
| `triton.runtime.JITFunction` | `_src`, `raw_src`, `starting_line_number`, `def_file_line_number` | Untouched. Extracted once at decoration time via `inspect.getsourcelines(fn)`. | Parsed by `JITFunction.parse()` into an AST only when compiling a specialization on a cache miss. |
| `triton.runtime.JITFunction` | `hash` (memoised `cache_key`) | Untouched. | SHA-256 over `_src`, `starting_line_number`, and every referenced `@triton.jit` callee's `cache_key`. Keys Triton's on-disk binary cache. Once computed, stays non-`None` unless explicitly cleared. |
| `triton.runtime.JITFunction` | `used_global_vals` | Untouched. | Dict `{(var_name, id(globals_dict)): (val_at_compile, globals_dict)}` of module globals read by this kernel or merged from its `@triton.jit` callees. Checked on every `JITFunction.run`; raises `RuntimeError` if a tracked global changed since compile. |
| `triton.runtime.JITFunction` | `device_caches` | Untouched. | Per-device tuple `(kernel_cache_dict, ...)` mapping runtime specialization keys (argument types, alignments, `constexpr` values) to loaded `CompiledKernel` objects. Never keys on `_src` or `hash`. |
| `triton.runtime.JITFunction` | `pre_run_hooks` | Untouched. | List of callables `hook(*args, **kw)` invoked at the top of `JITFunction.run` before the `device_caches` lookup on every non-graph launch (and during stream capture). |
| `triton.runtime.Autotuner` | `fn`, `cache`, `configs`, `keys` | Untouched. | `cache` maps tuples of the kernel arguments named in `keys` to the winning `triton.Config`. `configs` and `keys` are fixed when `@triton.autotune` builds the wrapper. |
| Caller `JITFunction` in any module | Caller's `hash`, `used_global_vals`, `device_caches`, `Autotuner.cache` | Untouched. | When `@triton.jit` kernel `caller` calls `@triton.jit` helper `callee`, Triton inlines `callee`'s IR into `caller`'s `cubin` at compile time and folds `callee.cache_key` and `callee.used_global_vals` into `caller`. |
| `torch.cuda.CUDAGraph` | `cudaGraphExec_t` kernel nodes | Untouched. | Holds the raw `CUfunction` pointer, launch grid, and pooled tensor addresses recorded when the stream was capturing. Replayed directly by the CUDA driver. |

---

## 2. In-place kernel rebuild and zero-recompile undo (`core/livelab/kernel/jit.py`)

`core/livelab/kernel/jit.py` bridges Path A and Path B inside the standard two-phase transaction without importing `triton` (all Triton objects are duck-typed against `jit.NEED`), without restarting the worker process, and without recompiling or re-tuning on undo.

### Step-by-step lifecycle

```
swap.analyze (text half, pure AST / source lines)
  |
  +-- 1. Match target (qualname, firstlineno) in base_tree and patch_tree
  +-- 2. BEFORE witness.inject modifies the target AST node:
  |      jit.text_of(lines, base_node, patch_node)
  |        -> cuts clean raw_src and dedented _src from patch lines (no __rsibox_hit__ in _src)
  |        -> checks if args/returns or decorator_list AST changed (sets guard if so)
  +-- 3. witness.inject inserts __rsibox_hit__(key) into Python AST (for added non-JIT defs
         and for fn.__code__ fingerprint consistency)
  |
swap.stage (live half, phase 1 prepare while engine steps continue)
  |
  +-- 4. Detect JIT target via decorator syntax (_jit_decorator) or live holder (_jit_holder)
  +-- 5. Call jit.plan(mod, qualname, jit_text):
  |        -> jit.unwrap(obj) follows .fn up to 8 hops (Autotuner -> Heuristics -> JITFunction)
  |        -> verify all attributes in jit.NEED exist and jit_text["guard"] is None
  |           (if not: emit blocking "jit_target" finding -> routes to worker restart)
  |        -> jit.scan() walks sys.modules via object.__getattribute__(m, "__dict__")
  |        -> jit.callers_of(jf, jits) does BFS over AST of all loaded JITFunctions to find
  |           every transitive caller at any inlining depth across all loaded modules
  |        -> collect every Autotuner whose .cache wraps an affected JITFunction
  |        -> emit advisory "jit_rebuild" finding naming callers and autotuners
  +-- 6. Plan added definitions in st.pending_adds (no exec or linecache mutation in swap.stage)
  |
swap.apply (phase 2 commit inside one step gap)
  |
  +-- 7. Seed linecache for mod_file (restored via journal.on_undo) and run _build_adds:
  |        -> exec each added def inside `with JIT.seeded_lines(mod.__file__, st.source):`
  |           so inspect.getsourcelines reads the patch text, not the disk file
  |        -> collect added @triton.jit objects into st.jit_adds
  +-- 8. journal.set_code swaps jf.fn.__code__ (keeps fingerprints, line maps, and chained
  |      edits working identically to plain functions)
  +-- 9. jit.apply(plan, journal, witness, key):
  |        -> snapshot jf's (_src, raw_src, starting_line_number, def_file_line_number)
  |        -> snapshot (f, f.hash, dict(f.used_global_vals), shallow_copy(f.device_caches))
  |           for jf and every transitive caller in plan["affected"]
  |        -> snapshot (w, dict(w.cache)) for every Autotuner in plan["tuners"]
  |        -> register restore() closure via journal.record_call
  |        -> update jf._src (via _unsafe_update_src), raw_src, starting_line_number, def_file_line_number
  |        -> call jit.invalidate(f) on every f in plan["affected"]:
  |             f.hash = None
  |             f.used_global_vals.clear()
  |             f.device_caches.clear()
  |        -> clear w.cache.clear() on every w in plan["tuners"]
  |        -> _hook_into arms witness.arm and appends witness.arm_launch(key, label(f), journal)
  |           (falling back to witness.arm if absent) to f.pre_run_hooks on every f in plan["affected"]
  +-- 10. jit.arm_added(..., added=...) attaches deduplicated pre_run_hooks for newly added
         @triton.jit helpers, matching rebuilt callers (re.findall(r"\w+", ...)) and added callers
```

### 1. Slicing clean source text before witness injection (`jit.text_of`)

For plain Python functions, `swap.analyze` calls `witness.inject(node, key)` to insert `__rsibox_hit__("<module>:<qualname>:<line>")` as the first statement of the function body, then compiles the modified AST into bytecode.

A Triton kernel cannot execute `__rsibox_hit__`: Triton's compiler parses `JITFunction._src` into Triton IR and rejects unknown Python calls. In `swap._analyze_trees`, `JIT.text_of(lines, bq[q][i], node)` therefore runs **before** `_inject(node, k, undo)`:

- It slices `raw = lines[start - 1 : node.end_lineno]` directly from the clean patched source text, starting at `node.decorator_list[0].lineno` (matching `inspect.getsourcelines`).
- It dedents `"".join(raw)` and strips the leading decorator lines down to `^def\s+\w+\s*\(` to form `src` (matching how `JITFunction.__init__` builds `_src`), while preserving both `start` (the first decorator line, stored in `starting_line_number`) and `def_line = node.lineno` (stored in `def_file_line_number`) so compiler diagnostics report exact line numbers.
- `_inject(node, k, undo)` still runs afterward on the Python AST so `jf.fn.__code__` receives the standard bytecode swap for `Session` fingerprinting, chained-edit `own_codes` tracking, and `base_mismatch` checks.

### 2. Why parameter and decorator changes block with `jit_target`

When `@triton.jit`, `@triton.autotune`, or `@triton.heuristics` decorates a function at import time, Triton inspects the signature and decorator arguments once and freezes derived metadata onto the wrapper instances:

- `JITFunction.__init__` builds `self.params`, `self.signature`, `constexpr` indices, `do_not_specialize` indices, and specialization key extractors from the function's parameter list and type annotations.
- `Autotuner.__init__` and `Heuristics.__init__` bind `self.configs`, `self.keys` (resolved to parameter indices), `prune_configs_by`, and heuristic callables on the outer wrapper objects.

Updating `_src` in place does not rebuild those parameter descriptors or wrapper objects: launching after a parameter or `@triton.autotune` key edit would bind arguments or index tuning keys using the old signature layout. `jit.text_of` therefore compares `ast.dump(base.args)`, `ast.dump(base.returns)`, and `ast.dump(ast.Tuple(base.decorator_list))`. Any difference sets `text["guard"]`, causing `jit.plan` to refuse the in-place rebuild and `swap.stage` to emit a blocking `jit_target` finding routed to a worker restart. The same refusal applies if the live wrapper lacks any attribute in `jit.NEED`:

```python
NEED = (
    "_unsafe_update_src", "device_caches", "pre_run_hooks", "parse",
    "signature", "raw_src", "starting_line_number", "used_global_vals",
)
```

### 3. Safe namespace sweep and transitive caller invalidation (`jit.scan` and `jit.callers_of`)

A `@triton.jit` helper function (for example, an activation or block-reduction helper in module A) is often called inside a top-level `@triton.autotune @triton.jit` kernel in module B. Because Triton inlines callees at compile time:

1. Rebuilding the helper in module A alone leaves module B's caller holding its old `CompiledKernel` in `device_caches` and its old `cache_key` in `hash`.
2. Even if module B's `device_caches` were cleared without clearing `hash`, `caller.cache_key` would return the memoised `caller.hash` and reload the pre-edit binary from Triton's on-disk cache.
3. In Triton 3.5+, `caller.cache_key` merges `callee.used_global_vals` into `caller.used_global_vals` before hashing `callee.cache_key`. If an edit removes a global constant `SCALE` from `callee` and `SCALE` is later changed in module A, leaving the stale `(SCALE, id(mod_a.__dict__))` entry in `caller.used_global_vals` causes `caller.run` to raise `RuntimeError: Global variable SCALE has changed since compile`.

`jit.plan` finds and resets the full transitive caller cone across all loaded modules:

- **No lazy-import side effects (`_dict_of`)**: Many ML packages install module-level `__getattr__` hooks (`LazyLoader`) that import heavy submodules or initialize device state on attribute access. `jit.scan()` reads `object.__getattribute__(mod, "__dict__")` on each module in `list(sys.modules.values())`, inspecting only already-bound dictionary values filtered by a per-type cache `_jit_like(type(v))`.
- **Unwrapping wrapper chains (`jit.unwrap`)**: Follows `.fn` through up to 8 wrapper hops (`Autotuner -> Heuristics -> JITFunction`), recording both the leaf `JITFunction` and any outer wrapper whose `cache` attribute is a `dict`.
- **Breadth-first AST resolution (`jit.callers_of`, `jit.named`, and `jit.calls`)**: Starting from `[jit_fn]`, `callers_of` inspects every discovered `JITFunction` `candidate`. `named(candidate)` parses `candidate.parse()` directly—with no callee-name substring pre-filter, so import aliases (`from a import helper as act`) and module attributes (`acts.act_fn`) still match—and resolves every `ast.Name` and dotted `ast.Attribute` load (`ast.Load`) against `candidate.__globals__` merged with `candidate.fn.__closure__` cells, returning `{id(v), id(unwrap(v)[0]), ...}`. `callers_of` caches `named(candidate)` once per candidate across the BFS queue and fails closed by raising `jit.UnreadableKernel` (making `jit.plan` refuse rather than silently skip an unparseable candidate); `calls(caller, callee)` checks `id(callee) in named(caller)`. Whenever `id(current)` appears in a candidate's resolved ID set, that candidate is added to `out` and queued so callers of callers at arbitrary depth are included.

### 4. Snapshot and sub-millisecond zero-recompile undo (`jit.apply`)

On commit, `jit.apply` snapshots the live state of `jf`, every kernel in `affected = [jf] + callers`, and every wrapping `Autotuner` in `tuners` before mutating anything:

```python
old = (jf._src, list(jf.raw_src), jf.starting_line_number,
       getattr(jf, "def_file_line_number", None))
states = [(f, f.hash, dict(f.used_global_vals), _caches(f.device_caches))
          for f in p["affected"]]
tuned = [(w, dict(w.cache)) for w in p["tuners"]]
```

Here `_caches(dc)` copies the per-device dictionary shells `(dict(v[0]), dict(v[1])) + tuple(v[2:])` while keeping references to the already-compiled `CompiledKernel` objects (`cubin` handles loaded in GPU driver memory).

When `uninstall` runs `Journal.undo_to(mark)`, the journaled `restore()` closure executes in reverse transaction order:

1. Restores `jf._src`, `jf.raw_src`, `jf.starting_line_number`, and `jf.def_file_line_number`.
2. Restores each affected kernel's memoised `hash`, `used_global_vals`, and `device_caches` dictionary entries pointing to the original `CompiledKernel` objects.
3. Restores each `Autotuner.cache` dictionary mapping tuning keys to the previously benchmarked `Config` objects.
4. Removes the transaction's witness hooks from `pre_run_hooks`.

Because the original `CompiledKernel` objects and autotuner `Config` choices were preserved in memory inside the closure, undo performs only Python dictionary and attribute restores (`< 1 ms`, zero Triton compiler invocations, zero disk-cache reads, and zero autotuner benchmarking runs).

### 5. Newly added `@triton.jit` helpers (`jit.seeded_lines` and `jit.arm_added`)

When a patch introduces a brand-new `@triton.jit` function that does not exist in the file on disk, two things must happen:

1. **Source lookup during `exec` (`jit.seeded_lines`)**: Executing `@triton.jit def bump(v): ...` calls `inspect.getsourcelines(bump)`, which looks up `mod.__file__` in Python's `linecache`. Without intervention, `linecache` reads the unedited disk file and either slices the wrong lines or raises `OSError` if the patch is longer than the file. `swap.stage` leaves `linecache` and module namespaces untouched (`pending_adds` only); at commit time, `swap.apply` seeds `linecache` (with `journal.on_undo` restore) and invokes `_build_adds`, which wraps each scratch `exec` in `with JIT.seeded_lines(mod.__file__, st.source):`, temporarily setting `linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)` and restoring the previous entry on exit.
2. **Witnessing an inlined added helper (`jit.arm_added`)**: Because the added function is a `@triton.jit` callable, `swap.analyze` does not inject `__rsibox_hit__` into its body (`not _jit_decorator(n)`). At commit, `swap.apply` calls `jit.arm_added(obj, key, plans, journal, witness, added=[...])`. Starting with `jf`, it collects every `f` in `p["affected"]` for each plan `p` in `plans` whose new source `p["text"]["src"]` contains `jf.fn.__name__` as a word token (`re.findall(r"\w+", ...)`), plus transitive callers from `callers_of(jf, others)` among the other `@triton.jit` helpers in `added`, deduplicating by `id(f)`. `_hook_into` then registers `witness.arm(key, journal)` and appends a per-kernel `witness.arm_launch(key, label(f), journal)` hook (falling back to the shared `witness.arm` closure if `witness` has no `arm_launch`) to each kernel's `pre_run_hooks`, ensuring a single launch increments each key once and attributes proof to the launched kernel.

### Failure modes prevented by `kernel/jit.py`

| Failure mode | Root cause if unhandled | How `kernel/jit.py` and `swap.py` prevent it |
|---|---|---|
| Triton compile error on `__rsibox_hit__` | AST witness injection runs before source extraction, polluting `JITFunction._src`. | `jit.text_of` slices clean source lines from `source.splitlines(keepends=True)` before `witness.inject` mutates the AST; `jit.arm_added` skips AST injection on added `@triton.jit` defs. |
| Leaf kernel edit commits but GPU runs old `cubin` | `JITFunction.fn.__code__` is swapped, but `device_caches` still maps the specialization tuple to the old `CompiledKernel`. | `jit.invalidate` clears `f.device_caches`, `f.hash`, and `f.used_global_vals`, and clears wrapping `Autotuner.cache` dicts. |
| Edited `@triton.jit` helper leaves cross-module caller on old binary | Caller kernel inlined the old helper at compile time; clearing only the helper leaves the caller's `device_caches` and `hash` intact (or reloads the old binary from the disk cache). | `jit.scan` + `jit.callers_of` walks the AST of every loaded `JITFunction` across `sys.modules` and invalidates all transitive callers at any depth. |
| Rebinding a removed global raises `RuntimeError` in caller | Triton 3.5+ merges callee `used_global_vals` into the caller before computing `cache_key`; stale global entries survive on the caller. | `jit.invalidate` clears `used_global_vals` on both the target kernel and every transitive caller. |
| Edited `@triton.jit` helper reports `zero_witness` despite running | Inlined helpers never execute `JITFunction.run` at launch; only the outer kernel's `pre_run_hooks` run. | `_hook_into` attaches the helper's witness hook to `pre_run_hooks` of the helper and of every transitive caller in `affected`. |
| Added `@triton.jit` helper reads wrong source lines or raises `OSError` | `inspect.getsourcelines` reads `mod.__file__` from disk during `_build_adds` in `swap.apply`. | `jit.seeded_lines` seeds `linecache.cache[mod.__file__]` with the patched text for the duration of the scratch `exec`. |
| Parameter or `@triton.autotune` edit launches with stale signature/keys | Triton fixes `params`, `constexpr` indices, and autotuner `configs`/`keys` when constructing the wrapper objects. | `jit.text_of` compares parameter, return, and decorator ASTs and refuses the in-place rebuild with a blocking `jit_target` finding. |
| Undo pays seconds of recompilation and re-autotuning | Naively re-running `jit.invalidate` on undo discards compiled kernels and benchmark winners. | `jit.apply` snapshots `device_caches` and `Autotuner.cache` before invalidation and restores the live `CompiledKernel` and `Config` objects via `journal.record_call`. |

---

## 3. Why CUDA graphs freeze compiled kernels (capture vs. replay and Inductor orthogonality)

Rebuilding a `JITFunction` in place ensures that the **next call to `JITFunction.run`** compiles and launches the new `cubin`. Whether the engine's next step actually calls `JITFunction.run` depends on whether the call site runs eagerly, under `torch.compile`, or inside a captured `torch.cuda.CUDAGraph`.

### Capture vs. replay execution model

A `torch.cuda.CUDAGraph` separates execution into two phases with completely different software stacks:

```
PHASE A: CAPTURE (runs once per graph / bucket)
  with torch.cuda.graph(cudagraph, pool=shared_pool, stream=capture_stream):
      out = model_or_piece(static_inputs)
        |
        +--> Python frames execute (or Inductor compiled FX wrapper executes)
        +--> JITFunction.run(*args) executes on CPU:
               1. fires pre_run_hooks  (witness sees capture_stream is capturing -> capture_counts[k] += 1)
               2. compiles new _src on device_caches miss -> loads new CUfunction
               3. calls cuLaunchKernel(new_CUfunction, grid, block, kernel_args_with_static_ptrs, capture_stream)
        +--> CUDA driver intercepts cuLaunchKernel on capture_stream:
               records a cudaGraphNode_t {CUfunction*, grid, block, static device pointers}
               (does NOT execute on the GPU during capture)
  cudaGraphInstantiate(&cudaGraphExec_t, cudaGraph_t)  --> freezes all kernel nodes

PHASE B: REPLAY (runs on every matching live step)
  static_inputs.copy_(step_inputs)
  cudagraph.replay()
    |
    +--> C++ / CUDA driver calls cudaGraphLaunch(cudaGraphExec_t, stream)
    +--> GPU hardware dispatches the recorded cudaGraphNode_t sequence directly
    +--> ZERO Python frames run; JITFunction.run is NOT called;
         JITFunction.device_caches is NOT read; pre_run_hooks do NOT fire.
```

Because `cudaGraphExec_t` holds the raw `CUfunction` pointer and launch parameters that were passed to `cuLaunchKernel` at capture time, mutating `JITFunction._src` and clearing `JITFunction.device_caches` in Python has **no effect on an already-instantiated `CUDAGraph`**. Until that `CUDAGraph` is either patched in place (Tier A, Section 6) or recaptured (Tier C), every `cudagraph.replay()` continues launching the pre-edit `CUfunction` binary on the GPU.

### Why `torch.compile` (Inductor) and CUDA graph capture are orthogonal for `@triton.jit` kernels

Engines frequently combine `torch.compile` (Dynamo bytecode tracing + Inductor code generation) with CUDA graphs. For live editing, it is critical to distinguish **Inductor-generated kernels** (synthesized by Inductor from traced PyTorch tensor ops) from **user-written `@triton.jit` kernels** (invoked directly or through custom ops):

```
Traced Python model forward
  |
  +-- [PyTorch ops: x * torch.sigmoid(x)]
  |     Dynamo traces bytecode into FX graph
  |       --> Inductor fuses ops and generates a synthetic Triton kernel file on disk
  |       --> Editing the Python math requires Dynamo/Inductor FX recompilation
  |
  +-- [User @triton.jit kernel or custom op wrapping @triton.jit]
        Dynamo treats the custom op / Triton kernel as an opaque call node in the FX graph
          --> Inductor's generated Python wrapper calls `my_triton_kernel[grid](...)`
              (which invokes the live Python `Autotuner` / `JITFunction` object)
          --> Editing `my_triton_kernel`'s body requires:
                * Without CUDA graphs: ONLY `jit.apply` (Inductor wrapper calls `JITFunction.run`
                  on the next step, which compiles the new `_src` immediately; 0 FX recompiles)
                * With CUDA graphs: `jit.apply` + CUDA graph recapture (re-running the existing
                  compiled Inductor wrapper under `torch.cuda.graph(...)` calls `JITFunction.run`
                  and records the new `CUfunction*` into `cudaGraphExec_t`; 0 FX recompiles)
```

| Edit target | Engine execution mode | What must happen in process after `swap.apply` | Dynamo / Inductor FX recompile needed? | CUDA graph recapture needed? |
|---|---|---|---|---|
| Plain Python function outside `torch.compile` and outside `CUDAGraph` (scheduler, sampler, split op) | Any | Nothing beyond `journal.set_code`: next step runs new `fn.__code__`. | No | No |
| User `@triton.jit` kernel or helper called outside `CUDAGraph` (eager mode, post-model sampler, or split op) | Eager or `torch.compile` without graph capture over that op | `jit.apply` updates `_src` and clears `device_caches` / `hash` / `used_global_vals`; next launch compiles new `cubin`. | No (Inductor wrapper calls live `JITFunction`) | No |
| User `@triton.jit` kernel or helper called inside a captured `CUDAGraph` | Eager + `CUDAGraph`, or `torch.compile` + `CUDAGraph` | `jit.apply` rebuilds the `JITFunction`, then the affected `CUDAGraph` entries are either updated in place via Tier-A `cuGraphExecKernelNodeSetParams` (implemented in vLLM, Section 6) or recaptured by re-running the wrapper under stream capture. | **No** (existing Inductor FX wrapper calls `JITFunction.run` during recapture; Tier A skips forward passes) | **Yes, or Tier-A node patch** (only affected subgraphs / buckets) |
| Plain Python model method traced by Dynamo into fused Inductor ops | `torch.compile` (+ `CUDAGraph`) | Invalidate the compiled frame / subgraph so Dynamo + Inductor re-trace and re-emit the fused kernel, then recapture affected `CUDAGraph` entries. | **Yes** | **Yes** (if captured in `CUDAGraph`) |
| `@triton.jit` parameter/decorator edit, constructor edit (`__init__`), or new instance field (`new_fields`) | Any | Blocked at `prepare` (`jit_target`, `constructor_edit`, `new_fields`); routed to an in-place object rebuild or worker restart. | N/A (refused before commit) | N/A |

### Shared `graph_pool` memory reuse across recapture cycles

During capture, any intermediate tensor allocated inside `with torch.cuda.graph(g, pool=pool):` is placed in a private CUDA caching-allocator pool identified by `pool` (`torch.cuda.graph_pool_handle()`). Those virtual addresses are baked into `cudaGraphExec_t` and remain reserved for the lifetime of the graph so live steps cannot overwrite them.

When recapturing graphs in place after a kernel rebuild:

1. **Release before or on replacement**: Calling `old_graph.reset()` (or dropping the old `CUDAGraph` reference before capturing its replacement into the same `pool`) returns the old graph's private pool blocks to the shared `graph_pool` free list without calling `cudaFree`.
2. **Reuse the same `pool` handle**: Passing the engine's existing `pool=shared_graph_pool` to the new `torch.cuda.graph(...)` capture re-allocates identical tensor shapes from the freed blocks in the pool.
3. **Zero steady-state memory growth**: When old graphs are reset before re-recording and buckets reuse the shared `graph_pool` (capturing largest-first when building multiple buckets from scratch, or replacing a single bucket in place), repeated edit $\to$ recapture $\to$ prove $\to$ undo cycles incur `0 MiB` of net allocated or reserved GPU memory growth. Conversely, allocating a fresh pool handle per recapture or leaving old `CUDAGraph` objects referenced retains the entire per-bucket activation footprint on every edit (`docs/lessons.md`, lessons 12 and 14).

---

## 4. Two-ledger launch witness (`live` vs. `capture`) — zero false positives

A core invariant of the transaction kernel (`core/livelab/kernel/README.md`) is that **a committed edit that never ran on a live workload step is never reported as proven**. With `@triton.jit` kernels and CUDA graphs, two opposite failure modes threaten a naive single-counter witness:

1. **False positive from unrecaptured graph replay**: An edit to a `@triton.jit` kernel commits, but the engine keeps replaying a `CUDAGraph` captured before the edit. The GPU runs the old `cubin`. If commit alone counted as proof, the transaction would pass while testing the old kernel.
2. **False positive from capture-only execution**: After the edit commits, the engine recaptures its CUDA graphs. During recapture, Python runs warm-up iterations and the stream-capture pass, firing `__rsibox_hit__` and `JITFunction.pre_run_hooks` thousands of times (for example, `13,280` calls per rank across layers and batch sizes in `test_witness.py` `CaptureWitnessTest`). If capture-time calls incremented the live witness counter, `prove` would report thousands of hits even when the live verification workload never replayed any captured graph.

### How `Witness` separates `counts` (`live`) and `capture_counts` (`capture`)

`core/livelab/kernel/witness.py` maintains two separate per-key dictionaries on each participant's `Witness` instance:

```python
class Witness:
    def __init__(self, capturing=None):
        self.counts = {}            # live ledger: calls during real workload steps
        self.capture_counts = {}    # capture ledger: calls while recording/warming up graphs
        self.capturing = capturing  # engine probe fn() -> bool (e.g. stream is capturing)
        self._capture = 0           # nesting depth of `with witness.capture():`
```

Every witness hit—whether from a swapped Python function calling `__rsibox_hit__(key)` or from a rebuilt Triton kernel firing `f.pre_run_hooks` (armed via `witness.arm` / `witness.arm_launch`)—calls `witness.hit(key)`:

```python
def _capturing(self):
    if self._capture:
        return True
    if self.capturing is None:
        return False
    try:
        return bool(self.capturing())
    except Exception:
        return False

def hit(self, key):
    c = self.capture_counts if self._capturing() else self.counts
    c[key] = c.get(key, 0) + 1
```

Two mechanisms route calls into `capture_counts`:

1. **Stream-level probe (`Witness(capturing=fn)`)**: The participant passes a probe such as `torch.cuda.is_current_stream_capturing`. Any call made while the CUDA stream is actively recording a graph increments `capture_counts[key]`.
2. **Recapture context (`with witness.capture():`)**: A CUDA graph recapture routine first runs 1–3 eager warm-up passes (to settle allocator addresses and trigger Triton compilation / autotuning) before entering `with torch.cuda.graph(...):`. Because those warm-up passes run on a non-capturing stream, the stream probe alone would misclassify warm-up launches as live workload steps. Wrapping the entire recapture routine (warm-ups + stream capture) in `with witness.capture():` increments `self._capture`, routing every warm-up compilation, autotuner benchmark, and stream-capture launch into `capture_counts`.

When `torch` is loaded, `witness.trace_safe(fn)` marks both `witness.hook` and `witness.stub` with `torch._dynamo.assume_constant_result`. When Dynamo traces a function containing `__rsibox_hit__(key)`, it executes the hook once during tracing (which lands in `capture_counts` when tracing runs inside `with witness.capture():`) and emits no counter mutation into the compiled FX graph.

### Generation-matched graph-replay witness (implemented for vLLM Tier A; target design for Tier C / other engines)

**Status: implemented for vLLM Tier-A pointer-swapped graphs; target design for Tier-C recapture and other engines.** The vLLM lab now implements generation-matched replay witness crediting for Tier-A pointer-swapped graphs via `GraphReplayGuard.register_replay_witness` and `GraphReplayGuard.check_before_replay` (`repos/vllm/livelab/adapter/graph_guard.py`, `repos/vllm/livelab/adapter/pointer_swap.py`). `GraphReplayGuard` tracks a monotonic `code_epoch` and per-mode `pool_epoch`, stamps each graph entry with `(code_epoch, mode_pool_epoch, input_sig)`, fails closed with `StaleGraphReplayError` on mismatch, and credits `witness.hit(key)` on the live ledger when a generation-matched Tier-A patched graph replays. Tier-C recapture (`repos/vllm/livelab/adapter/recapture.py`) and other engines remain uncredited on replay: a target that runs only inside a Tier-C recaptured graph still ends with `capture > 0`, `live == 0` and reads `zero_witness` with the `DEVICE_PROBE` note (`core/livelab/kernel/witness.py`, `Witness.check`). The diagram below illustrates the generation-matched bridge.

Once a post-commit `CUDAGraph` has been captured or Tier-A patched inside `with witness.capture():`, `capture_counts[key] > 0` proves that the new code or rebuilt `cubin` was recorded into that graph entry, while `counts[key]` is still `0`. Because `cudagraph.replay()` runs in C++/driver without firing Python hooks or `pre_run_hooks`, proving that a live step actually executed the updated graph requires a replay-time bridge that cannot be fooled by stale graphs or unused buckets:

```
Recapture of bucket B at transaction generation G (inside `with witness.capture():`):
  1. snap_before = dict(witness.capture_counts)
  2. run warm-ups + `with torch.cuda.graph(entry.graph, pool=shared_pool): forward(static_in)`
     (JITFunction.run compiles new _src and fires pre_run_hooks -> capture_counts[k] increases)
  3. entry.captured_keys = {k for k, v in witness.capture_counts.items()
                            if v > snap_before.get(k, 0)}
  4. entry.txn_generation = G

Live workload step dispatching bucket B (outside `with witness.capture():`, _capturing() == False):
  1. entry = graph_table[B]
  2. entry.graph.replay()
  3. if entry.txn_generation == current_txn_generation:
         for k in entry.captured_keys:
             witness.hit(k)   # _capturing() is False -> increments witness.counts[k] (live ledger)
```

Why the design needs both `entry.txn_generation == current_txn_generation` and `entry.captured_keys`:

- **Unrecaptured graph (`entry.txn_generation < current_txn_generation`)**: The graph entry was captured before the current transaction committed, so its `cudaGraphExec_t` still holds the old `CUfunction` pointer. Its generation does not match `current_txn_generation`, so replaying it credits **0** to `witness.counts`.
- **Recaptured bucket that never replays**: Bucket $B_{32}$ is recaptured under `with witness.capture():` (`capture_counts[k] > 0`, `entry.txn_generation == current_txn_generation`), but the live verification request runs at batch size $B_1$. Bucket $B_{32}$'s `replay()` is never called during the live step, so `witness.counts[k]` remains **0**.
- **Device-side atomic probe (alternative hardware witness, not implemented)**: `witness.DEVICE_PROBE` is only the text of the note attached to the finding; no lab installs such a counter. A kernel or graph could also increment a pinned/device tensor counter `d_hits[key_id] += 1` inside the captured GPU kernel node itself; reading `d_hits` after `replay()` would observe the hardware execution directly.

### Complete `(capture_count, live_count)` state matrix

When the workload finishes, `Participant.witness_check(txn)` calls `Witness.check(keys, old_codes)`. A transaction is `proven: True` if and only if **every** witness key armed by the transaction has `counts[key] > 0` (`live > 0`) on every participant that committed it:

| `capture` (`capture_counts[k]`) | `live` (`counts[k]`) | Old frame in `live_codes`? | Physical scenario | `Witness.check` result (`ok` / `proven` and finding) |
|---:|---:|---|---|---|
| `0` | `0` | **Yes** (on a thread stack or in a suspended generator / coroutine) | A long-running loop or suspended coroutine entered the function before the swap and is still executing the old code object. | **`proven: False`** — finding `running_frame` (`"0 calls; a frame of the old code is still running"`). |
| `0` | `0` | No | **Case 1 (Eager dead path):** The workload never called the function or launched the kernel.<br>**Case 2 (Unrecaptured CUDA graph):** The workload replayed a `CUDAGraph` captured before the edit; the GPU ran the old `cubin` and bypassed `pre_run_hooks`. | **`proven: False`** — finding `zero_witness` (`"0 calls since the swap"`, expected `"at least one call"`). |
| `> 0` | `0` | No | The edited function or rebuilt `@triton.jit` kernel ran `capture` times while warming up, capturing, or Tier-A patching CUDA graphs (`with witness.capture():` or stream capturing), so the new code/kernel is inside the graph, **but no live execution credited `counts`**. For Tier-C recapture and other engines (uncredited replay), a graph-only target stays in this row; for vLLM Tier-A pointer-swapped graphs, a target stays here only if no patched bucket replayed during the live workload. | **`proven: False`** — finding `zero_witness` (`"0 live calls since the swap; <cap> calls while graphs were captured (a replay runs no Python)"`), with the `DEVICE_PROBE` note and route `R1`. |
| `0` | `> 0` | No | The swapped Python function or rebuilt `@triton.jit` kernel ran `live` times eagerly during live workload steps outside CUDA graph capture (e.g., eager execution, sampler kernel, or graph-splitting op). | **`proven: True`** — `findings: []`, `ran[k] == "ran during capture 0, ran live <live>"`. |
| `> 0` | `> 0` | No | The target was captured or Tier-A patched into a post-commit graph (`capture > 0` during `witness.capture()`), **and** at least one live workload step either called it in Python (for example a mixed eager prefill step) or replayed a generation-matched Tier-A patched graph credited via `GraphReplayGuard.check_before_replay`. Crediting Tier-C recaptured graph replays or a device-side counter remains target design. | **`proven: True`** — `findings: []`, `ran[k] == "ran during capture <cap>, ran live <live>"`. |

---

## 5. Selective and lazy graph recapture (three-level pruning; Levels 2 and 3 are target design)

**Status.** Implemented, in the vLLM lab only:

- Level 1: `repos/vllm/livelab/adapter/graphs.py` classifies a target as split, captured or unknown, and `repos/vllm/livelab/adapter/side.py` turns that into `captured_target` / `captured_partly` at prepare.
- A coarser form of selection in `repos/vllm/livelab/adapter/recapture.py` (`Recapture.run`, `Recapture.extend`): it recaptures per cudagraph mode (`modes`: PIECEWISE, FULL) and per batch descriptor (`sizes`), eagerly in the step gap of the commit, under `with witness.capture():`. Descriptors it did not choose are removed from the dispatcher keys, so those steps run eagerly on the new code until `extend` captures them. Undo puts the base entry dicts and keys back and captures nothing.

Target design, not implemented in any lab: the per-subgraph dependency map of Level 2 (all pieces of a chosen descriptor are recaptured together), and the on-dispatch lazy recapture and Tier-C replay crediting of Level 3 (`GraphReplayGuard` in `repos/vllm/livelab/adapter/graph_guard.py` implements `(code_epoch, mode_pool_epoch, input_sig)` epoch stamps and replay witness crediting for Tier-A pointer-swapped graphs, Section 6). The FastVideo and Megatron labs have no graph recapture. The times in this section's diagram and cost table are estimates, not measurements; the measured recapture costs are in `repos/vllm/docs/architecture.md` (3.0-5.6 s per region and 14.4 s for all graphs, Qwen3-8B TP1).

In a production engine, full CUDA graph capture sweeps across two dimensions:

- **$P$ piecewise subgraphs**: When compilation splits the model at dynamic or communication ops (for example, attention or MoE all-to-all ops configured as graph-splitting boundaries), an $L$-layer model is partitioned into $P = O(L)$ piecewise subgraph callables $G_1, G_2, \dots, G_P$ (plus an optional full-model decode graph).
- **$B$ batch-shape buckets**: Each graph-backed callable captures a separate `CUDAGraph` for every padded batch size or token-count bucket $b \in \{b_1, b_2, \dots, b_B\}$ (typically $30\text{--}64$ buckets per rank).

Re-running warm-up and stream capture across all $P \times B$ entries after a single function or kernel edit takes `5–20 s` per worker rank—dominating edit-to-run latency when the bytecode swap or `jit.apply` itself took only `10–80 ms`.

The design has three orthogonal pruning levels that remove unnecessary recaptures while keeping the two-ledger witness guarantee:

```
Committed edit (swap.apply + jit.apply in < 80 ms)
  |
  +-- [Level 1: Graph vs. Eager Boundary]
  |     Does the edited target execute inside any captured CUDAGraph on this workload?
  |       +-- NO (scheduler, sampler, eager-only op, or split-op tree in piecewise mode)
  |       |     --> 0 graphs invalidated, 0 recaptures. Next step runs new code immediately.
  |       +-- YES
  |             |
  +-- [Level 2: Piecewise Subgraph Isolation]
  |     Which subgraph wrappers G_i in {G_1 .. G_P} called the target (or its affected
  |     @triton.jit callers) during capture?
  |       +-- Untouched subgraphs G_j: keep existing cudaGraphExec_t handles warm (0 work)
  |       +-- Dirty subgraphs G_i: mark bucket table stale (generation bump / reset entries)
  |             |
  +-- [Level 3: Active-Bucket Warmup + On-Demand Lazy Bucket Capture]
        Which batch-shape buckets b in {b_1 .. b_B} of dirty subgraph G_i are needed now?
          +-- Warm up only active bucket(s) b_active (0 or 1 bucket) under `with witness.capture():`
          +-- Leave remaining buckets b != b_active uncaptured (stale generation)
          +-- On live step with bucket b:
                if entry.txn_generation < current_txn_generation:
                    with witness.capture():
                        recapture G_i[b] in shared graph_pool, record captured_keys, stamp generation
                replay G_i[b] and credit witness.counts for entry.captured_keys
```

### Level 1 — Graph vs. eager boundary pruning (`0` recaptures)

Before touching any graph table, the worker's adapter classifies where the edited target executes (implemented for vLLM as static intra-module reachability in `repos/vllm/livelab/adapter/graphs.py`):

1. **Outside the captured model forward pass**: Code in the engine core (scheduler, block manager), request pre/post-processing, and worker post-model steps (such as token-sampling kernels that run on logits after the model's captured forward pass returns) never enters `torch.cuda.graph(...)`. Rebuilding a `@triton.jit` kernel there via `jit.apply` takes effect on the very next step with **0 graph recaptures**.
2. **Graph-splitting ops**: In piecewise graph modes, custom ops that split the graph (such as attention or MoE dispatch ops listed in the engine's compilation config or marked with a graph-break decorator) run eagerly in Python between captured piecewise segments on every step. Static intra-module call-graph reachability from those split-op roots classifies a target into one of three regimes:
   - **Piecewise on all steps**: When both prefill and decode split the graph around those ops, a target reached exclusively through split-op roots runs in Python on every step (**0 recaptures**).
   - **Piecewise on prefill, full graph on decode (`captured_partly`)**: When prefill and mixed batches run piecewise while decode-only batches replay a single full graph, a target reached through split ops runs in Python on prefill immediately and inside the full graph on decode (prefill proves the edit with 0 recaptures; running the edit on decode steps requires recapturing the decode graph).
   - **Decode-only graphs (`captured_partly` / eager prefill)**: When the engine captures CUDA graphs for decode-only steps and runs prefill eagerly without compilation, prefill steps execute the new code with **0 recaptures**, while decode steps require graph recapture.
3. **Import- and decoration-time graph machinery (`captured_target`)**: A target that decides *how* ops split graphs when decorators run at import time cannot take effect via either bytecode swap or graph recapture alone, and is blocked at `prepare` with `captured_target`.

### Level 2 — Piecewise subgraph isolation ($O(1/P)$ of subgraphs; target design)

In piecewise graph modes, each segment $G_i \;(i \in \{1, \dots, P\})$ between graph-splitting ops has its own wrapper and its own `batch_descriptor -> CUDAGraphEntry` lookup table, all sharing the worker's `graph_pool`.

When $G_i$ is captured for the first time (or recaptured), wrapping the capture of $G_i$ in a per-subgraph dependency tracker (or recording which `JITFunction` objects executed `pre_run_hooks` and which modules/qualnames ran during $G_i$'s capture) maps each target and each `JITFunction` in `plan["affected"]` to the subset of subgraphs $\mathcal{D} \subseteq \{G_1, \dots, G_P\}$ that invoke it:

- **Layer-type or head-only edits**: An edit to a final norm kernel, an embedding kernel, a vision-encoder kernel, or a layer kind present in only $k \ll P$ layers marks only those $k$ subgraph wrappers dirty. The remaining $P - k$ subgraph wrappers keep their `cudaGraphExec_t` instances untouched.
- **Uniform transformer-layer edits**: An edit to a kernel called in every transformer block (such as a per-layer residual+RMSNorm or MLP activation kernel) marks all $P$ layer subgraphs dirty, passing the work reduction to Level 3.

### Level 3 — Active-bucket warmup and on-demand lazy bucket capture ($O(1/B)$ of buckets; target design)

Even when all $P$ subgraphs are dirty, a live verification workload or single benchmarking step typically exercises only **1 to 2 batch-size buckets** (for example, $b = 1$ for single-request latency or $b = 32$ for a fixed-concurrency check) out of the $B = 32\text{--}64$ buckets pre-captured at engine startup.

Instead of eagerly recapturing all $B$ buckets across every dirty subgraph $G_i \in \mathcal{D}$:

1. **$O(1)$ invalidation at commit**: Increment the worker's `current_txn_generation`. For each dirty subgraph $G_i \in \mathcal{D}$, mark its cached bucket entries stale (`entry.txn_generation < current_txn_generation`).
2. **Optional active-bucket warmup**: If the host or workload specifies the active batch bucket $b_{\text{active}}$ (or the most recently dispatched bucket), eagerly recapture only $G_i[b_{\text{active}}]$ for $G_i \in \mathcal{D}$ under `with witness.capture():`.
3. **Lazy on-demand recapture on step dispatch**: When a live step dispatches $G_i$ at batch bucket $b$:
   - If `entry.txn_generation == current_txn_generation`, call `entry.graph.replay()` directly and credit `witness.counts` for `entry.captured_keys`.
   - If `entry.txn_generation < current_txn_generation`:
     1. Release the stale `entry.graph.reset()` back to the shared `graph_pool`.
     2. Enter `with witness.capture():` so warm-up runs, Triton compilation (on the first bucket only; subsequent buckets hit `JITFunction.device_caches` unless `constexpr` specialization differs), and stream capture all record into `witness.capture_counts`.
     3. Record `entry.captured_keys` from the `capture_counts` delta and stamp `entry.txn_generation = current_txn_generation`.
     4. Outside `with witness.capture():`, execute `entry.graph.replay()` for the current live step and credit `witness.counts[k] += 1` for each `k in entry.captured_keys`.
4. **Zero-recompile restore on `uninstall`**: When the transaction is undone via `Journal.undo_to`, `jit.apply`'s `restore()` closure immediately puts the base `CompiledKernel` (`cubin`) back into `JITFunction.device_caches` and restores `Autotuner.cache`. Only the 1–2 buckets that were actually recaptured during the transaction need to be reverted (either by swapping back saved base `CUDAGraphEntry` handles if retained in a staging slot, or by marking those 1–2 entries stale so the next step recaptures them in milliseconds against the already-restored base `cubin` with zero Triton compilation).

### Cost comparison across pruning levels (estimates, not measured)

| Strategy | Subgraphs recaptured | Buckets recaptured per dirty subgraph | Triton compiles triggered | Typical recapture overhead |
|---|---:|---:|---:|---:|
| **Worker restart** (no in-place rebuild) | All $P$ | All $B$ (plus Python imports, weight load, KV init, Dynamo/Inductor compile) | Cold or disk-cache reload of all kernels | `13–60+ s` |
| **Level 0: Full-sweep recapture** after `jit.apply` | All $P$ | All $B$ ($30\text{--}64$ buckets) | `1` per distinct `constexpr` specialization of edited kernel | `5–20 s` |
| **Level 1: Eager / split-op boundary** (sampler, scheduler, piecewise split op) | `0` | `0` | `1` on next live step (`~30 ms` cold, `~1 ms` disk cache) | **`0 ms`** |
| **Level 2: Piecewise subgraph isolation** ($k$ dirty subgraphs out of $P$) | $k \le P$ | All $B$ | `1` per specialization | $\frac{k}{P} \times (5\text{--}20\text{ s})$ |
| **Level 3: Piecewise + lazy bucket capture** (recapture only buckets hit by live workload) | $k \le P$ | $1\text{--}2$ active buckets (out of $B$) | `1` on first bucket; `0` on buckets sharing specialization | **`50–400 ms`** |

---

## 6. Whole-sweep all-shapes update hierarchy (Tier A implemented in vLLM; Tier B and lazy Tier C target design)

**Status: Tier A implemented in the vLLM lab; Tier B and lazy Tier C remain target design.** Tier A (`cuGraphExecKernelNodeSetParams` with transparent `keep_graph=True` + auto-`instantiate()`) is implemented in the vLLM lab (`repos/vllm/livelab/adapter/pointer_swap.py`) with automatic fail-closed fallback to Tier-C eager recapture (`repos/vllm/livelab/adapter/recapture.py`) and generation-matched replay witness crediting (`repos/vllm/livelab/adapter/graph_guard.py`). Tier B (`cuGraphExecUpdate`) and lazy Tier C remain target design and are not implemented in any lab. Measured B200 timings for Tier A and Tier-C eager fallback are reported below; Tier-B and lazy Tier-C numbers remain estimates from the driver API cost model.

When a serving benchmark or production rollout requires **all pre-captured batch-shape buckets** (e.g., 50–83 buckets) to be warm before traffic resumes so no live step pays lazy capture latency, updating all shape graphs follows a three-tier hardware/driver hierarchy ordered from cheapest (millisecond CPU pointer patching) to most general (stream recapture):

```
Need all B captured shape buckets updated after an edit
  |
  +-- [Tier A: Surgical In-Place Node Patching — cuGraphExecKernelNodeSetParams (implemented in vLLM)]
  |     Preconditions hold?
  |       * target is a @triton.jit kernel (1–5 distinct constexpr specializations across all buckets)
  |       * keep_graph=True + auto-instantiate() retained template CUgraph and CUgraphExec handles
  |       * kernel parameter ABI (signature, constants, attrs, scratch, shared-mem tier, blockDim)
  |         and graph node topology are unchanged (checked fail-closed by check_abi_compatible)
  |       +-- YES: compile the matched specializations once; patch matching CU_GRAPH_NODE_TYPE_KERNEL
  |       |        nodes across all CUgraphExec + template CUgraph handles (cuGraphExecKernelNodeSetParams
  |       |        then cuGraphKernelNodeSetParams). Zero dummy forwards!
  |       +-- NO (non-Triton edit, topology, buffer allocation, or ABI changed -> fail-closed fallback)
  |             |
  +-- [Tier B: Whole-Graph Topological Diff Update — cuGraphExecUpdate (target design)]
  |     Re-run stream capture into a temporary CUgraph and call cuGraphExecUpdate(hGraphExec, hGraph):
  |       +-- Topology matches (CU_GRAPH_EXEC_UPDATE_SUCCESS):
  |       |     driver updates changed kernel nodes in place without full cuGraphInstantiate
  |       +-- CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED:
  |             |
  +-- [Tier C: Recapture — eager R1 implemented in vLLM; piecewise/lazy target design]
        Destroy stale CUgraphExec, re-instantiate via cuGraphInstantiate in shared graph_pool
        (vLLM recapture.py recaptures chosen modes/sizes eagerly; per-subgraph / lazy on-demand
        recapture remains target design)
```

### 1. Surgical in-place kernel-node patching (`cuGraphExecKernelNodeSetParams` with `keep_graph=True`)

Across dozens of padded batch-size buckets (for example, 83 FULL buckets), a `@triton.jit` kernel almost never compiles a separate binary per bucket. Runtime dimensions such as `num_tokens` or `batch_size` are passed as scalar `int32`/`int64` values in `kernelParams`, so all shape buckets typically share only **1 to 5 distinct `tl.constexpr` specializations** (for example, `BLOCK_M=16` for small batches and `BLOCK_M=64` for large batches).

When a body-only edit to a `@triton.jit` kernel preserves its parameter signature and launch topology, running dummy forward passes through Python across every bucket just to re-record the new `CUfunction` pointer into each `CUgraphExec` instance is redundant work. In the vLLM lab (`repos/vllm/livelab/adapter/pointer_swap.py`), `try_pointer_swap` patches the `CUfunction` handle directly inside each already-instantiated `CUgraphExec`:

1. **Retain inspectable graph nodes at capture (`enable_keep_graph`)**:
   - By default, `torch.cuda.CUDAGraph.capture_end()` calls `cudaGraphInstantiate` and immediately destroys the template `cudaGraph_t` (`CUgraph`), leaving only the opaque `cudaGraphExec_t` (`CUgraphExec`), which cannot be enumerated via `cuGraphGetNodes`.
   - `pointer_swap.enable_keep_graph()` idempotently defaults `torch.cuda.CUDAGraph(keep_graph=True)` and automatically calls `self.instantiate()` on `capture_end()` so both `raw_cuda_graph()` (`CUgraph`) and `raw_cuda_graph_exec()` (`CUgraphExec`) are available without changing upstream vLLM capture call sites.
2. **Snapshot and compile only the matched specializations (`snapshot_triton_commit`, `recompile_specialization`)**:
   - Before `jit.apply` clears `device_caches`, `snapshot_triton_commit` records every pre-edit loaded `CompiledKernel` (`CUfunction` handle, `ASTSource`, metadata, and witness keys) for eligible Triton-only edits.
   - `_scan_matching_nodes` enumerates nodes via `cuGraphGetNodes` + `cuGraphNodeGetType` + `cuGraphKernelNodeGetParams` (failing closed on child-graph nodes, unknown node types, unmatched affected kernel names via `cuFuncGetName`, `params.extra != 0`, or non-1D block dimensions).
   - `recompile_specialization` compiles only the unique matched specializations directly from their recorded `ASTSource` and options (1–5 compiles across all 83 FULL graphs), and `check_abi_compatible` verifies signature, constants, attrs, global/profile scratch, shared-memory carveout tier, block dimensions, `num_ctas == 1`, cooperative/PDL/cluster/tensordesc/tmem/maxnreg invariants before touching any graph.
3. **Patch all `CUgraphExec` and template `CUgraph` nodes in place on CPU (`~8–12 ms` driver patch for 2,324–2,988 nodes)**:
   - After `engine.sync()`, `_apply_node_patches` updates `params.func`, `params.blockDimX`, and `params.sharedMemBytes` on each matched node and calls `cuGraphExecKernelNodeSetParams(exec_handle, node, params)` followed by `cuGraphKernelNodeSetParams(node, params)` (so template reads and later swaps/undos remain consistent), rolling back all already-patched nodes if any driver call fails.
   - `pin_graph_kernels` pins both old and new `CompiledKernel` objects against garbage collection while their `CUfunction` handles remain live in `CUgraphExec`.
   - Under `with witness.capture():`, the patched nodes credit `witness.capture_counts`, `GraphReplayGuard.stamp` stamps the updated entries at the new epoch, and `GraphReplayGuard.register_replay_witness` registers the witness keys so `check_before_replay` credits `witness.counts` on generation-matched live replay.
   - **Undo (`~10.5–14.2 ms` measured on B200)**: `Journal.record_call` runs `_rollback_patch_state`, which syncs, restores `(old_func_int, old_block, old_grid, old_shared_mem)` via `cuGraphExecKernelNodeSetParams` and `cuGraphKernelNodeSetParams`, restores `device_caches`, clears replay witnesses, and restores `GraphReplayGuard` epochs without recompiling or recapturing.

#### Exact preconditions and mandatory fallback triggers for `cuGraphExecKernelNodeSetParams`

| Check | Precondition for surgical node patching (implemented in `pointer_swap.py`; fail-closed) | When violated $\to$ automatic fallback to Tier C (`recapture.py`) |
|---|---|---|
| **Triton-only edit scope** | Staged commit edits only `@triton.jit` functions (no repair cells, no structural Python add/remove/import edits, no mixed plain-Python swaps). | Non-Triton or mixed edit (for example `fallback_reason=not_triton_jit_edit`) $\to$ fallback to Tier C. |
| **Graph node handle availability** | `enable_keep_graph()` retained both `raw_cuda_graph()` (`CUgraph`) and `raw_cuda_graph_exec()` (`CUgraphExec`) at capture time. | `keep_graph_unavailable` or `no_captured_graphs` $\to$ fallback to Tier C. |
| **Node type & matching safety** | Every graph node is a `CU_GRAPH_NODE_TYPE_KERNEL` or allowlisted non-kernel node (`MEMCPY`, `MEMSET`, `EMPTY`, `EVENT_RECORD`, `WAIT_EVENT`, `MEM_ALLOC`, `MEM_FREE`, `HOST`, `BATCH_MEM_OP`, `EXT_SEMAS_*`), no child graphs, no `params.extra`, 1D `blockDim`, and no unmatched kernel sharing an affected name. | Child graph node, unsupported node type, `unmatched_affected_node`, or `no_matching_kernel_nodes` $\to$ fallback to Tier C. |
| **Parameter ABI (`kernelParams` layout)** | `check_abi_compatible` confirms identical `signature`, `constants`, `attrs`, `global_scratch_size` / alignment, `profile_scratch_size == 0`, `num_ctas == 1`, shared-memory tier, `blockDimX <= n_max_threads`, and launch flags (`launch_cooperative_grid`, `launch_pdl`, `tensordesc_meta`, `tmem_size`, `instrumentation_mode`, `cluster_dims`, `maxnreg`). | Any `abi_mismatch` or `compile_failed` $\to$ restores `device_caches` and falls back to Tier C. |
| **Grid / topology / buffer invariance** | The edit replaces 1 kernel body with 1 kernel body using the same pooled tensor addresses and host-computed `gridDim` already bound in `CUDA_KERNEL_NODE_PARAMS`. | Host wrapper changes grid math, splits a kernel into 2 launches, or allocates new intermediate tensors $\to$ Tier B (target design) or Tier C (`recapture.py`). |

#### Measured B200 Tier-A vs. Tier-C timings

Measured on `1× NVIDIA B200`, vLLM `0.33.0`, `TRITON_ATTN`, `FULL_AND_PIECEWISE` (83 FULL + 83 PIECEWISE buckets), our 2026-11-24 live verification run:

| Model (graph footprint) | Path & edited target | Specs / captures | `compile_ms` | `patch_ms` | Total update | `undo_ms` | Perturbed max abs logprobs diff | Post-undo diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen3-0.6B` (28 layers, 83 FULL graphs / 2,324 kernel nodes) | **Tier A**: `kernel_unified_attention` | 5 specs | `10.3 ms` (warm) | `9.28 ms` | `0.113 s` | `10.82 ms` | `6.377676` | `0.0` (16/16 tokens exact) |
| `Qwen3-0.6B` (28 layers, 83 FULL graphs / 2,324 kernel nodes) | **Tier A**: `reshape_and_cache_kernel_flash` | 1 spec | `2.31 ms` (warm) | `8.31 ms` | `0.102 s` | `10.52 ms` | `3.469832` | `0.0` |
| `Qwen3-0.6B` (28 layers, 83 FULL + 83 PIECEWISE) | **Tier C (R1 fallback)**: plain Python `unified_attention` (`fallback_reason=not_triton_jit_edit`) | 166 captures | — | — | `5.550 s` | `415.2 ms` | `6.377676` (exact match with Tier A) | `0.0` |
| `Qwen3-8B` (36 layers, 83 FULL graphs / 2,988 kernel nodes) | **Tier A**: `kernel_unified_attention` | 5 specs | `1,635.5 ms` (cold) | `11.93 ms` | `1.769 s` (cold) | `14.20 ms` | `6.669460` | `0.0` |
| `Qwen3-8B` (36 layers, 83 FULL graphs / 2,988 kernel nodes) | **Tier A**: `reshape_and_cache_kernel_flash` | 1 spec | `2.60 ms` (warm) | `10.70 ms` | `0.135 s` | `13.28 ms` | `6.934308` | `0.0` |
| `Qwen3-8B` (36 layers, 83 FULL + 83 PIECEWISE) | **Tier C (R1 fallback)**: plain Python `unified_attention` (`fallback_reason=not_triton_jit_edit`) | 166 captures | — | — | `8.041 s` | `473.1 ms` | `6.669460` (exact match with Tier A) | `0.0` |

### 2. Whole-graph topological diff update (`cuGraphExecUpdate` / `cudaGraphExecUpdate`; target design)

When an edit modifies Python launch glue (such as changing how a Python wrapper computes grid dimensions or scalar launch arguments across shape buckets) or swaps multiple kernels without changing the sequence of graph nodes, surgical single-node patching cannot infer the new host-computed `kernelParams` without running the Python wrapper, yet full `cuGraphInstantiate` across all buckets is still avoidable:

1. **Capture a new template `CUgraph` on the capture stream**: Run the Python/Inductor wrapper for bucket $b$ under stream capture (`with witness.capture():`) to produce a new `CUgraph` (`hNewGraph`) reusing `shared_graph_pool`.
2. **Apply topological diff in place (`cuGraphExecUpdate`)**: Call `cuGraphExecUpdate(entry.hGraphExec, hNewGraph, &resultInfo)` (runtime API: `cudaGraphExecUpdate`).
   - The CUDA driver walks `entry.hGraphExec` and `hNewGraph` in lockstep, comparing node types and dependency edges.
   - If the topology matches (`resultInfo.result == CU_GRAPH_EXEC_UPDATE_SUCCESS`), the driver patches the modified `CU_GRAPH_NODE_TYPE_KERNEL` parameters and `CUfunction` pointers inside `entry.hGraphExec` in place, skipping full command-buffer allocation and hardware validation.
3. **Automatic fallback on `CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED`**: If the edit added or removed a kernel launch, changed node types, or exceeded driver update constraints (`resultInfo.result != CU_GRAPH_EXEC_UPDATE_SUCCESS`), the runner destroys the old `CUgraphExec` and calls `cuGraphInstantiate(&entry.hGraphExec, hNewGraph, 0)` on the same shared `graph_pool`.

### 3. Piecewise subgraph + active-bucket / lazy on-demand recapture (general fallback; eager R1 implemented, lazy target design)

When an edit changes graph topology (splitting 1 kernel into 2, adding an intermediate tensor allocation, or editing traced Python control flow) or fails any Tier-A precondition, the vLLM lab currently falls back automatically to eager Tier-C recapture (`repos/vllm/livelab/adapter/recapture.py`). When all buckets do not need to be materialized up front before the first verification step, the target design extends this fallback with the three-level pruning strategy in Section 5:

- Restrict invalidation to the $k \le P$ dirty piecewise subgraphs $G_i \in \mathcal{D}$;
- Eagerly recapture only the active workload bucket $b_{\text{active}}$ via `cuGraphInstantiate` inside `with witness.capture():`;
- Leave the remaining $B - 1$ shape buckets marked stale (`entry.txn_generation < current_txn_generation`) so they recapture lazily on first hit.

### Summary of the whole-sweep update hierarchy

| Tier | Status | Mechanism | When it applies | Forward passes run | Driver graph ops | Latency (measured where implemented; estimates for target design) |
|---|---|---|---|---:|---|---:|
| **Tier A** | **Implemented** (vLLM: `pointer_swap.py`) | **Surgical node patch** (`cuGraphExecKernelNodeSetParams` + `cuGraphKernelNodeSetParams`, transparent `keep_graph=True` + auto-`instantiate()`) | `@triton.jit` body edit; unchanged `kernelParams` ABI and node topology; fails closed to Tier C on any mismatch | **`0`** (compiles 1–5 matched specializations directly) | $N_{\text{nodes}} \times$ (`cuGraphExecKernelNodeSetParams` + `cuGraphKernelNodeSetParams`) | **Measured (B200, 83 FULL graphs / 2,324–2,988 nodes):** **`0.102–0.135 s`** warm / **`1.769 s`** cold (`8.31–11.93 ms` patch, `10.52–14.20 ms` undo) |
| **Tier B** | Target design | **Whole-graph diff update** (`cuGraphExecUpdate`) | Python wrapper / grid math / multi-node parameter change with unchanged node topology | $B$ stream captures (no warm-up needed once `cubin` is cached) | $B \times \text{cuGraphExecUpdate}$ (no `cuGraphInstantiate`) | Estimate (50 buckets): **`~0.3–1.5 s`** |
| **Tier C (lazy)** | Target design | **Piecewise + active-bucket / lazy recapture** (`cuGraphInstantiate`) | Arbitrary topology change, new intermediate buffers, or `keep_graph=False` when only active buckets run now | $1\text{--}2$ active buckets now; rest on demand | $1\text{--}2 \times \text{cuGraphInstantiate}$ now | Estimate: **`50–400 ms`** to first step |
| **Tier C (eager)** | **Implemented** (vLLM: `recapture.py`, auto fallback from Tier A) | **Full-sweep / selected-descriptor recapture** (`cuGraphInstantiate` across chosen modes/sizes) | Non-Triton edits, topology/ABI changes, or any Tier-A fallback when all buckets must be pre-instantiated | $B \times$ (warm-up + stream capture) | $B \times P \times \text{cuGraphInstantiate}$ | **Measured (B200, 166 captures):** **`5.550–8.041 s`** (`415.2–473.1 ms` undo); `14.4 s` all graphs on Qwen3-8B TP1 (`repos/vllm/docs/architecture.md`) |

### Boundary between the shared kernel and engine adapters

Per `AGENTS.md`, the shared transaction kernel (`core/livelab/kernel/`) stays pure Python (standard library only, importing no engine, `torch`, or `triton`), while each engine's live lab (`repos/<engine>/livelab/`) supplies the engine-specific graph hooks:

| Responsibility | Owner | Mechanism |
|---|---|---|
| Bytecode swap (`fn.__code__`), closure pairing, and exact code undo | `core/livelab/kernel/swap.py`, `journal.py` | `stage`, `validate`, `apply`, `Journal.set_code`, `Journal.undo_to` |
| In-place `@triton.jit` rebuild, cross-module transitive caller reset, `seeded_lines`, and zero-recompile undo | `core/livelab/kernel/jit.py` | `text_of`, `plan`, `apply`, `arm_added`, `seeded_lines` (duck-typed via `jit.NEED`) |
| Two-ledger witness (`counts` vs. `capture_counts`), Dynamo constant marking, and `pre_run_hooks` arming | `core/livelab/kernel/witness.py` | `Witness(capturing=...)`, `with witness.capture():`, `witness.arm`, `witness.arm_launch`, `witness.check`, `trace_safe` |
| Finding catalog (`jit_rebuild`, `jit_target`, `captured_target`, `captured_partly`, `zero_witness`, `running_frame`) | `core/livelab/kernel/findings.py` | `CATALOG`, `finding`, `decide` |
| Static graph-boundary detection (`captured_target` / `captured_partly`) and stream-capture probe | `repos/<engine>/livelab/adapter/` (implemented for vLLM: `graphs.py`, `side.py`) | Engine `LocalParticipant(checks=[...], capturing=...)` hooks |
| Tier-A surgical node patching (`cuGraphExecKernelNodeSetParams`), `keep_graph=True` + auto-`instantiate()`, `CUmodule` pinning, O(1) epoch staleness guard, and generation-matched replay witness crediting | `repos/vllm/livelab/adapter/pointer_swap.py`, `graph_guard.py` | `enable_keep_graph`, `snapshot_triton_commit`, `recompile_specialization`, `check_abi_compatible`, `try_pointer_swap` (fail-closed fallback to Tier C), `pin_graph_kernels`, `GraphReplayGuard.stamp` / `register_replay_witness` / `check_before_replay` |
| Tier-C recapture per cudagraph mode and batch descriptor in the step gap, with journaled undo (also automatic fallback from Tier A) | `repos/vllm/livelab/adapter/recapture.py` | Replaces the wrappers' entry dicts and dispatcher keys, captures under `with witness.capture():`, restores the base dicts on undo |
| Target design, not implemented: Tier-B topological diff update (`cuGraphExecUpdate`), Tier-C replay witness crediting, per-subgraph dependency tracking (Level 2), and on-dispatch lazy bucket recapture (Level 3) | `repos/<engine>/livelab/` (planned owner) | Would extend `GraphReplayGuard` crediting to Tier-C captures, apply `cuGraphExecUpdate` when topology matches, and lazily recapture dirty subgraphs/buckets on first dispatch |

---

## 7. Cross-engine warm-cache restart tax breakdown (`#106`)

Even when OS page caches and on-disk compiler caches (Triton, TorchInductor, FlashInfer, CUDA PTX JIT) are already warm, starting a fresh engine process to test an edit still costs **`29–228 s` on standard workloads** (and up to **`1,012 s`** when upstream autotune caches miss). Comparing the measured phase timings across `repos/vllm/kb/costs.md`, `repos/megatron/kb/costs.md`, and `repos/fastvideo/kb/costs.md` shows that every engine pays the same three per-process tax layers, and that each layer maps to a specific LiveLab mechanism:

### Cross-engine warm-cache fresh-process restart breakdown

| Tax layer | What happens in a fresh process (with warm OS page & disk JIT caches) | vLLM (`repos/vllm/kb/costs.md`) | Megatron-LM (`repos/megatron/kb/costs.md`) | FastVideo (`repos/fastvideo/kb/costs.md`) | How LiveLab eliminates or disaggregates the tax |
|---|---|---|---|---|---|
| **1. Python import storm** (`12.7–18.1 s` local / warm; `20.5–85 s` over shared NFS) | Importing `8,000–8,600+` modules (`torch`, `triton`, `transformers`, `flashinfer`, engine packages), thousands of `stat()` calls, dynamic `.so` symbol relocations, and eager import-time registry decorators (`Library.define`, `CustomOp.register`). Spawned workers pay this a second time unless forked. | **`12.7 s`** (`8,631` modules in zygote; `4.7–4.9 s` just to import `vllm_args` parser; `+11–13 s` if workers fall back to `spawn`). | **`11.4–12.4 s`** on RAM disk (`1.0–1.7 s` `torch` + `10.0–10.4 s` `megatron.training`, `8,300` modules); **`33.3–54.5 s`** on shared NFS (`85 s` at 2 nodes). | **`18.1 s`** in driver with path-stable warm caches (`17.4–38.4 s` range) + **`17–20 s`** re-import in spawned worker (`357+` top-level modules). | **Live swap / in-place rebuild (`0 s`):** process stays alive.<br>**Zygote fork + module refresh (`0.1–2.8 s`):** pre-imports heavy stack once with `forksafe.py` and `gc.freeze()`, then forks workers (`VLLM_WORKER_MULTIPROC_METHOD=fork`) and drops only edited modules + their importers (`zygote_refresh.py`). |
| **2. Weight deserialization, sharding & placement** (`6.8–124.9 s` from disk/page cache vs. `0.2–2.2 s` via IPC holder) | Reading safetensors/checkpoints from storage or OS page cache, CPU staging, tensor/sequence-parallel slicing, FP8/NVFP4 quantization or packing, and copying into a fresh CUDA context. | **Qwen3-8B TP1:** **`6.82 s`** warm disk load $\to$ **`2.24 s`** via `--weights` (`ipc_cache` zero-copy CUDA IPC, `weights_probe` `0.041 s`).<br>**Scheduler benchmark (1× GB200):** keeping weights cuts restart from `157 s` to `100 s` (`-57 s`).<br>**DeepSeek-V4.1 TP4:** **`124.9 s`** warm (`178.6 s` cold). | **Random GPU init (14B / 32B):** `0.3–0.4 s` model + optimizer build.<br>**Live lab state restore (`S`):** `96.3–106.8 GiB/rank` in pinned host RAM restored in **`0.25–0.79 s`** instead of checkpoint reload. | **MiniMax H3 (NVFP4):** **`30.1–35.3 s`** warm `from_config` (`195.6 s` cold): VAE `5.0–5.2 s`, DiT `2.5–3.0 s`, text encoder `2.3–2.7 s`.<br>**With IPC holder:** **`0.72–1.06 s`** total (`1,838` DiT tensors / `36.9 GiB` mapped in `0.12–0.48 s`, text encoder `0.19 s`, VAE `0.05–0.09 s`). | **Live swap / in-place rebuild (`0 s`):** weights stay resident in the worker's GPU tensors (`rebuild.py` swaps `__class__` and merges `meta`-device `__init__` diffs around live parameters).<br>**IPC weight holder (`0.2–2.2 s`):** `weights.py` (`vllm preload` + `ipc_cache`) or FastVideo `holder=True` maps pre-loaded GPU pages into the new worker via POSIX FD CUDA IPC. |
| **3. Warm JIT, profiling, communicator & graph capture taxes** (`1.6–15.6 s` normal warm; up to `627 s` on upstream cache bugs) | Per-process CUDA context & NCCL init, peak-activation profiling (`determine_available_memory`), Dynamo bytecode re-tracing (`0.7–1.6 s`), Inductor `torch_key()` hashing ~2,200 torch files (`0.8–2.6 s`), per-process `cuModuleLoad` across cached cubins, CUDA graph capture (`5–40 s`), and upstream autotune cache misses. | **Qwen3-8B TP1 eager:** `determine_available_memory` profile run **`1.63–2.10 s`** $\to$ **`0.005–0.006 s`** with `--kvplan` (`kvplan.py`).<br>**Scheduler benchmark:** autotune `71 s`, CUDA graphs `42 s`, remaining process init `34.4 s`.<br>**DeepSeek-V4.1 TP4:** profile + graph capture **`79.0 s`** + **`40 s`**, JIT warmup **`52.5 s`**, FlashInfer autotune **`627.2 s`** (retuned despite saved configs). | **14B / 32B warm start:** `warmup_training_kernels` **`2.2–3.2 s`** (`0.6–0.8 s` when Dynamo/Inductor preloaded in zygote); **first train step excess `+5.4–8.7 s`** over steady step (`4.4–6.9 s`): NCCL communicators `1.7 s`, cuDNN fused-attn plans `1.96 s`, allocator `cudaMalloc` growth `1.72 s`, un-warmed Dynamo frames `1.6 s`. | **First video excess `+7.8 s`** (`13.95 s` vs. `6.1 s` warm video): Inductor `torch_key()` `2.55 s` (eliminated by `preset_torch_key`), Dynamo bytecode tracing `0.79–1.2 s`, Triton autotune `1.94–2.64 s` (eliminated by `TRITON_CACHE_AUTOTUNING=1`), cached cubin loads `0.39–1.26 s`, first-call CUDA init `1.5 s`.<br>**Moved cache root bug:** FlashInfer `build.ninja` embeds resolved absolute paths, triggering a **`153–159 s`** rebuild unless staged at a fixed path (`jitcache.py`). | **KV plan holder (`kvplan.py`, `0.005 s`):** skips `determine_available_memory` profile run when config + free memory match.<br>**Path-stable JIT cache (`jitcache.py` / `CACHE_LAYOUT`):** fixes cache roots, presets `torch_key`, and enables Triton autotune caching.<br>**In-process live swap (`0.05–0.06 s`), `@triton.jit` rebuild (`jit.py`), vLLM Tier-A surgical node patch (`pointer_swap.py`, `0.102–0.135 s` warm / `1.769 s` cold on B200), in-place `rebuild.py` (`0.19–1.27 s`), and vLLM Tier-C graph recapture (`5.550–8.041 s` full B200 fallback; `3.0–5.6 s` per region, `repos/vllm/docs/architecture.md`):** keeps the live CUDA context, NCCL communicators, cuDNN plans, allocator pools, compiled FX frames, and untouched `CUDAGraph` buckets warm in memory. |
| **End-to-end warm restart vs. LiveLab iteration** | Sum of all phases from command invocation to `/health`, first train step, or first generated video. | **Fresh warm start:** **`51.85 s`** (Qwen3-8B eager), **`228 s`** (scheduler bench), **`1,012 s`** (DeepSeek-V4.1 TP4).<br>**Zygote + `--weights` + `--kvplan`:** **`7.37 s`** to `/health` (`7.0×` faster).<br>**In-process live swap:** **`0.022 s`** install (`2.4 s` small-batch check, **`95×`** faster). | **Fresh warm start:** **`58.0 s`** (14B NFS) / **`29.3 s`** (14B RAM disk); **`110.1 s`** (32B NFS) / **`37.6 s`** (32B RAM disk).<br>**Zygote restart:** **`15.0–16.5 s`** (14B) / **`22.8–25.1 s`** (32B).<br>**In-process live swap (`install` + restore `S`):** **`0.05 s`** install + **`0.25 s`** state restore (`0.45 s` 3-step tiny check vs. `57.5 s` restart, **`127×`** faster). | **Fresh warm start:** **`66.4–72 s`** to first video (`507 s` cold; `223.4 s` if cache path moved).<br>**Zygote + IPC weights holder:** **`4.78–5.62 s`** respawn (`~15 s` to first video).<br>**In-place `rebuild`:** **`0.19–1.27 s`** (`6.4–7.5 s` quiet to first video).<br>**In-process live swap:** **`0.06 s`** install (`6.7 s` edit-to-first-video, **`10×`** faster than warm restart, **`75×`** faster than cold). | **Four-tier escalation ladder:**<br>1. Bytecode swap + `@triton.jit` in-place rebuild (`10–80 ms`), plus vLLM Tier-A surgical graph node patching (`pointer_swap.py`: `0.102–0.135 s` warm / `1.769 s` cold on B200, `8.31–11.93 ms` patch, `10.52–14.20 ms` undo)<br>2. Tier-C CUDA graph recapture in the vLLM lab (`5.550–8.041 s` full fallback on B200; `3.0–5.6 s` per region in `repos/vllm/docs/architecture.md`; Tier B `cuGraphExecUpdate` and lazy Tier C remain target design) or in-place class/module `rebuild` (`0.2–1.3 s`)<br>3. Zygote fork + `--refresh` + IPC weights + cached KV plan (`4.8–16 s`)<br>4. Full process restart with path-stable caches (reserved for startup-only topology/flag changes and final confirmation). |



