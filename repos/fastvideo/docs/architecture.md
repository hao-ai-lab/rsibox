# FastVideo architecture: process layout, restart boundaries, costs

The map a reviewer needs for a FastVideo live-lab PR, for the MiniMax H3 workload
(`../kb/workload.md`): what runs where, which box an edit invalidates, and what each restart
action costs. `file:line` is at FastVideo `376e445f` (`../kb/priors.md`) unless marked. Numbers
come from `../kb/costs.md`, which names their sources.

## 1. Process layout

| | 1 GPU | 2 or 4 GPUs |
|---|---|---|
| Executor | `UniprocExecutor` (`executor.py:63-66`; also chosen when backend is `mp` and `num_gpus == 1`). The live lab forces `MultiprocExecutor` here too (`../livelab/README.md`) | `MultiprocExecutor` (`executor.py:67-69`) |
| Worker | In the caller's process (`uniproc_executor.py:63-69`) | One spawned process per rank, named `FVWorkerProc-<rank>` (`multiproc_executor.py:570-600`; spawn is forced, `utils.py:1273-1288`) |
| Request loop | Caller thread calls `execute_forward` | Parent sends `execute_forward` over a pipe; each worker runs `worker_busy_loop` (`multiproc_executor.py:797-830`) |
| Run a function in the worker | `collective_rpc(callable)` (`uniproc_executor.py:230-242`) runs it on the caller thread | `collective_rpc` sends `{"method", "args", "kwargs"}` to each pipe and waits on every `recv` (`multiproc_executor.py:346-374`) |

`collective_rpc` details:

- A callable `method` is called as `method(worker_wrapper, *args)` (`utils.py:816-835`). A
  `bytes` value is `cloudpickle.loads(...)` first, so send a function defined in a script as
  cloudpickle bytes; a plain pickled function from `__main__` will not resolve in a spawned
  worker.
- `timeout` is ignored by both executors (`uniproc_executor.py:240`; the multiprocess `recv`
  is unbounded, `../kb/lessons.md`). A hung patched call blocks the whole RPC; the live lab
  calls ranks with its own deadline.
- Workers run the busy loop on the main thread: a call cannot interleave with a running
  request. Swap between requests.
- On the encoder-split layout (`h3_encoder_split`) ranks load different module sets
  (`gpu_worker.py:146-155`). Not used in the 1/2/4-GPU workload here.

Where the work is:

- **Denoise loop:** `MiniMaxH3DenoisingStage.forward`
  (`pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py:193`). The loop is
  `for index, (video_timestep, audio_timestep) in enumerate(...)` at `:298`: build VSA metadata
  (`:310`), call `self.transformer(...)` (`:334`), `scheduler.step` for video and audio
  (`:349-361`). 8 iterations for this workload.
- **Step gap:** the loop top checks `self.interrupt` at `:300`, before the next metadata
  build. That boundary, between `batch.timestep = ...` (`:362`) and the next iteration, is
  the only place inside one request where Python state is quiescent. It is not a hook: the
  live lab installs between requests, and an install there would need a new pause point.
- **Stage order:** input preparation, conditioning, (release text encoder, load denoise
  modules), latent preparation, denoising, video decode, audio decode
  (`minimax_h3_pipeline.py:639-656`).
- **Sequence-parallel all-to-all in attention:** `DistributedAttention.forward`
  (`attention/layer.py:108-190`): all-to-all on QKV at `:148`, `attn_impl.forward` at `:173`,
  all-to-all on the output at `:188`. The VSA variant repeats the pattern at `:244`, `:270`.
  The all-to-all uses NCCL by default and the fused NVLink kernel when
  `FASTVIDEO_ULYSSES_A2A=auto` (`envs.py:296-301`). Any edit to attention code between
  these calls, to the all-to-all, or to head/sequence padding is RANKS.
- **Single rank:** with `sp_size == 1` the same code runs, and the all-to-all returns the input unchanged (`base_device_communicator.py:142,187`);
  a swap there is OP (not run).

## 2. Startup phases

Order on a cold start. "Warm" means the cost can stay resident in a running lab.

| # | Phase | What happens | Source | Keep warm? | Measured (1 GPU, `../kb/costs.md`) |
|---|---|---|---|---|---|
| 1 | Imports | `import fastvideo`, torch, Triton, flashinfer (NVFP4 imports flashinfer lazily at first use) | `layers/quantization/nvfp4_config.py:43-56` | Yes: one process | 93.5 s cold, 18-22 s warm page cache |
| 2 | Executor and worker | 1 GPU: worker built in the caller process. N GPUs: one spawned process per rank | `worker/executor.py:63-71`, `worker/uniproc_executor.py:49-74`, `worker/multiproc_executor.py:570-600` | Yes | inside the model load |
| 3 | Distributed init | NCCL group, sequence-parallel group (`sp_size`) | `worker/gpu_worker.py:140-161` | Yes | inside the model load |
| 4 | Text encoder load | Qwen3-VL NVFP4 loaded first; DiT and VAEs deferred when sequential or lazy load is on | `pipelines/basic/minimax_h3/minimax_h3_pipeline.py:422-506` | Yes, but see note 1 | 91.1 s cold, 4.3 s warm |
| 5 | DiT load | Weights, then NVFP4 weight conversion (`_maybe_quantize_model`) | `models/loader/fsdp_load.py:543`, `fastvideo_args.py:499` | Yes, but see note 1 | 76.3 s cold, 2.5 s warm |
| 6 | VAE and audio VAE load | H3 VAE (optionally NVFP4 decoder), audio VAE | `pipelines/basic/minimax_h3/minimax_h3_pipeline.py:507-523` | Yes | 21.9 + 3.6 s cold, 5.0 + 0.5 s warm |
| 7 | Optional compile setup | Per-block compile wrappers attached; compile itself runs at the first call | `pipelines/composed_pipeline_base.py:271-294`, `models/loader/fsdp_load.py:572-600` | Wrappers: yes. Compiled code: no, per edit | off in the workload |
| 8 | Triton JIT and autotune | First call of each `@triton.jit` kernel per shape and constexpr set | section 3 | JIT cache on disk; autotune in process | 23 kernel loads 0.39-1.26 s; VSA autotune 1.94-2.64 s per process |
| 9 | First request | Lazy compile, JIT, flashinfer kernel setup, allocator warm-up | `../kb/costs.md` | Yes after the first request | 217.9 s cold JIT, 13.9-15.1 s warm JIT; warm video 6.1 s |

Notes:

1. **H3 can release and reload modules between requests.** With `lazy_module_load` or
   `h3_sequential_load`, the text encoder is freed after encode and the DiT and VAEs are loaded
   afterwards (`minimax_h3_pipeline.py:639-656`: encode, release text encoder, load denoise
   modules, run denoise and decode). Both default to on for unified-memory hosts and off
   otherwise (`minimax_h3_pipeline.py:422-448`, `fastvideo_args.py:1141`). On a discrete-GPU
   node, check the effective setting at startup; if on, module identity changes per request
   and a patch applied to a module instance is lost on reload.
2. Phase 8 on a restart: the Triton disk cache survives a restart; the in-process autotune
   table does not unless `TRITON_CACHE_AUTOTUNING=1` (`livelab/services/jitcache.py`). Besides
   VSA's `_attn_fwd_sparse`, `minimax_h3_sparse_int8.py:36` uses `@triton.autotune` on the H3
   path (INT8 sparse attention, not the default BF16 VSA route).

## 3. Restart classes: what an edit invalidates

Groups (OP, GRAPH, STATE, RESOURCE, RANKS, NATIVE) are the function groups defined in
`core/livelab_overview.md`. "Live" means the engine process keeps running (the live swap,
`install`); it does not mean the check is free. Section 5 says which restart action covers
each class and what it costs.

| Edit | Group | Live or restart | Why (source) |
|---|---|---|---|
| Stage Python (`pipelines/basic/minimax_h3/stages/*.py`) | OP | Live | Stage objects are built once and called through `_stage_name_mapping[name](batch, args)` (`minimax_h3_pipeline.py:639-656`). Swap the method on the class or instance. Stages are rebuilt only if `_denoise_stages_ready` is false |
| Denoise loop body (`minimax_h3_denoising.py:298-360`) | OP | Live, takes effect on the next request | The loop runs inside one `forward` call. A swap during a request does not reach the running call |
| DiT block `forward` (`minimax_h3.py:677-749`), attention module `forward` (`:352`) | OP, or GRAPH if compiled | Live when eager (default). If `inference_torch_compile` is on, the bound `forward` is wrapped by `torch.compile` per block (`fsdp_load.py:823`): swapping Python does not change the compiled graph | `fsdp_load.py:572-600` attaches the compile wrapper after load. Recompile the affected blocks |
| Fused Triton helpers (`models/dits/minimax_h3_fusions/*.py`) | OP for the wrapper, GRAPH for the kernel | Live for the wrapper. A kernel body edit re-JITs on its next call | `@triton.jit` at `exact.py:45-127`, `modulation.py:33,72`, `qknorm_rope.py:23`, `swiglu.py:36`. Which helper runs is chosen once per DiT at construction from the env (`minimax_h3.py:794`, `:805`), so enabling a fusion that was off needs the DiT blocks rebuilt (STATE) or a restart |
| Attention backend selection (`attention/selector.py`, `MiniMaxH3Attention.__init__`) | STATE | Restart, or rebuild the attention modules | The backend class is chosen in `MiniMaxH3Attention.__init__` (`minimax_h3.py:252-262`) and held by `distributed_attention.attn_impl`. The resolver is `@cache`d on all its inputs (`selector.py:275`), so it needs no cache clear but existing layers keep their backend. FastVideo's `attention/AGENTS.md` forbids changing the backend env var mid-process (read at an older pin; not checked at `376e445f`) |
| VSA-H3 impl (`attention/backends/video_sparse_attn_h3.py`: `forward` `:783`, `tile` `:682`, metadata builder `:398`) | OP | Live in eager mode | Per-call Python. The tile buffer holder is state owned by the metadata builder (`:350-393`); a swap that changes buffer shape or dtype must drop the holder |
| VSA Triton kernels (`fastvideo_kernel/triton_kernels/*`, `block_sparse_attn.py`) | GRAPH (re-JIT) | Live | Imported by function at call time (`block_sparse_attn.py:165`), so a reload plus re-JIT takes effect. Installed library lives outside the engine tree: patch the installed copy or rebind |
| Native `fastvideo-kernel` CUDA/CuTe (`csrc`, `attn_qat_infer/blackwell/*.cu`, `block_sparse_attn_ptx_blackwell.py`) | NATIVE | Live only as a versioned side library plus rebind; otherwise restart | Built by scikit-build-core/CMake (`fastvideo-kernel/pyproject.toml:30-33`, `build.sh`). A loaded shared library cannot be unloaded. `_C.fastvideo_kernel_ops` is looked up at call time (`block_sparse_attn.py:15-22`) |
| Sequence-parallel all-to-all (`attention/layer.py:148,188,244,270`, `distributed/device_communicators/*`) | RANKS | All ranks or none | Every rank must make the same collective calls in the same order. An edit between two collectives, or a different revision on one rank, hangs or corrupts |
| Loader and quantization (`models/loader/fsdp_load.py`, `layers/quantization/nvfp4_*.py`, `_maybe_quantize_model`) | RESOURCE | Restart, or reload one module | Load-time layout: packed FP4 weights and scales are built once from BF16. `NVFP4QuantizeMethod.apply` (`nvfp4_config.py:635`) is per-call and can be swapped as OP |
| VAE decode code (`models/vaes/minimax_h3_*.py`) | OP, GRAPH if graphed | Live; NVFP4 fused decoder keeps CUDA graphs per shape | `minimax_h3_nvfp4_fused.py:596-663` caches graphs (`GRAPH_CACHE_LIMIT = 2` at `:591`). Drop `blocks._nvfp4_cuda_graphs` after a swap |
| Pipeline config (`configs/pipelines/minimax_h3.py`), `FastVideoArgs` fields | STATE | Restart | Parsed once; `_apply_transformer_quant` and checkpoint arch configs run at init (`minimax_h3_pipeline.py:162-186,301`) |
| Env vars (`FASTVIDEO_*`, `envs.py`) | none | Restart | Fusion and exact-kernel sets are read at DiT construction (`minimax_h3.py:794-805`). Some flags are read per call (for example `FASTVIDEO_H3_VAE_TILE_PARALLEL`, `minimax_h3_video.py:1130`), but a changed env var is not an installable edit |

Reading rule for this table: if the edit changes what a module builds at construction, it is
STATE or RESOURCE and the module must be rebuilt: the in-place rebuild covers STATE when no new
tensors are needed, a RESOURCE edit needs a restart that loads the weights again. If it changes
what a method does per call, it is OP and the live swap covers it. Only the first request after
an install proves it ran; use the lab receipt.

## 4. What is compiled or captured on the H3 path

### torch.compile

| Site | Always on? | Source |
|---|---|---|
| `_gated_residual` (`@torch.compile(dynamic=True, fullgraph=True)`) | Only when the `modulate` fusion is on and not already under compile | `models/dits/minimax_h3.py:102`, called at `:744-745` |
| FP8 and FP4 VSA helpers (7 `@torch.compile` functions) | Only on the opt-in `FASTVIDEO_H3_VSA_FP4` route, which requires an eager DiT | `models/dits/minimax_h3_vsa_fp4.py:210,439-487`, `fastvideo_args.py:451` |
| Per-block fullgraph DiT compile | Opt-in: `inference_torch_compile` or `FASTVIDEO_INFERENCE_TORCH_COMPILE` | `fastvideo_args.py:293,421-427`, `envs.py:396`, `fsdp_load.py:572-600,823` |
| Whole DiT, text encoder, VAE, audio VAE | Opt-in: `enable_torch_compile`, `_text_encoder`, `_vae`, `_audio_vae` | `fastvideo_args.py:264-271`, `composed_pipeline_base.py:271-294` |
| VAE tile helpers with `mode="reduce-overhead"` | Opt-in, only with VAE compile | `models/vaes/minimax_h3_video.py:734-752` |
| Vocab embedding, MXFP8 linear | Not on the NVFP4 H3 path | `layers/vocab_parallel_embedding.py:123`, `layers/mxfp8linear.py:26` |

Regional compile of VSA-H3 only works with 64-token tiles and the `ptx_blackwell` route
(`video_sparse_attn_h3.py:626-680`, `fsdp_load.py:678-700`). The default tile size is 256
(`fastvideo_args.py:460-461`), so enabling regional compile on this workload falls back to eager
with a warning unless `VSA_tile_size=64` is set. `FASTVIDEO_DISABLE_ATTENTION_COMPILE` defaults
to true and keeps attention out of compile graphs (`envs.py:373-376`).

Edits to the DiT forward under compile: the wrapper is attached per block, so recompile only
the changed blocks. The fusions are registered as custom ops with fake implementations so
Dynamo treats them as opaque nodes (`minimax_h3.py:998-1026`).

### Triton

| Group | Sites | Source |
|---|---|---|
| DiT fusions | 14 `@triton.jit` kernels | `minimax_h3_fusions/exact.py:45-127`, `modulation.py:33,72`, `qknorm_rope.py:23`, `swiglu.py:36` |
| Fused add and norm | 1 | `layers/triton_fused_norm.py:50` |
| VSA tile scatter | 1 | `attention/backends/video_sparse_attn_h3_scatter.py:30` |
| Sparse attention (tile 64, 128, 256 fallback) | `fastvideo_kernel/triton_kernels/*` | `block_sparse_attn.py:153-176`, `block_sparse_attn_256.py` |
| INT8 sparse attention | `@triton.autotune` + `@triton.jit` | `attention/backends/minimax_h3_sparse_int8.py:18-39` |
| VAE | NVFP4 fused decoder (16 jit kernels) and INT8 | `models/vaes/minimax_h3_nvfp4_fused.py:42-232`, `minimax_h3_int8_kernels.py:11` |
| NVFP4 dequant | 1 | `layers/quantization/nvfp4_dequant.py:8` |

Which of these run in the default FastH3-QAD-v2 BF16-VSA path depends on env flags and tile
size. A warm first video loads 23 Triton kernels from the disk cache (`../kb/costs.md`); which
ones is not recorded.

### CUDA graphs

- **DiT path: none found at `376e445f`.** Search for `CUDAGraph`, `cuda.graph`, `reduce-overhead`
  under `models/dits/minimax_h3*` and `attention/backends/*h3*` finds only comments. The
  denoise loop calls `torch.compiler.cudagraph_mark_step_begin()` each step
  (`minimax_h3_denoising.py:326`); it matters only if a user sets
  `torch_compile_kwargs={"mode": "reduce-overhead"}`. Inductor with
  `reduce-overhead` is the only way a DiT graph could appear.
- **VAE: yes.** The NVFP4 fused decoder replays small calls as per-shape CUDA graphs, on by
  default (`FASTVIDEO_H3_VAE_NVFP4_CUDA_GRAPH`, `envs.py:729`; `minimax_h3_nvfp4_fused.py:635-663`).
  Only active with the NVFP4 H3 VAE decoder checkpoint.

### Native extensions (built ahead of time)

| Piece | Built by | Used for |
|---|---|---|
| `fastvideo_kernel._C.fastvideo_kernel_ops` | CMake (scikit-build-core), `fastvideo-kernel/pyproject.toml:30-33` | SM90 block-sparse, Ulysses all-to-all (`csrc/comm`) |
| `block_sparse_attn_ptx_blackwell` | native extension with a Python wrapper | Tile 64/128 VSA-H3 on datacenter Blackwell (`video_sparse_attn_h3.py:84,559`) |
| `attn_qat_infer/blackwell/*.cu` | same CMake build | FP4 attention (`ATTN_QAT_INFER` backend) |
| FlashInfer `mm_fp4`, `nvfp4_quantize` | external package | NVFP4 linears (`nvfp4_config.py:43-56,308`) |

## 5. Restart actions and what they keep

From the cheapest. Overheads are measured at 1 GPU (`../kb/costs.md`, "Restart ladder", which
also has the 2-rank numbers and the time to a first video).

| Action | Keeps | Covers | Overhead |
|---|---|---|---|
| Live swap (`install`) | everything | OP: a function body, on every rank, between requests | install 0.06 s, no restart |
| In-place rebuild (`rebuild`, `livelab/services/rebuild.py`) | process, CUDA context, weights, JIT and compile state | STATE without new tensors: constructor edits, new fields, module constants, new classes; releases CUDA graphs the old code held | 0.19-1.27 s |
| Zygote respawn with weights holder (`livelab/services/holder.py`) | imports (forked), DiT, text encoder and VAE on the GPU | any Python edit outside the loader, quantization and model construction code (the holder refuses when those changed) | 4.8-5.6 s to ready |
| Zygote respawn (`livelab/services/zygote.py`) | imports, refreshed for edited modules | any Python edit, RESOURCE included | 14.1-15.3 s to ready |
| Plain respawn (`reset("respawn")`) | the lab host process | any Python edit | 43-45 s to ready |
| Warm restart with path-stable caches (`livelab/services/jitcache.py`) | page cache, JIT caches | any edit, env vars and config included | 51.4 s to ready |
| Cold restart | nothing | native code (`fastvideo-kernel`), anything | 289 s to ready |

GRAPH edits: a Triton kernel body re-JITs on its next call after a live swap of the code that
launches it; a `torch.compile` region recompiles on the next call after an in-place rebuild.
RANKS edits go to every rank in one transaction or to none. NATIVE edits need a restart unless
built as a versioned side library and rebound.

## 6. How the live lab maps onto this

The lab host owns a `MultiprocExecutor` worker group, one worker per rank, at every GPU count
(`../livelab/README.md`). Each worker runs the agent (`livelab/adapter/agent.py`), which
installs edits through the transaction kernel (`livelab/kernel/`) in the gap between requests.
The services (`livelab/services/`) are the restart actions above below a plain respawn: each
keeps one more box warm (weights, imports, JIT caches) and refuses when its box is stale.
