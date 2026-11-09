# FastVideo priors: MiniMax H3 / FastH3 inference

What we knew before the first live-lab run. Nothing here is a new measurement.

- **Workload:** FastH3-QAD-v2 checkpoint. NVFP4 DiT linears, BF16 video sparse attention
  (VSA_H3), NVFP4 Qwen3-VL text encoder, lightweight H3 VAE, audio VAE. 480x832, 124 frames,
  9 sigma points = 8 DiT forwards. 1, 2 or 4 GPUs (sequence parallel).
- **Pin:** FastVideo `376e445f0e8d192c2d9a5f28010a6b3a7012667e` ("FastH3 OmniRef inference on
  consumer devices (#1980)", committed 2026-10-19). All `file:line` below are at this pin
  unless marked otherwise. Link form:
  `https://github.com/hao-ai-lab/FastVideo/blob/376e445f0e8d192c2d9a5f28010a6b3a7012667e/<path>#L<n>`.
- **Older pins:** an earlier source survey (pin `02a027c`) and an earlier CPU pilot (pin
  `d5253fd`) studied the Wan path and some shared code. A fact from them is marked "older pin"
  and was re-checked against `376e445f` where the text says so.
- **Measured:** one earlier GB200 run (2026-10-20), 1 GPU, BF16 VSA: imports 65 s, first video
  53 s end to end (includes lazy compile/JIT), second video 6.1 s, denoise stage 5.5 s.
  Everything else is **unmeasured**.

## 1. Startup phases

Order on a cold start. "Warm" means the cost can stay resident in a running lab.

| # | Phase | What happens | Source | Keep warm? | Measured |
|---|---|---|---|---|---|
| 1 | Imports | `import fastvideo`, torch, Triton, flashinfer (NVFP4 imports flashinfer lazily at first use) | `layers/quantization/nvfp4_config.py:43-56` | Yes: one process | 65 s (earlier GB200 run, 1 GPU) |
| 2 | Executor and worker | 1 GPU: worker built in the caller process. N GPUs: one spawned process per rank | `worker/executor.py:63-71`, `worker/uniproc_executor.py:49-74`, `worker/multiproc_executor.py:570-600` | Yes | unmeasured |
| 3 | Distributed init | NCCL group, sequence-parallel group (`sp_size`) | `worker/gpu_worker.py:140-161` | Yes | unmeasured |
| 4 | Text encoder load | Qwen3-VL NVFP4 loaded first; DiT and VAEs deferred when sequential or lazy load is on | `pipelines/basic/minimax_h3/minimax_h3_pipeline.py:422-506` | Yes, but see note 1 | unmeasured |
| 5 | DiT load | Weights, then NVFP4 weight conversion (`_maybe_quantize_model`) | `models/loader/fsdp_load.py:543`, `fastvideo_args.py:499` | Yes, but see note 1 | unmeasured |
| 6 | VAE and audio VAE load | H3 VAE (optionally NVFP4 decoder), audio VAE | `pipelines/basic/minimax_h3/minimax_h3_pipeline.py:507-523` | Yes | unmeasured |
| 7 | Optional compile setup | Per-block compile wrappers attached; compile itself runs at the first call | `pipelines/composed_pipeline_base.py:271-294`, `models/loader/fsdp_load.py:572-600` | Wrappers: yes. Compiled code: no, per edit | unmeasured |
| 8 | Triton JIT and autotune | First call of each `@triton.jit` kernel per shape and constexpr set | section 3 | JIT cache on disk; autotune in process | inside the 53 s first video |
| 9 | First request | Lazy compile, JIT, flashinfer kernel setup, allocator warm-up | earlier GB200 run | Yes after the first request | 53 s first, 6.1 s second |

Notes:

1. **H3 can release and reload modules between requests.** With `lazy_module_load` or
   `h3_sequential_load`, the text encoder is freed after encode and the DiT and VAEs are loaded
   afterwards (`minimax_h3_pipeline.py:639-656`: encode, release text encoder, load denoise
   modules, run denoise and decode). Both default to on for unified-memory hosts and off
   otherwise (`minimax_h3_pipeline.py:422-448`, `fastvideo_args.py:1141`). On a discrete-GPU
   node, check the effective setting at startup; if on, module identity changes per request
   and a patch applied to a module instance is lost on reload (an earlier source survey made
   the same point for lazy ownership).
2. The earlier GB200 run is the only phase-level evidence: it does not split phases 2 to 8.
   The first costs.md measurement should split them.
3. Phase 8 on a restart: the Triton disk cache survives a restart; the in-process autotune
   table does not. Only `minimax_h3_sparse_int8.py:36` uses `@triton.autotune` on the H3 path
   (INT8 sparse attention, not the default BF16 VSA route).

## 2. Restart classes

Groups are from `core/livelab_overview.md`. "Live" means the engine process keeps running;
it does not mean the check is free. Nothing in this table has been run on FastVideo yet.

| Edit | Group | Live or restart | Why (source) |
|---|---|---|---|
| Stage Python (`pipelines/basic/minimax_h3/stages/*.py`) | OP | Live | Stage objects are built once and called through `_stage_name_mapping[name](batch, args)` (`minimax_h3_pipeline.py:639-656`). Swap the method on the class or instance. Stages are rebuilt only if `_denoise_stages_ready` is false |
| Denoise loop body (`minimax_h3_denoising.py:298-360`) | OP | Live, takes effect on the next request | The loop runs inside one `forward` call. A swap during a request does not reach the running call |
| DiT block `forward` (`minimax_h3.py:677-749`), attention module `forward` (`:352`) | OP, or GRAPH if compiled | Live when eager (default). If `inference_torch_compile` is on, the bound `forward` is wrapped by `torch.compile` per block (`fsdp_load.py:823`): swapping Python does not change the compiled graph | `fsdp_load.py:572-600` attaches the compile wrapper after load. Recompile the affected blocks |
| Fused Triton helpers (`models/dits/minimax_h3_fusions/*.py`) | OP for the wrapper, GRAPH for the kernel | Live for the wrapper. A kernel body edit re-JITs on its next call | `@triton.jit` at `exact.py:45-127`, `modulation.py:33,72`, `qknorm_rope.py:23`, `swiglu.py:36`. Which helper runs is chosen once per DiT at construction from the env (`minimax_h3.py:794`, `:805`), so enabling a fusion that was off needs the DiT blocks rebuilt (STATE) or a restart |
| Attention backend selection (`attention/selector.py`, `MiniMaxH3Attention.__init__`) | STATE | Restart, or rebuild the attention modules | The backend class is chosen in `MiniMaxH3Attention.__init__` (`minimax_h3.py:252-262`) and held by `distributed_attention.attn_impl`. The resolver is `@cache`d on all its inputs (`selector.py:275`), so it needs no cache clear but existing layers keep their backend. The attention instructions in the repo forbid changing the backend env var mid-process (older pin, from an earlier CPU pilot; re-check `attention/AGENTS.md` at 376e445f) |
| VSA-H3 impl (`attention/backends/video_sparse_attn_h3.py`: `forward` `:783`, `tile` `:682`, metadata builder `:398`) | OP | Live in eager mode | Per-call Python. The tile buffer holder is state owned by the metadata builder (`:350-393`); a swap that changes buffer shape or dtype must drop the holder |
| VSA Triton kernels (`fastvideo_kernel/triton_kernels/*`, `block_sparse_attn.py`) | GRAPH (re-JIT) | Live | Imported by function at call time (`block_sparse_attn.py:165`), so a reload plus re-JIT takes effect. Installed library lives outside the engine tree: patch the installed copy or rebind |
| Native `fastvideo-kernel` CUDA/CuTe (`csrc`, `attn_qat_infer/blackwell/*.cu`, `block_sparse_attn_ptx_blackwell.py`) | NATIVE | Live only as a versioned side library plus rebind; otherwise restart | Built by scikit-build-core/CMake (`fastvideo-kernel/pyproject.toml:30-33`, `build.sh`). A loaded shared library cannot be unloaded. `_C.fastvideo_kernel_ops` is looked up at call time (`block_sparse_attn.py:15-22`) |
| Sequence-parallel all-to-all (`attention/layer.py:148,188,244,270`, `distributed/device_communicators/*`) | RANKS | All ranks or none | Every rank must make the same collective calls in the same order. An edit between two collectives, or a different revision on one rank, hangs or corrupts |
| Loader and quantization (`models/loader/fsdp_load.py`, `layers/quantization/nvfp4_*.py`, `_maybe_quantize_model`) | RESOURCE | Restart, or reload one module | Load-time layout: packed FP4 weights and scales are built once from BF16. `NVFP4QuantizeMethod.apply` (`nvfp4_config.py:635`) is per-call and can be swapped as OP |
| VAE decode code (`models/vaes/minimax_h3_*.py`) | OP, GRAPH if graphed | Live; NVFP4 fused decoder keeps CUDA graphs per shape | `minimax_h3_nvfp4_fused.py:596-663` caches graphs (`GRAPH_CACHE_LIMIT = 2` at `:591`). Drop `blocks._nvfp4_cuda_graphs` after a swap |
| Pipeline config (`configs/pipelines/minimax_h3.py`), `FastVideoArgs` fields | STATE | Restart | Parsed once; `_apply_transformer_quant` and checkpoint arch configs run at init (`minimax_h3_pipeline.py:162-186,301`) |
| Env vars (`FASTVIDEO_*`, `envs.py`) | none | Restart | Fusion and exact-kernel sets are read at DiT construction (`minimax_h3.py:794-805`). Some flags are read per call (for example `FASTVIDEO_H3_VAE_TILE_PARALLEL`, `minimax_h3_video.py:1130`), but a changed env var is not an installable edit |

Reading rule for this table: if the edit changes what a module builds at construction, it is
STATE or RESOURCE and the module must be rebuilt. If it changes what a method does per call,
it is OP. Only the first request after an install proves it ran; use the lab receipt.

## 3. What is compiled or captured on the H3 path

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
size; the first live-lab run should record the set that actually launched. Unmeasured.

### CUDA graphs

- **DiT path: none found at `376e445f`.** Search for `CUDAGraph`, `cuda.graph`, `reduce-overhead`
  under `models/dits/minimax_h3*` and `attention/backends/*h3*` finds only comments. The
  denoise loop calls `torch.compiler.cudagraph_mark_step_begin()` each step
  (`minimax_h3_denoising.py:326`); it matters only if a user sets
  `torch_compile_kwargs={"mode": "reduce-overhead"}`. The earlier source survey (older pin
  `02a027c`) also found no DiT graphs; this re-check agrees. Inductor with
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

## 4. Process layout

| | 1 GPU | 2 or 4 GPUs |
|---|---|---|
| Executor | `UniprocExecutor` (`executor.py:63-66`; also chosen when backend is `mp` and `num_gpus == 1`) | `MultiprocExecutor` (`executor.py:67-69`) |
| Worker | In the caller's process (`uniproc_executor.py:63-69`) | One spawned process per rank, named `FVWorkerProc-<rank>` (`multiproc_executor.py:570-600`; spawn is forced, `utils.py:1273-1288`) |
| Request loop | Caller thread calls `execute_forward` | Parent sends `execute_forward` over a pipe; each worker runs `worker_busy_loop` (`multiproc_executor.py:797-830`) |
| Run a function in the worker | `collective_rpc(callable)` (`uniproc_executor.py:230-242`) runs it on the caller thread | `collective_rpc` sends `{"method", "args", "kwargs"}` to each pipe and waits on every `recv` (`multiproc_executor.py:346-374`) |

`collective_rpc` details:

- A callable `method` is called as `method(worker_wrapper, *args)` (`utils.py:816-835`). A
  `bytes` value is `cloudpickle.loads(...)` first, so send a function defined in a script as
  cloudpickle bytes; a plain pickled function from `__main__` will not resolve in a spawned
  worker.
- `timeout` is ignored by both executors (`uniproc_executor.py:240`; the multiprocess `recv`
  is unbounded: an earlier CPU pilot at the older pin `d5253fd` found this, re-check at
  `376e445f` before relying on a deadline). A hung patched call blocks the whole RPC.
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
  the only place inside one request where Python state is quiescent. It is not a hook
  today: an install there needs a new pause point. Unmeasured, not yet built.
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

## 5. Real H3 performance PRs to replay

Source: `gh pr view`/`gh pr list` on `hao-ai-lab/FastVideo`, 2026-11-07. "In pin" means the
change is in the `376e445f` tree (merged before 2026-10-19 and grep-checked); in-pin PRs are
replayed by revert then reapply, the rest by applying forward. Function group and live-ness are
read from the file list and diff, not run.

| PR | Title (short) | Merged | Files (main) | Group | Live or restart | In pin |
|---|---|---|---|---|---|---|
| #1828 | uint8 frames from the decode worker | 09-25 | `minimax_h3_decoding.py`, `video_generator.py` | OP | Live (decode stage and parent conversion) | Yes |
| #1858 | VSA-H3 tile buffer out of autograd | 09-25 | `video_sparse_attn_h3.py` | OP, STATE (holder) | Live; drop the holder | Yes |
| #1882 | Reuse cached VAE offload for H3 conditioning | 09-25 | `minimax_h3_latent_preparation.py` | OP | Live (uses `pinned_offload`, `:206`) | Yes |
| #1735 | Opt-in Sol-Engine fusions (modulate, qknorm_rope, swiglu) | 08-21 | `minimax_h3.py`, `minimax_h3_fusions/*.py` | OP, GRAPH (Triton), STATE (flag read at init) | Wrapper live; enabling needs block rebuild | Yes |
| #1742 | Opt-in packed-varlen FA4 | 08-23 | `flash_attn.py`, `minimax_h3.py` | STATE (backend choice) | Restart or rebuild attention | Yes |
| #1744 | Parallelize VAE over sequence ranks | 08-23 | `minimax_h3_parallel.py`, `minimax_h3_video.py`, decode/latent stages | RANKS | All ranks together | Yes |
| #1793 | Sequential start with GPU-direct DiT load | 08-31 | `fsdp_load.py`, `minimax_h3_pipeline.py`, `fastvideo_args.py` | RESOURCE, STATE | Restart | Yes |
| #1807 | Harden Ulysses agreement, bounded H3 transfers | 10-07 | `ulysses_all_to_all.cu`, `comm_ops.py`, `ulysses_a2a.py` | NATIVE, RANKS | Side library plus all ranks, else restart | Yes |
| #1900 | Fuse VSA64 tile layout for SM100 inference | 10-07 | `triton_kernels/vsa_tile_layout.py`, `video_sparse_attn.py` | GRAPH (Triton) | Live (re-JIT) | Yes |
| #1930 | Exact Triton RoPE/AdaLN/residual/SwiGLU, heads-first tiles, VAE tile parallelism | 10-07 | `minimax_h3_fusions/exact.py`, `minimax_h3.py`, `video_sparse_attn_h3*.py`, `minimax_h3_video.py`, `memo.py` | OP, GRAPH, RANKS (VAE tile split) | Mixed: kernels live; flags read at init need rebuild | Yes |
| #2040 | Decode eight VAE spatial tiles per call by default | 11-02 | `minimax_h3_video.py`, `minimax_h3_parallel.py`, `envs.py` | OP, config default | Live code; default change is an env/config restart | No (pin has `FASTVIDEO_H3_VAE_TILE_BATCH` default 1, `envs.py:711`) |
| #2041 | Fuse VAE decoder QK RMSNorm and partial RoPE | 11-02 | `minimax_h3_qk_norm_rope.py` (new), `minimax_h3_video.py` | OP, GRAPH (Triton) | Live (re-JIT); enabling flag is an env var | No |
| #2042 | Opt-in faster DiT fused-op tier (`mk`) | 11-01 | `minimax_h3_fusions/mk/*`, `minimax_h3.py`, `attention/layer.py` | OP, GRAPH (Triton), STATE | Kernels live; tier choice at init | No |
| #2046 | Faster FP4 attention padding, uint8 VAE path, cuDNN FP4 GEMMs | 11-02 | `flash_attn.py`, `cudnn_sdpa.py`, `nvfp4_config.py`, decoding stage | OP, RESOURCE (GEMM backend) | Mostly live; backend env var needs restart | No |
| #2052 | Fused Q/K preconditioning + NVFP4 quantize for FA4 (bit-identical) | 11-03 | `flash_attn.py`, `attention/utils/nvfp4_fused_qk_quant.py` | OP, GRAPH (Triton) | Live (re-JIT) | No |

Notes on the table:

- Merge dates are 2026; PRs merged after the pin date (2026-10-19) are absent by construction.
  "In pin" for the others was checked by grepping for a mark of the change (for example
  `_tile_heads_first`, `FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT`, `pinned_offload.load`,
  `minimax_h3_fusions/exact.py`, `comm_ops.py`); a PR's whole diff was not compared.
- Several PRs touch `minimax_h3.py` and `flash_attn.py`: replay them one at a time on a
  clean tree, since later PRs depend on earlier ones.
- PR bodies carry the authors' own end-to-end numbers on other setups (for example #2042 reports
  DiT 16.54 s to 15.70 s at 768x1344, 6 forwards, 1 GB200). They are not our workload and are
  not priors for it.
- Replay candidates with the most DiT-path weight for the 8-forward workload: #1735, #1930,
  #2042 (fusions), #1900 and #1930 (VSA layout and tiles), #2052 and #2046 (FP4 attention and
  GEMMs). Decode-side: #1828, #2040, #2041.

## Not verified

- No phase-level startup split exists; only the one earlier GB200 run.
- Which Triton kernels launch in the default FastH3-QAD-v2 BF16-VSA path.
- Effective `lazy_module_load` and `h3_sequential_load` on a discrete-GPU node.
- `timeout` behavior of `collective_rpc` at `376e445f` (known at an older pin).
- Whether FlashInfer builds kernels by JIT on first NVFP4 call, and where it caches them.
- PR "in pin" status beyond the marker greps above.
