# Services: pre-import fork server (zygote) for SGLang

Starting `sglang.launch_server` from cold Python pays module import overhead (`torch`,
`sglang.srt.*`, tokenizers, and HTTP server dependencies) before argument parsing or server
setup begins. `zygote.py` imports those modules once in a CUDA-free parent process, freezes
the GC heap (`forksafe.freeze_gc_before_fork`), listens on a short `AF_UNIX` socket, and starts
each server (or callable entry) via `os.fork()`.

In SGLang (`python/sglang/srt/entrypoints/engine.py`, `_set_envs_and_config` and
`_launch_subprocesses`), the engine forces `mp.set_start_method("spawn", force=True)` before
spawning `run_scheduler_process` and `run_detokenizer_process`. As a result, the forked
HTTP/tokenizer server process skips its own Python imports while scheduler subprocesses still
start cleanly via `spawn`.

## Audit guards

Every guard in `zygote.py` corresponds to a failure mode observed in sibling engine labs:

- **(a) Pre-fork `CUTE_DSL_ARCH` and `cuInit` stub (`install_guards`, `child_main`).**
  `nvidia-cutlass-dsl` (imported transitively through FlashInfer/CUTLASS) calls `cuInit` at
  import time to detect the target architecture when `CUTE_DSL_ARCH` is unset. Any process
  that runs `cuInit` poisons `fork` for CUDA in all children. Before importing the preload
  list, `install_guards` resolves the GPU compute capability via `forksafe.triton_capability()`
  (`RSIBOX_ZYGOTE_CC` or NVML, never the CUDA runtime/driver), sets `CUTE_DSL_ARCH` (for
  example `sm_100a`), and stubs `cuda.bindings.driver.cuInit` to return
  `CUDA_ERROR_NOT_INITIALIZED` until `child_main` restores it after `os.fork()`.
- **(b) `forksafe` pre-import guards, child `unpatch`, and `CUDA_VISIBLE_DEVICES`
  (`install_guards`, `child_main`).** Before heavy imports, `install_guards` sets
  `PYTORCH_NVML_BASED_CUDA_CHECK=1` and installs `forksafe.install(torch.cuda)`,
  `forksafe.install_properties(torch.cuda)`, and `forksafe.pin_triton_target()` so import-time
  device capability, device property, and Triton target queries do not initialize CUDA.
  `preload` verifies after every module that neither `torch.cuda.is_initialized()` nor
  `libcuda.so.1` (`cuDeviceGetCount != CUDA_ERROR_NOT_INITIALIZED`) was triggered. In the
  forked child, `child_main` calls `forksafe.unpatch(None)`, clears variables injected by the
  zygote, and applies the full request `env` (including `CUDA_VISIBLE_DEVICES`; an early vLLM
  zygote filtered request `env` down to volatile keys and silently dropped the requested GPU
  selection).
- **(c) Post-fork module and `linecache` refresh (`refresh_child`).** When `.py` files under
  `--refresh` (default `sglang`) change on disk after preload, the child drops those packages
  from `sys.modules`, unbinds parent attributes, removes stale `.pyc` files, and evicts every
  `linecache.cache` file entry whose cached text differs from the current file on disk (including
  `mtime=None` pinned entries that `linecache.checkcache()` skips), followed by
  `linecache.checkcache()`. Without this, re-imported functions execute new bytecode while
  `@triton.jit` and `inspect.getsource` compile or inspect stale pre-edit lines from
  `linecache` (Megatron overlay bug). Modified files outside `--refresh` (or non-`.py`
  extensions) refuse the fork with a `stale` finding unless `accept` includes `"stale"`.
- **(d) Child tracking/reaping by `ppid` and `AF_UNIX` path cap (`children`, `reap_exited`,
  `kill_children`, `sock_path_error`).** Each forked child calls `os.setsid()` to lead its own
  session, leaving the zygote's process group; scanning by process group misses every forked
  child (Megatron M-H17). `children()` enumerates direct children strictly by `ppid` (`/proc`
  on Linux, `ps -A -o pid=,ppid=` on macOS/BSD), `reap_exited()` reaps exited children via
  `waitpid(-1, WNOHANG)`, and `kill_children()` signals each child and its session group.
  `sock_path_error()` enforces `len(path) <= MAX_SOCK_PATH` (`< 107` bytes for `sun_path`)
  before `preload()` runs so overlong socket paths fail immediately instead of after imports.

## Files

| File | Responsibility |
|---|---|
| [`__init__.py`](__init__.py) | Package marker for `services`. |
| [`zygote.py`](zygote.py) | Pre-import server (`serve`), client RPC (`call`, `request`), pre-fork guards (`install_guards`, `preload`), child refresh/exec (`refresh_child`, `child_main`), and `ppid` process supervision (`children`, `reap_exited`, `kill_children`). |
| [`test_zygote.py`](test_zygote.py) | CPU `unittest` suite exercising the real `serve` + `AF_UNIX` + `os.fork()` path across all four guard groups without requiring `torch` or `sglang`. |

## Usage

```bash
# Start the zygote with a short socket path (< 107 bytes)
python3 repos/sglang/livelab/services/zygote.py serve --sock /tmp/sgl-zyg.sock

# Fork an sglang.launch_server child on specific GPUs
python3 repos/sglang/livelab/services/zygote.py fork --sock /tmp/sgl-zyg.sock \
  --env CUDA_VISIBLE_DEVICES=0,1 --log /tmp/sgl-server.log -- \
  --model-path Qwen/Qwen3-8B --tp-size 2 --port 30000

# Inspect active child PIDs / exit codes, kill children, or stop the zygote
python3 repos/sglang/livelab/services/zygote.py ping --sock /tmp/sgl-zyg.sock
python3 repos/sglang/livelab/services/zygote.py kill --sock /tmp/sgl-zyg.sock
python3 repos/sglang/livelab/services/zygote.py stop --sock /tmp/sgl-zyg.sock
```

From Python (`adapter/lab.py` or tests), `zygote.request(sock_path, argv=..., env=..., log=..., entry=...)`
accepts either default module execution (`{"module": "sglang.launch_server"}`) or a direct
callable spec (`{"callable": "pkg.module:fn", "args": [...], "kwargs": {...}}`).

## Tests

From the repository root (standard library `unittest`, CPU only):

```bash
python3 -m unittest discover -s repos/sglang/livelab/services -t repos/sglang/livelab -v
```
