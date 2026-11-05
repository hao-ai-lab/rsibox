# Live lab dashboard

A browser page on a real running `vllm serve`. It shows startup phases as the engine log prints
them, and it can hot-patch the sampler, send prompts, and undo the patch, all without a restart.

```
browser  <-- /events (server-sent events) --  server.py  --  vllm serve (dev mode + lab extension)
```

- `server.py` starts the engine with the cache root (`startup/caches.py`) and the lab extension
  (`fastrsi_lab_ext.LabExtension`). It follows the engine log with the marks in
  `startup/phases.py` and turns them into events: `engine.start`, `phase.end`, `cache.diff`,
  `engine.ready`, `patch.install`, `patch.uninstall`, `request.done`, `witness`, `error`.
- `index.html` draws those events. The browser can only call fixed actions (start, stop,
  install, undo, send a prompt). It cannot send code or commands.
- The demo patch adds one line to `apply_all_penalties` in `vllm/v1/sample/ops/penalties.py`
  that bans one word. That function runs on every sampling step outside CUDA graphs, so the
  change takes effect on the next request. Requests set a tiny repetition penalty (1.0001) so
  the function runs on the base code too.

## Run

```bash
python3 server.py --model Qwen/Qwen3-8B --served-name qwen3-8b \
    --cache-root <cache dir> --log-dir <log dir> --gpus 0 --port 8766
```

Then open `http://127.0.0.1:8766/`. The server and the engine listen on 127.0.0.1 only: dev
mode lets any local caller run code in the engine.

## Walkthrough (real output)

Qwen3-8B, real weights, one B200, vLLM 0.15.1, `--max-model-len 4096`, 2026-11-05.

| Step | Result |
|---|---|
| Start, empty cache root | ready in 70.1 s; 480 new kernel cache files |
| Stop, start again | ready in 32.9 s; 0 new files; graph capture 21.7 s → 3.4 s |
| Prompt "The capital of France is", base | " Paris. The capital of Italy is Rome. …" |
| Install, ban "Paris" | 2.5 ms; code hash `c0e7435a5832` → `d93b6e62441c` |
| Same prompt, patched | " a city in the country of France. …" |
| Witness | the patched function ran 32 times (one per generated token) |
| Undo | 2.4 ms; hash back to `c0e7435a5832` |
| Same prompt, base again | identical to the first base answer |

Phases of the warm start: import and config 12.9 s, weights 4.5 s, profiling and KV memory
10.2 s, graph capture 3.4 s, server ready 1.8 s.

## Limits

- The ban drops the first token of each spelling of the word. For a word that splits into
  several tokens, that first piece may be a common fragment, so other words get banned too.
- One engine per server, TP from extra `vllm serve` arguments; tested with one GPU only.
- Phase seconds are wall time between log lines as the server reads them, not log timestamps.
