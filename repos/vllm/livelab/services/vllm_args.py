"""Read a `vllm serve` argv with vLLM's own parser, so vLLM stays the one source of truth for its
flags: the launcher passes the argv through verbatim and only reads back what it needs.

  parse(argv) -> {"ok": True, "args": {field: value}, "explicit": [fields set in argv],
                  "seconds": {"import", "build", "parse"}}
              or {"ok": False, "error": "<vLLM's argparse message>"}

Contract:
- The parser is `make_arg_parser(FlexibleArgumentParser())`, the one `vllm serve` builds, so
  aliases (`-tp`), `--x=v`, underscores, `--config` YAML files and dotted JSON flags read the
  same as in the server.
- model follows `vllm serve`'s rule (vllm/entrypoints/cli/serve.py): a positional model lands in
  `model_tag` and replaces `model`; reading `.model` alone gives vLLM's default model.
- "explicit" means given in the argv (or in its --config file), even when the value equals
  vLLM's default: `--load-format auto` is explicit. It is measured by parsing a second time
  into a namespace that holds a sentinel for each field; argparse leaves a field it did not see
  at the sentinel. The model is explicit when model_tag or --model is given.
- Importing vLLM's parser pulls torch, transformers and triton but does not initialize CUDA
  (test_vllm_args.py checks this in a fresh process). Cost on our host: import 4.7-4.9 s, build
  0.16 s, parse 1 ms. The zygote has these imported already (zygote.py, request "parse").
"""
import argparse, contextlib, io, sys, time

# What the launcher reads back: dest names of vLLM's parser.
FIELDS = ("model", "port", "host", "tensor_parallel_size", "load_format", "enforce_eager",
          "worker_cls", "model_loader_extra_config")
_UNSET = object()


def build():
    """vLLM's `vllm serve` parser and the seconds its import and build took. What the import
    and build print goes to stderr: vLLM's logger writes to stdout (with VLLM_LOGGING_LEVEL=DEBUG,
    vLLM 0.15.1 printed 16 lines at import), and the launcher's stdout is one JSON line."""
    t = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from vllm.entrypoints.common.cli_args import make_arg_parser
        except ImportError:                   # vLLM releases before the common/ split
            from vllm.entrypoints.openai.cli_args import make_arg_parser
        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser
        t1 = time.time()
        parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    return parser, {"import": round(t1 - t, 3), "build": round(time.time() - t1, 3)}


def parse(argv, parser=None):
    seconds = {}
    if parser is None:
        parser, seconds = build()
    t = time.time()
    out = io.StringIO()
    try:
        with contextlib.redirect_stderr(out), contextlib.redirect_stdout(out):
            ns = parser.parse_args(list(argv))
            seen = parser.parse_args(list(argv), namespace=argparse.Namespace(
                **{f: _UNSET for f in FIELDS}))
    except SystemExit:                        # argparse error (or --help): its last line
        lines = out.getvalue().strip().splitlines()
        return {"ok": False, "error": lines[-1] if lines else "vllm serve: exited parsing"}
    except Exception as e:                    # FlexibleArgumentParser raises ValueError too
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    args = {f: getattr(ns, f, None) for f in FIELDS}
    explicit = [f for f in FIELDS if getattr(seen, f, _UNSET) is not _UNSET]
    if getattr(ns, "model_tag", None) is not None:         # serve.py: model_tag wins
        args["model"] = ns.model_tag
        explicit = sorted(set(explicit) | {"model"}, key=FIELDS.index)
    seconds["parse"] = round(time.time() - t, 4)
    return {"ok": True, "args": args, "explicit": explicit, "seconds": seconds}


if __name__ == "__main__":
    import json
    print(json.dumps(parse(sys.argv[1:])))
