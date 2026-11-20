"""Run vLLM conformance or the Qwen live-edit walkthrough with the serving Python."""
import argparse
import os
import shlex
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    conformance = sub.add_parser("conformance", help="run the real vLLM conformance suite")
    conformance.add_argument("--model", required=True)
    conformance.add_argument("--work", required=True, help="private scratch directory")
    conformance.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    conformance.add_argument("--graphs", action="store_true",
                             help="allow CUDA graphs by turning off enforced eager execution")
    conformance.add_argument("--only", help="comma-separated conformance case names")
    conformance.add_argument("--receipts", help="keep the kernel receipts in this file")
    conformance.add_argument("--json", action="store_true")
    sub.add_parser("live-edit", add_help=False, help="run the Qwen walkthrough (see live-edit --help)")
    args, rest = parser.parse_known_args(argv)
    if args.command == "live-edit":
        script = Path(__file__).resolve().parents[1] / "examples/live-edit-qwen3/walkthrough.py"
        os.execv(sys.executable, [sys.executable, str(script), *rest])
    if rest or args.tp < 1:
        parser.error(f"unexpected arguments {rest}" if rest else "--tp must be positive")
    import adapter  # resolves the shared conformance harness
    from conformance.__main__ import main as run_cases

    os.environ.update(FASTRSI_VLLM_MODEL=args.model, FASTRSI_VLLM_WORK=args.work,
                      FASTRSI_VLLM_TP=str(args.tp))
    if args.graphs:
        from adapter.conformance_lab import DEFAULT_EXTRA
        flags = shlex.split(os.environ.get("FASTRSI_VLLM_EXTRA", DEFAULT_EXTRA))
        os.environ["FASTRSI_VLLM_EXTRA"] = shlex.join([*flags, "--no-enforce-eager"])
    opts = ["--lab", "adapter.conformance_lab:VllmLab"]
    for key in ("only", "receipts"):
        if getattr(args, key):
            opts.extend(["--" + key, getattr(args, key)])
    return run_cases(opts + (["--json"] if args.json else []))


if __name__ == "__main__":
    sys.exit(main())
