"""A fresh FastVideo process for `FastVideoLab.fresh_check`: load, run one check, print it.

Usage: python -m adapter.fresh SPEC.json   (SPEC: {"config": run config, "workload": ...}).
The caller puts the patched package first on PYTHONPATH. Prints one `FRESH {json}` line.
"""
import json
import sys
import time

from adapter.lab import build_request, force_multiproc, output_of


def main(path):
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    from fastvideo import VideoGenerator
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import RunConfig
    force_multiproc()
    t0 = time.perf_counter()
    gen = VideoGenerator.from_config(parse_config(RunConfig, spec["config"]).generator)
    load_s = time.perf_counter() - t0
    try:
        t1 = time.perf_counter()
        out, _, _ = output_of(gen.generate(build_request(spec["config"], spec.get("workload"))))
        print("FRESH " + json.dumps({"output": out, "load_s": round(load_s, 3),
                                     "seconds": round(time.perf_counter() - t1, 3)}), flush=True)
    finally:
        gen.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
