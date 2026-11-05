"""Turn a vLLM engine log into a startup timeline: which phase took how long.

Usage: python3 phases.py <engine.log>  -> prints one JSON object.
Phases come from log lines vLLM already prints; nothing is instrumented.
Standard library only.
"""
import json, re, sys

TS = re.compile(r"INFO (\d\d-\d\d) (\d\d):(\d\d):(\d\d)")
MARKS = [  # (phase that ENDS at this line, regex)
    ("engine_init", re.compile(r"Initializing a V1 LLM engine")),
    ("model_load", re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")),
    ("graphmem_profile", re.compile(r"Available KV cache memory")),
    ("jit_warmup", re.compile(r"JIT kernel warmup finished in ([\d.]+)s")),
    # Autotune runs once per token-count group; the phase ends at the start of the last group.
    ("autotune", re.compile(r"Running FlashInfer autotune with \d+ tokens")),
    ("graph_capture", re.compile(r"Graph capturing finished in (\d+) secs")),
    ("ready", re.compile(r"Application startup complete")),
]


LAST = {"graph_capture", "autotune"}


def seconds(m):
    d, h, mi, s = m.groups()
    return int(d.split("-")[1]) * 86400 + int(h) * 3600 + int(mi) * 60 + int(s)


def parse(path):
    first, last_t, marks = None, None, {}
    for line in open(path, errors="replace"):
        m = TS.search(line)
        if m:
            last_t = seconds(m)
            first = first if first is not None else last_t
        for name, rx in MARKS:
            # graph_capture: the last match (the first is the memory-profiling pass)
            if (name not in marks or name in LAST) and rx.search(line):
                marks[name] = {"t": last_t, "detail": rx.search(line).groups() or None}
    out, prev = {"start_t": first}, first
    for name, _ in MARKS:
        if name in marks and marks[name]["t"] is not None:
            out[name] = {"seconds": marks[name]["t"] - prev, "detail": marks[name]["detail"]}
            prev = marks[name]["t"]
    out["total_seconds"] = (prev - first) if first is not None else None
    return out


if __name__ == "__main__":
    print(json.dumps(parse(sys.argv[1]), indent=1))
