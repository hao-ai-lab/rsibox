"""Run one workload check in a fresh process started from an overlaid package tree."""
import hashlib
import json
import sys


def run_once(spec):
    mode = spec.get("mode", "real")
    config = dict(spec.get("config") or {})
    workload = dict(spec.get("workload") or {})
    sampling = {"temperature": float(workload.get("temperature", 0.0)),
                "max_new_tokens": int(workload.get("max_new_tokens", 16))}
    if mode == "toy":
        from adapter.fixture.toysgl.srt.entrypoints.engine import Engine
        engine = Engine(**config)
        try:
            outputs = engine.generate(workload["input_ids"], sampling_params=sampling, timeout=30)
            output_ids = [item["output_ids"] for item in outputs]
            tp_size = config.get("tp_size", 2)
            return {f"rank{i}": output_ids for i in range(tp_size)}
        finally:
            engine.shutdown()
    import sglang
    sampling["ignore_eos"] = bool(workload.get("ignore_eos", True))
    engine = sglang.Engine(**config)
    try:
        outputs = engine.generate(prompt=workload["prompts"], sampling_params=sampling)
        token_ids = [item.get("output_ids") or item["text"] for item in outputs]
        digest = hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()[:16]
        return {"digest": digest, "tokens": token_ids}
    finally:
        engine.shutdown()


def main():
    with open(sys.argv[1]) as handle:
        spec = json.load(handle)
    print("FRESH " + json.dumps({"output": run_once(spec)}), flush=True)


if __name__ == "__main__":
    main()
