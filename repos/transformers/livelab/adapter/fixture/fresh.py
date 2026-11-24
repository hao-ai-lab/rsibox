"""Fresh subprocess runner for `TransformersLab.fresh_check()`."""
import json
import sys

from adapter import agent

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        spec = json.load(handle)
    print("FRESH " + json.dumps({"output": agent.generate_tokens(agent.load_backend(spec["config"]), spec["workload"])}))
