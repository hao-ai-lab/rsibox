"""Fresh-process runner for `ToyVerlLab.fresh_check`."""
import json
import sys

from adapter.toyverl.cluster import ToyVerlCluster


def main():
    with open(sys.argv[1], encoding="utf-8") as handle:
        spec = json.load(handle)
    cluster = ToyVerlCluster(spec["config"])
    try:
        cluster.capture_s0()
        result = cluster.run_steps(spec["workload"])
        print("FRESH " + json.dumps({"output": result}))
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()
