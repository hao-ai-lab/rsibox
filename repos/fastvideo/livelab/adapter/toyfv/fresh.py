"""Fresh process for the toy lab's fresh_check: start workers from the overlay, run, print."""
import json
import sys

from adapter.toyfv.engine import Generator


def main(path):
    spec = json.load(open(path))
    g = Generator(spec["config"]["ranks"])
    try:
        print("FRESH " + json.dumps({"output": g.generate(spec["workload"] or {"seed": 1, "frames": 4})}))
    finally:
        g.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
