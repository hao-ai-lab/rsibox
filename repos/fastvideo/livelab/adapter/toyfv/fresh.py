"""Fresh process for the toy lab's fresh_check: start workers from the overlay, run, print."""
import json
import sys

from adapter.toyfv.engine import Generator


def main(path):
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    g = Generator(spec["config"]["ranks"])
    try:
        print("FRESH " + json.dumps({"output": g.generate(spec["workload"] or {"seed": 1, "frames": 4})}))
    finally:
        g.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
