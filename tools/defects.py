"""Break the code on purpose. The suite must fail for every defect. Run from the repo root."""

import subprocess
import sys

DEFECTS = [
    (
        "commit swaps nothing",
        "src/fastrsi/agent.py",
        "                entry[2].__code__ = entry[4]\n",
        "                pass\n",
    ),
    (
        "undo restores nothing",
        "src/fastrsi/agent.py",
        "        for entry in self.live:\n            entry[2].__code__ = entry[3]\n        self.live = []",
        "        self.live = []",
    ),
    (
        "no rollback in agent",
        "src/fastrsi/agent.py",
        "            for entry in done:\n                entry[2].__code__ = entry[3]\n            raise",
        "            raise",
    ),
    (
        "witness never armed",
        "src/fastrsi/agent.py",
        "        self.live = plan\n        self.arm()\n",
        "        self.live = plan\n",
    ),
    (
        "host skips undo on partial commit",
        "src/fastrsi/core.py",
        '            for path in committed:\n                try:\n                    call(path, "undo")',
        '            for path in []:\n                try:\n                    call(path, "undo")',
    ),
    (
        "identity check always passes",
        "src/fastrsi/core.py",
        'if not reply["ok"] or reply["functions"] != want:',
        "if False:",
    ),
    (
        "classifier allows signature change",
        "src/fastrsi/edits.py",
        '            raise Refused("signature", f"{name}: signature changed")',
        "            pass",
    ),
    (
        "hang never poisons",
        "src/fastrsi/core.py",
        '            self.poison(f"no result within {timeout} s")',
        "            pass",
    ),
    (
        "prepare skips one process",
        "src/fastrsi/core.py",
        "            targets = self._targets(list(files))\n            if not targets:",
        "            targets = self._targets(list(files))[:1]\n            if not targets:",
    ),
]


def main():
    survived = []
    for name, path, old, new in DEFECTS:
        with open(path) as handle:
            source = handle.read()
        if old not in source:
            raise SystemExit(f"{name}: target text not found in {path}")
        with open(path, "w") as handle:
            handle.write(source.replace(old, new))
        try:
            done = subprocess.run(
                ["uv", "run", "pytest", "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            with open(path, "w") as handle:
                handle.write(source)
        caught = done.returncode != 0
        print(f"{'caught  ' if caught else 'SURVIVED'} {name}")
        if not caught:
            survived.append(name)
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
