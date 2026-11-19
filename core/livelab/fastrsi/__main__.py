"""Entrypoint for `python3 -m fastrsi ...`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
