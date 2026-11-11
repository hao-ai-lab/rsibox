"""Launch flags: the toy lab declares this file restart-only."""
import argparse
import os

DEFAULT_PORT = int(os.environ.get("TOY_PORT", "8000"))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=DEFAULT_PORT)
    return p


def serve(args):
    return args.port
