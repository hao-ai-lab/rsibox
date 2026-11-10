#!/usr/bin/env python3
"""Serve the dashboard folder read-only on 127.0.0.1:8742 (stdlib only).

  python3 dashboard/serve.py [--port 8742] [--build]

--build runs build.py first. The server binds to 127.0.0.1 only and answers GET/HEAD.
"""
import argparse
import functools
import http.server
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):  # read-only
        self.send_error(405)

    do_PUT = do_DELETE = do_PATCH = do_POST


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8742)
    ap.add_argument("--build", action="store_true", help="run build.py before serving")
    a = ap.parse_args()
    if a.build or not os.path.exists(os.path.join(HERE, "dashboard.json")):
        subprocess.run([sys.executable, os.path.join(HERE, "build.py")], check=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=HERE))
    print(f"serving {HERE} on http://127.0.0.1:{a.port}/ (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
