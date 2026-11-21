"""The box daemon: `Box` behind a standard-library HTTP server with a bearer token.

The server listens before the lab boots, then writes the box file (`<state>/box.json`: url,
token, pid, lab) that launchers wait for and clients read.
"""
import hmac
import http.server
import json
import os
import secrets
import socket
import threading
import urllib.parse

from . import box as B

BOX_FILE = "box.json"


def _handler(box, token, stop):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):          # the event list is the log
            pass

        def _send(self, status, payload):
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
                return True
            self._drain()
            self._send(401, {"ok": False, "error": "missing or wrong token"})
            return False

        def _drain(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _route(self):
            url = urllib.parse.urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            name = parts[1] if len(parts) == 2 and parts[0] == "v1" else None
            return name, urllib.parse.parse_qs(url.query)

        def do_GET(self):
            if not self._authorized():
                return
            name, query = self._route()
            if name == "status":
                return self._send(*box.call("status"))
            if name == "info":
                return self._send(200, box.info())
            if name == "events":
                try:
                    since = int((query.get("since") or ["0"])[0])
                except ValueError:
                    return self._send(400, {"ok": False, "error": "since must be an integer"})
                return self._send(200, box.events_since(since))
            if name == "receipts":
                return self._send(200, box.read_receipts())
            if name == "source":
                module = (query.get("module") or [""])[0]
                if not module:
                    return self._send(400, {"ok": False, "error": "module is required"})
                return self._send(*box.source(module))
            self._send(404, {"ok": False, "error": f"unknown route GET {self.path}"})

        def do_POST(self):
            if not self._authorized():
                return
            name, _ = self._route()
            raw = self._drain()
            try:
                body = json.loads(raw) if raw.strip() else {}
            except ValueError:
                body = None
            if not isinstance(body, dict):
                return self._send(400, {"ok": False, "error": "the body must be a JSON object"})
            if name == "close":
                reply = box.close()
                self._send(200, reply)
                return stop()
            if name in B.VERBS and name != "status":
                return self._send(*box.call(name, body))
            self._send(404, {"ok": False, "error": f"unknown route POST {self.path}"})

    return Handler


class Daemon:
    """`Daemon(box).start()` serves until `stop()` or `POST /v1/close`."""

    def __init__(self, box, bind="127.0.0.1", port=0, token=None, state_dir=None):
        self.box, self.state_dir = box, state_dir
        self.token = token or secrets.token_urlsafe(24)
        self.httpd = http.server.ThreadingHTTPServer((bind, port), _handler(box, self.token, self.stop))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        host = "127.0.0.1" if bind in ("127.0.0.1", "localhost") else socket.gethostname()
        self.url = f"http://{host}:{self.port}"
        self._thread = None

    def write_box_file(self):
        if not self.state_dir:
            return None
        os.makedirs(self.state_dir, exist_ok=True)
        path = os.path.join(self.state_dir, BOX_FILE)
        fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"url": self.url, "token": self.token, "pid": os.getpid(),
                       "host": socket.gethostname(), "port": self.port, "lab": self.box.spec}, handle)
        os.replace(path + ".tmp", path)
        return path

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="rsibox-http", daemon=True)
        self._thread.start()
        self.write_box_file()
        return self

    def stop(self):
        threading.Thread(target=self.httpd.shutdown, daemon=True).start()

    def wait(self):
        self._thread.join()
        self.httpd.server_close()
