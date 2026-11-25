"""Client for a box: one HTTP request per call, standard library only.

    box = Client.from_env()            # RSIBOX (a box file) or RSIBOX_URL + RSIBOX_TOKEN
    box.wait(); box.install({"files": {"pkg.mod": new_text}}); box.check()

Replies are the daemon's JSON with `http` (the status code) added. A request that cannot reach
the daemon returns `{"ok": False, "http": 0, "error": ...}` instead of raising.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

POSTS = ("install", "check", "reset", "uninstall", "close")


class Client:
    def __init__(self, url, token, timeout=600.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    @classmethod
    def from_box_file(cls, path, **kw):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data["url"], data["token"], **kw)

    @classmethod
    def from_env(cls, env=None, **kw):
        env = os.environ if env is None else env
        if env.get("RSIBOX"):
            return cls.from_box_file(env["RSIBOX"], **kw)
        if env.get("RSIBOX_URL") and env.get("RSIBOX_TOKEN"):
            return cls(env["RSIBOX_URL"], env["RSIBOX_TOKEN"], **kw)
        raise ValueError("no box: set RSIBOX to a box file, or RSIBOX_URL and RSIBOX_TOKEN")

    def request(self, name, body=None, query=None, timeout=None):
        url = f"{self.url}/v1/{name}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body or {}).encode() if name in POSTS else None
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return {**json.loads(resp.read()), "http": resp.status}
        except urllib.error.HTTPError as exc:
            try:
                try:
                    payload = json.loads(exc.read())
                except ValueError:
                    payload = {"ok": False, "error": str(exc)}
                return {**payload, "http": exc.code}
            finally:
                exc.close()
        except (OSError, ValueError) as exc:
            return {"ok": False, "http": 0, "error": f"{type(exc).__name__}: {exc}"}

    # -- verbs ---------------------------------------------------------------------------
    def install(self, patch):
        return self.request("install", {"patch": patch})

    def check(self, workload=None):
        return self.request("check", {"workload": workload} if workload is not None else {})

    def reset(self, kind="state"):
        return self.request("reset", {"kind": kind})

    def uninstall(self, id=None):
        return self.request("uninstall", {"id": id} if id else {})

    def status(self):
        return self.request("status")

    def info(self):
        return self.request("info", timeout=10)

    def events(self, since=0):
        return self.request("events", query={"since": since}, timeout=10)

    def receipts(self):
        return self.request("receipts")

    def source(self, module):
        return self.request("source", query={"module": module})

    def close(self):
        return self.request("close")

    def wait(self, timeout=1800.0, poll_s=0.5):
        """Block until the box leaves `booting`. Returns the last `info`."""
        deadline, info = time.monotonic() + timeout, {"ok": False, "error": "not asked yet"}
        while time.monotonic() < deadline:
            info = self.info()
            if info.get("http") == 200 and info.get("state") != "booting":
                return {**info, "ok": info.get("state") == "ready"}
            time.sleep(poll_s)
        return {**info, "ok": False, "error": info.get("error") or f"still booting after {timeout} s"}
