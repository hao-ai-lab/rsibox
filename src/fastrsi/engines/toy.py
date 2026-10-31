"""The template plugin: a CPU engine with a driver and two workers. Runs in CI."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from fastrsi.engine import Engine, Spec, agent_env, rss_kib

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toyapp")


class Toy(Engine):
    name = "toy"

    def __init__(self, workdir, **options):
        super().__init__(workdir, **options)
        self.app = os.path.join(self.workdir, "app")
        self.process = None
        self.pid_file = os.path.join(self.workdir, "pid")

    def root(self):
        return self.app

    def start(self, agent=True, overlay=None):
        os.makedirs(self.workdir, exist_ok=True)
        shutil.rmtree(self.app, ignore_errors=True)
        shutil.copytree(APP, self.app, ignore=shutil.ignore_patterns("__pycache__"))
        for item in overlay.files if overlay else ():
            with open(os.path.join(self.app, item.path), "w") as handle:
                handle.write(item.new)
        directory = self.new_agent_dir()
        self.socket = os.path.join(directory, "engine")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if agent:
            env.update(agent_env(directory))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [self.app, env.get("PYTHONPATH")]))
        with open(os.path.join(self.workdir, "engine.log"), "ab") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-u", os.path.join(self.app, "toyserver.py"), self.socket],
                env=env, stdout=log, stderr=log, start_new_session=True)
        with open(self.pid_file, "w") as handle:
            handle.write(str(self.process.pid))
        deadline = time.time() + 20
        while not os.path.exists(self.socket):
            if time.time() > deadline or self.process.poll() is not None:
                self.stop()
                raise RuntimeError("toy engine did not start")
            time.sleep(0.01)

    def _pid(self):
        try:
            with open(self.pid_file) as handle:
                return int(handle.read())
        except (OSError, ValueError):
            return None

    def stop(self):
        pid = self._pid()
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process:
            self.process.wait()
        if os.path.exists(self.pid_file):
            os.unlink(self.pid_file)
        self.remove_agent_dir()

    def alive(self):
        pid = self._pid()
        if pid is None:
            return False
        if self.process is not None:
            return self.process.poll() is None
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def _request(self, timeout, **fields):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(timeout)
            sock.connect(os.path.join(self.agent_dir, "engine"))
            sock.sendall(json.dumps(fields).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = sock.recv(1 << 16)
                if not chunk:
                    raise RuntimeError("engine closed the connection")
                data += chunk
        return json.loads(data)

    def exercise(self, timeout, xs=None):
        began = time.perf_counter()
        reply = self._request(timeout, xs=list(xs) if xs is not None else list(range(32)))
        reply["seconds"] = time.perf_counter() - began
        return reply

    def reset(self):
        return self._request(30.0, reset=True)["digest"]

    def _pids(self):
        pid = self._pid()
        out = subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True).stdout
        return [int(line) for line in out.split()]

    def stats(self):
        return {"rss_kib": rss_kib(self._pids())}

    def fresh(self, edit):
        other = Toy(os.path.join(self.workdir, "fresh"))
        other.start(overlay=edit)
        return other

    def spec(self):
        path = "toymodel.py"
        return Spec(
            noop=(path, "return x * 2 + 1", "return x + x + 1"),
            visible=(path, "return x * 2 + 1", "return x * 3 + 1"),
            multiple=((path, "return x * 2 + 1", "return x * 3 + 1"),
                      (path, "return [value + 0 for value in values]",
                       "return [value + 5 for value in values]")),
            refused={
                "signature": (path, "def forward(x):", "def forward(x, y=0):"),
                "module-level": (path, '"""The code that edits target."""',
                                 '"""The code that edits target."""\nLIMIT = 3'),
                "new-definition": (path, "def postprocess(values):",
                                   "def helper():\n    return 1\n\n\ndef postprocess(values):"),
                "syntax": (path, "return x * 2 + 1", "return x * ("),
            },
            chain=[(path, "return x * 2 + 1", f"return x * 2 + 1 + {n}") for n in range(1, 8)]
            + [(path, "self.total += value", "self.total += value + 1"),
               (path, "return [value + 0 for value in values]",
                "return [value + 1 for value in values]")],
            crash=(path, "return x * 2 + 1", "__import__('os')._exit(3)"),
            hang=(path, "return x * 2 + 1", "__import__('time').sleep(3600)"),
            witness_processes=2,
            exercise_timeout=3.0,
        )
