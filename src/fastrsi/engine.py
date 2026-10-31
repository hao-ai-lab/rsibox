"""The contract. A plugin answers engine questions; the session does the rest."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from fastrsi import edits

BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_boot")
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def agent_env(directory):
    """Environment that boots the agent in every Python child."""
    return {"FASTRSI_AGENT_DIR": directory,
            "PYTHONPATH": os.pathsep.join([BOOT, PACKAGE_ROOT])}


def rss_kib(pids):
    """Resident set, summed over pids that exist."""
    total = 0
    for pid in pids:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True)
        total += int(result.stdout.strip() or 0)
    return total


@dataclass
class Spec:
    """What the conformance suite needs from an engine. Each edit is (path, old, new)."""

    noop: tuple
    visible: tuple
    multiple: tuple  # at least two edits in different functions, applied as one
    refused: dict
    chain: list
    crash: tuple
    hang: tuple
    witness_processes: int
    exercise_timeout: float = 30.0
    exercise_args: dict = field(default_factory=dict)
    gpu: bool = False


class Engine:
    name = "engine"

    def __init__(self, workdir, **options):
        self.workdir = os.path.abspath(workdir)
        self.options = options
        self._agent_pointer = os.path.join(self.workdir, "agent-dir")

    @property
    def agent_dir(self):
        """A private directory of sockets. Short, because socket paths are limited to ~104 bytes."""
        with open(self._agent_pointer) as handle:
            return handle.read()

    def new_agent_dir(self):
        self.remove_agent_dir()
        os.makedirs(self.workdir, exist_ok=True)
        path = tempfile.mkdtemp(prefix="fastrsi-", dir="/tmp")
        with open(self._agent_pointer, "w") as handle:
            handle.write(path)
        return path

    def remove_agent_dir(self):
        if os.path.exists(self._agent_pointer):
            shutil.rmtree(self.agent_dir, ignore_errors=True)
            os.unlink(self._agent_pointer)

    def start(self, agent=True, overlay=None):
        """Launch to ready. overlay is an Edit already present in the fresh process's source."""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def alive(self):
        raise NotImplementedError

    def root(self):
        """Directory that edit paths are relative to, holding the base source."""
        raise NotImplementedError

    def exercise(self, timeout, **fields):
        """Run a deterministic workload. Return a dict with 'digest' and 'seconds'."""
        raise NotImplementedError

    def reset(self):
        """Return engine state to a known point. Return its digest."""
        raise NotImplementedError

    def stats(self):
        """Measured resident costs, such as rss_kib and gpu_mib."""
        return {}

    def classify(self, edit):
        return edits.classify(edit)

    def fresh(self, edit):
        """A separate engine started with the edit, for comparison. The caller stops it."""
        raise NotImplementedError

    def spec(self):
        raise NotImplementedError
