"""An engine in a Docker container, on one GPU, with no network."""

import hashlib
import os
import subprocess
import time

from fastrsi.engine import PACKAGE_ROOT, Engine, agent_env


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


class Container(Engine):
    image = ""
    site = ""
    sources = ()
    pins = {}
    server = ""

    def __init__(self, workdir, gpu="2", fresh_gpu="3", model=None, image=None):
        super().__init__(workdir)
        self.gpu, self.fresh_gpu = str(gpu), str(fresh_gpu)
        self.image = image or self.image
        self.model = model or os.environ.get("FASTRSI_MODEL")
        if not self.model:
            raise SystemExit("set -o model=<path> or FASTRSI_MODEL")
        tag = hashlib.sha256(self.workdir.encode()).hexdigest()[:10]
        self.container = f"fastrsi-{tag}"

    def root(self):
        return os.path.join(self.workdir, "src")

    def gpu_mib(self):
        out = sh("nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits").stdout
        return int(out.strip()) if out.strip() else -1

    def start(self, agent=True, overlay=None):
        if self.alive():
            raise RuntimeError("engine already running")
        sh("docker", "rm", "-f", self.container)
        if (used := self.gpu_mib()) > 100:
            raise RuntimeError(f"GPU {self.gpu} has {used} MiB in use; refusing to start")
        directory = self.new_agent_dir()
        for name in ("home", "tmp", "cache"):
            os.makedirs(os.path.join(self.workdir, name), exist_ok=True)
        with open(os.path.join(self.workdir, "passwd"), "w") as handle:
            handle.write(f"fastrsi:x:{os.getuid()}:{os.getgid()}::{self.workdir}/home:/bin/sh\n")
        env = {**self.pins, "HOME": f"{self.workdir}/home", "TMPDIR": f"{self.workdir}/tmp",
               "HF_HOME": f"{self.workdir}/cache", "XDG_CACHE_HOME": f"{self.workdir}/cache",
               "PYTHONDONTWRITEBYTECODE": "1"}
        if agent:
            env.update(agent_env(directory))
        command = ["docker", "run", "-d", "--name", self.container, "--network", "none",
                   "--gpus", f'"device={self.gpu}"', "--cpus=16", "--shm-size=4g",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{self.workdir}/passwd:/etc/passwd:ro",
                   "-v", f"{self.workdir}:{self.workdir}",
                   "-v", f"{directory}:{directory}",
                   "-v", f"{PACKAGE_ROOT}:{PACKAGE_ROOT}:ro",
                   "-v", f"{self.model}:/model:ro", "-w", f"{self.workdir}/tmp"]
        for item in overlay.files if overlay else ():
            path = os.path.join(self.workdir, "overlay", item.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(item.new)
            command += ["-v", f"{path}:{self.site}/{item.path}:ro"]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        command += ["--entrypoint", "/usr/bin/env", self.image, "python3", "-u", "-P", self.server,
                    os.path.join(directory, "engine")]
        result = sh(*command)
        if result.returncode:
            raise RuntimeError(result.stderr[-800:])
        deadline = time.time() + 420
        while not os.path.exists(os.path.join(directory, "engine")):
            if time.time() > deadline or not self.alive():
                log = self.logs()
                self.stop()
                raise RuntimeError(f"{self.name} did not start:\n" + log)
            time.sleep(0.25)
        os.makedirs(self.root(), exist_ok=True)
        for path in self.sources:
            target = os.path.join(self.root(), path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh("docker", "cp", f"{self.container}:{self.site}/{path}", target)

    def logs(self, lines=60):
        return sh("docker", "logs", "--tail", str(lines), self.container).stderr

    def stop(self):
        sh("docker", "rm", "-f", self.container)
        self.remove_agent_dir()

    def alive(self):
        out = sh("docker", "inspect", "-f", "{{.State.Running}}", self.container).stdout
        return out.strip() == "true"

    def exercise(self, timeout, prompts=32, tokens=64):
        began = time.perf_counter()
        reply = self.request(timeout, prompts=prompts, tokens=tokens)
        reply["seconds"] = time.perf_counter() - began
        return reply

    def reset(self):
        return self.request(60.0, reset=True)["digest"]

    def stats(self):
        rows = sh("docker", "top", self.container, "-o", "pid,rss").stdout.splitlines()[1:]
        return {"rss_kib": sum(int(row.split()[1]) for row in rows),
                "gpu_mib": self.gpu_mib()}

    def fresh(self, edit):
        other = type(self)(os.path.join(self.workdir, "fresh"), gpu=self.fresh_gpu,
                           model=self.model, image=self.image)
        other.start(overlay=edit)
        return other
