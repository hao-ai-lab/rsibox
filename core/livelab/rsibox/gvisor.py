"""Run one command under gVisor (`runsc run`) from a generated OCI bundle. Needs root.

    python3 -m rsibox.gvisor --runsc /path/runsc [--allow-unsupported-driver] [--bind DIR ...] -- CMD ...

The sandbox sees this machine's (or this container's) root filesystem read-only behind an
in-memory overlay, so the engine's environment needs no image. `--bind DIR` mounts a folder
read-write through to the host (the box's state folder). The network is the host's, so the
daemon's port is reachable from outside the sandbox.

GPUs: the NVIDIA control devices plus one `/dev/nvidia<N>` per index in `CUDA_VISIBLE_DEVICES`
(all of them when it is unset or not a list of indices) are passed in, and gVisor's `nvproxy`
forwards their ioctls. AMD accelerators have no proxy in gVisor.
"""
import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile

CONTROL_DEVICES = ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools")
CAPS = ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_KILL", "CAP_NET_BIND_SERVICE",
        "CAP_SETGID", "CAP_SETUID", "CAP_SYS_PTRACE"]


def gpu_device_names(visible, present):
    """Device file names to pass in, from `CUDA_VISIBLE_DEVICES` and the names under /dev."""
    cards = sorted(n for n in present if n.startswith("nvidia") and n[6:].isdigit())
    wanted = [p.strip() for p in (visible or "").split(",") if p.strip()]
    if wanted and all(p.isdigit() for p in wanted):
        cards = [n for n in cards if n[6:] in wanted]
    return [n for n in CONTROL_DEVICES if n in present] + (cards if cards else [])


def sandbox_env(env, names, present):
    """The command's environment. Inside the sandbox the driver lists only the cards passed in,
    numbered from 0, so a `CUDA_VISIBLE_DEVICES` that picked a subset is renumbered to match."""
    out = dict(env)
    cards = [n for n in names if n not in CONTROL_DEVICES]
    if cards and len(cards) < sum(1 for n in present if n.startswith("nvidia") and n[6:].isdigit()):
        out["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(cards)))
    return out


def device_nodes(names, dev_dir="/dev"):
    nodes = []
    for name in names:
        st = os.stat(os.path.join(dev_dir, name))
        if stat.S_ISCHR(st.st_mode):
            nodes.append({"path": f"/dev/{name}", "type": "c", "major": os.major(st.st_rdev),
                          "minor": os.minor(st.st_rdev), "fileMode": 0o666, "uid": 0, "gid": 0})
    return nodes


def build_spec(argv, env, cwd, binds=(), devices=()):
    """The OCI runtime spec: rootfs `/` read-only, host network, the given binds and devices."""
    mounts = [{"destination": "/proc", "type": "proc", "source": "proc"},
              {"destination": "/dev", "type": "tmpfs", "source": "tmpfs"},
              {"destination": "/sys", "type": "sysfs", "source": "sysfs", "options": ["ro"]},
              {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs"},
              {"destination": "/dev/shm", "type": "tmpfs", "source": "shm"}]
    mounts += [{"destination": d, "type": "bind", "source": d, "options": ["rbind", "rw"]} for d in binds]
    return {"ociVersion": "1.0.0",
            "process": {"terminal": False, "user": {"uid": 0, "gid": 0}, "args": list(argv), "cwd": cwd,
                        "env": [f"{k}={v}" for k, v in sorted(env.items())],
                        "capabilities": {k: CAPS for k in ("bounding", "effective", "permitted")}},
            "root": {"path": "/", "readonly": True},
            "mounts": mounts,
            "linux": {"devices": list(devices),
                      "namespaces": [{"type": t} for t in ("pid", "ipc", "uts", "mount")]}}


def runsc_argv(runsc, bundle, name, allow_unsupported_driver=False, gpu=True, extra=()):
    flags = ["--network=host", "--ignore-cgroups", "--overlay2=root:memory"]
    if gpu:
        flags.append("--nvproxy")
        if allow_unsupported_driver:
            flags.append("--nvproxy-allow-unsupported-driver")
    return [runsc, *flags, *extra, "run", "--bundle", bundle, name]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m rsibox.gvisor")
    ap.add_argument("--runsc", default="runsc")
    ap.add_argument("--allow-unsupported-driver", action="store_true")
    ap.add_argument("--bind", action="append", default=[], help="folder mounted read-write through to the host")
    ap.add_argument("--runsc-flag", action="append", default=[])
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command after --")
    present = os.listdir("/dev")
    names = gpu_device_names(os.environ.get("CUDA_VISIBLE_DEVICES"), present)
    spec = build_spec(cmd, sandbox_env(os.environ, names, present), os.getcwd(),
                      [os.path.abspath(b) for b in a.bind], device_nodes(names))
    bundle = tempfile.mkdtemp(prefix="rsibox-bundle-")
    with open(os.path.join(bundle, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
    return subprocess.call(runsc_argv(a.runsc, bundle, f"rsibox{os.getpid()}", a.allow_unsupported_driver,
                                      gpu=bool(names), extra=a.runsc_flag))


if __name__ == "__main__":
    sys.exit(main())
