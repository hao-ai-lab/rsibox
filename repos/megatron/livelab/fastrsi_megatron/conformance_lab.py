"""MegatronLab: a real Megatron-LM training job behind the conformance harness (LabUnderTest).

The engine is a launcher command (FASTRSI_MG_CMD, a shell command that runs torchrun with
`-m fastrsi_megatron.run pretrain_gpt.py <args>` on the lab's GPUs; it must honor $PORT and
pass the environment through). The lab starts it with FASTRSI_LAB=1 in a new session (its own
process group, so respawn kills only what the lab started), waits for the ready file, and drives
it over the socket (client.py).

Workload: `check` runs `steps` real train steps from the lab's start state S on frozen batches;
its output is the per-step loss and grad norm. Patches edit `megatron.core.transformer.mlp`
and `megatron.core.transformer.attention` (MLP.forward, Attention.forward_post_core_attn,
Attention.forward, MLP.__init__), read from the running Megatron tree (FASTRSI_MG_ROOT).

Faults are real: crash and lost make a rank's process exit (torchrun then stops the job);
hang makes a rank sleep in a step, which the client's timeout turns into poison; respawn
kills the lab's process group and starts the job again.

One job is shared across cases while it stays at base (a start costs about a minute); a
poisoned or edited job is respawned instead.
"""
import atexit
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from fastrsi import harness, ipc, proc

from . import client

MLP = "megatron.core.transformer.mlp"
ATT = "megatron.core.transformer.attention"
_edit = harness.edit_once
_descendants = proc.descendants
_free_port = proc.free_port


class _Job:
    """One launched torchrun job in its own process group. Its folder (ready file, job.log) and
    socket are removed by kill()."""

    def __init__(self, cmd, env, tag, timeout=900):
        self.dir = tempfile.mkdtemp(prefix=f"fastrsi-mg-{tag}-")
        self.ready = os.path.join(self.dir, "ready.json")
        self.sock = ipc.short_sock_path(f"fmg-{tag}")
        env = {**os.environ, **env, "FASTRSI_READY": self.ready, "FASTRSI_SOCK": self.sock}
        self.log = open(os.path.join(self.dir, "job.log"), "w")
        self.t0 = time.time()
        self.p = subprocess.Popen(["bash", "-c", cmd], env=env, stdout=self.log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        self.timeout = timeout

    def wait_ready(self):
        while time.time() - self.t0 < self.timeout:
            if os.path.exists(self.ready) and os.path.getsize(self.ready):
                with open(self.ready) as f:
                    self.info = json.load(f)
                self.start_s = round(time.time() - self.t0, 2)
                return True
            if self.p.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def wait_exit(self):
        try:
            return self.p.wait(self.timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            return None

    def kill(self):
        """Kill the job and every process it started, by PID. torchrun starts its workers in
        sessions of their own, so a kill of the launcher's process group leaves them running
        (and holding GPU memory); walk the process tree instead, children first."""
        pids = _descendants(self.p.pid)
        proc.kill_tree(self.p.pid, signal.SIGKILL, include_pg=False)
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 60
        while any(proc.pid_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.2)
        if not self.log.closed:
            self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.unlink(self.sock)

    def tail(self, n=30):
        try:
            with open(os.path.join(self.dir, "job.log")) as f:
                return f.read().splitlines()[-n:]
        except OSError:
            return []


class MegatronLab:
    name = "megatron"
    _shared = None

    def __init__(self, receipts=None, steps=None, check_timeout=None):
        self.cmd = os.environ["FASTRSI_MG_CMD"]
        self.root = os.environ["FASTRSI_MG_ROOT"]
        self.steps = steps or int(os.environ.get("FASTRSI_CHECK_STEPS", "3"))
        self.check_timeout = check_timeout or float(os.environ.get("FASTRSI_CHECK_TIMEOUT",
                                                                   "120"))
        self.receipts = receipts
        self.poison = []
        self.base = {m: open(os.path.join(self.root, *m.split(".")) + ".py").read()
                     for m in (MLP, ATT)}
        sh = MegatronLab._shared
        if sh is not None and sh.p.poll() is None and self._at_base(sh):
            self.job = sh
        else:
            if sh is not None:
                sh.kill()
            self._spawn()

    # -- process layout ------------------------------------------------------------------
    def _env(self, port):
        env = {"FASTRSI_LAB": "1", "PORT": str(port), "FASTRSI_CHECK_STEPS": str(self.steps)}
        if self.receipts:
            env["FASTRSI_RECEIPTS"] = os.path.abspath(self.receipts)
        return env

    def _spawn(self):
        job = _Job(self.cmd, self._env(_free_port()), "lab")
        if not job.wait_ready():
            tail = job.tail()
            job.kill()
            raise RuntimeError("lab did not start: " + "\n".join(tail))
        self.job = MegatronLab._shared = job
        self.poison = []

    @staticmethod
    def _at_base(job):
        try:
            st = client.call(job.sock, {"verb": "status"}, 60)
            return st.get("state") == "base"
        except Exception:
            return False

    def _call(self, req, timeout=None):
        try:
            return client.call(self.job.sock, req, timeout or self.check_timeout)
        except (TimeoutError, ConnectionError, FileNotFoundError, OSError) as e:
            self.poison.append({"check": "participant_failed", "observed": f"{type(e).__name__}"
                                f": {e}", "expected": "the lab answers",
                                "why": "a rank crashed, hung or was lost"})
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "lost": True}

    def close(self):
        pass

    # -- verbs ---------------------------------------------------------------------------
    def install(self, patch):
        if self.poison:
            return {"ok": False, "outcome": "refused", "codes": ["session_poisoned"],
                    "id": None, "mechanisms": [], "findings": list(self.poison), "cost_s": 0}
        r = self._call({"verb": "install", "patch": patch})
        if r.get("lost"):
            return {"ok": False, "outcome": "refused", "codes": ["participant_failed"],
                    "id": None, "mechanisms": [], "findings": list(self.poison), "cost_s": 0}
        return r

    def check(self, workload=None):
        steps = (workload or {}).get("steps", self.steps)
        r = self._call({"verb": "check", "steps": steps})
        if r.get("lost"):
            return {"ok": False, "error": r["error"], "output": None, "proven": None,
                    "witness": {}, "findings": list(self.poison)}
        return r

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        if kind == "respawn":
            self.job.kill()
            MegatronLab._shared = None
            self._spawn()
            return {"ok": True, "reset": ["worker group", "session"],
                    "start_s": self.job.start_s, "cost_s": round(time.perf_counter() - t0, 3)}
        r = self._call({"verb": "reset", "kind": kind})
        return {**r, "cost_s": round(time.perf_counter() - t0, 3)}

    def uninstall(self, id=None):
        return self._call({"verb": "uninstall", **({"id": id} if id else {})})

    def status(self):
        if self.poison:
            return {"ok": True, "state": "poisoned", "txns": [], "fingerprint": None,
                    "poison": list(self.poison)}
        r = self._call({"verb": "status"}, 120)
        if r.get("lost"):
            return {"ok": True, "state": "poisoned", "txns": [], "fingerprint": None,
                    "poison": list(self.poison)}
        return r

    # -- harness -------------------------------------------------------------------------
    def patches(self):
        mlp, att = self.base[MLP], self.base[ATT]
        tail = '        nvtx_range_pop(suffix="linear_fc2")\n'
        m2 = _edit(mlp, tail, tail + "        output = output * 0.5\n")
        m3 = _edit(m2, "        output = output * 0.5\n", "        output = output * 0.25\n")
        a_ret = "        nvtx_range_pop(suffix=\"linear_proj\")\n\n        return output, bias\n"
        a2 = _edit(att, a_ret, a_ret.replace("return output, bias", "return output * 0.5, bias"))
        a_fwd = ("        return self.forward_post_core_attn(self.forward_pre_attn_and_core_attn("
                 "*args, **kwargs))\n")
        a_multi = _edit(a2, a_fwd, "        pre = self.forward_pre_attn_and_core_attn(*args, "
                        "**kwargs)\n        return self.forward_post_core_attn(pre)\n")
        init_end = "\n    def forward(\n        self, hidden_states: torch.Tensor, per_token_scale"
        unreached = _edit(mlp, init_end, "\n        with torch.no_grad():\n"
                          "            self.linear_fc2.weight.mul_(0.5)\n" + init_end)

        def p(**files):
            return {"files": dict(files)}
        return {
            "noop": p(**{MLP: _edit(mlp, tail, tail + "        output = output * 1\n")}),
            "visible": p(**{MLP: m2}),
            "chain": [p(**{MLP: m2}), p(**{MLP: m3}), p(**{ATT: a2})],
            "multi": p(**{MLP: m2, ATT: a_multi}),
            "unreached": {**p(**{MLP: unreached}), "accept": ["constructor_edit"]},
            "refused": {
                "syntax": p(**{MLP: _edit(mlp, "        return output, output_bias\n",
                                          "        return output,, output_bias\n")}),
                "closure": p(**{MLP: _edit(
                    mlp, "                        return self.config.activation_func(x_glu) * (",
                    "                        return bias_parallel * "
                    "self.config.activation_func(x_glu) * (")}),
                "module_state": p(**{MLP: _edit(mlp, "    HAVE_TE = True\n",
                                                "    HAVE_TE = 1\n")}),
                "empty": p(**{MLP: _edit(mlp, "        # [s, b, 4 * h/p]\n",
                                         "        # [s, b, 4 * h/p] (fc1)\n")}),
            },
        }

    def fresh_check(self, patch, workload=None):
        if (patch or {}).get("repairs"):
            return {"ok": False, "output": None, "error": "fresh overlays apply source files only"}
        d = tempfile.mkdtemp(prefix="fastrsi-mg-overlay-")
        out = os.path.join(d, "out.json")
        try:
            for m, text in ((patch or {}).get("files") or {}).items():
                with open(os.path.join(d, m + ".py"), "w") as f:
                    f.write(text)
            env = {**self._env(_free_port()), "FASTRSI_FRESH": out,
                   "FASTRSI_OVERLAY": d}
            env.pop("FASTRSI_LAB")
            job = _Job(self.cmd, env, "fresh")
            rc = job.wait_exit()
            tail = job.tail(8)
            job.kill()
            if not os.path.exists(out):
                return {"ok": False, "error": f"fresh run rc={rc}: " + " | ".join(tail),
                        "output": None}
            with open(out) as f:
                r = json.load(f)
            return {"ok": bool(r.get("ok")), "output": r.get("output"), "error": r.get("error")}
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def inject(self, fault, **kw):
        target = kw.get("participant")
        if fault == "tamper":
            req = {"verb": "inject", "fault": "tamper", "target": target, "module": ATT,
                   "qualname": "Attention.forward",
                   "source": "def forward(self, *args, **kwargs):\n    return self."
                             "forward_post_core_attn(self.forward_pre_attn_and_core_attn("
                             "*args, **kwargs))\n"}
        else:
            req = {"verb": "inject", "fault": fault, "target": target, **kw}
        r = self._call(req, 60)
        return {"ok": bool(r.get("ok")), "fault": fault, **({"error": r.get("error")}
                                                          if not r.get("ok") else {})}


@atexit.register
def _stop_shared():
    if MegatronLab._shared is not None:
        MegatronLab._shared.kill()

if __name__ == "__main__":
    import argparse
    from conformance.__main__ import main

    parser = argparse.ArgumentParser(description="Run conformance on a local Megatron checkout")
    parser.add_argument("--root", required=True, help="Megatron-LM checkout")
    parser.add_argument("--nproc", required=True, type=int, help="number of local GPU ranks")
    parser.add_argument("--only", default="", help="comma-separated conformance case names")
    parser.add_argument("--receipts", help="append transaction receipts here")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="-- followed by pretrain_gpt arguments")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    training = args.training_args
    if training[:1] == ["--"]:
        training = training[1:]
    launch = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(args.nproc)]
    worker = ["-m", "fastrsi_megatron.run", os.path.join(root, "pretrain_gpt.py"), *training]
    os.environ["FASTRSI_MG_CMD"] = f'exec {shlex.join(launch)} --master-port "$PORT" {shlex.join(worker)}'
    os.environ["FASTRSI_MG_ROOT"] = root
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH"))))
    options = ["--only", args.only]
    if args.receipts:
        options += ["--receipts", args.receipts]
    if args.json:
        options.append("--json")
    sys.exit(main(["--lab", "fastrsi_megatron.conformance_lab:MegatronLab", *options]))
