"""Restart per check vs live edit on a real `vllm serve`: the run behind README.md.

Run it with the Python of a vLLM environment, on a machine with one free GPU:

    CUDA_VISIBLE_DEVICES=0 python walkthrough.py --model <path or HF id> --work <scratch dir>

It never edits the installed vLLM: it copies the package to <work>/src and serves from the copy.
Every step prints one JSON line; --out keeps the full replies (receipts included).

Phases (--phases, default all three):
  warm   start a plain server once so on-disk caches are warm; record the base output
  fresh  per edit: write the edit into the package copy, start a new plain server, check, stop
  live   one server with the kernel adapter; per edit: txn, 3 checks, prove, undo, base check
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIVELAB = os.path.abspath(os.path.join(HERE, "..", "..", "livelab"))

# One line each. `on`: the process role that runs the module (the kernel's plan field).
EDITS = {
    "scheduler": {
        "module": "vllm.v1.core.sched.scheduler", "on": ["core"],
        "old": "        token_budget = self.max_num_scheduled_tokens\n",
        "new": "        token_budget = min(self.max_num_scheduled_tokens, 4096)\n"},
    # the sampler the default (V2) model runner calls: mask the top token, so greedy
    # decoding picks the runner-up. A deliberately visible test edit.
    "sampler": {
        "module": "vllm.v1.worker.gpu.sample.sampler", "on": ["worker"],
        "old": "            processed_logits = apply_top_k_top_p("
               "processed_logits, top_k, top_p)\n",
        "old_alt": ("            processed_logits = apply_top_k_top_p(\n"
                    "                processed_logits, top_k, top_p, "
                    "has_topp_only_rows=has_topp_only_rows\n"
                    "            )\n"),
        "suffix": ("            processed_logits = processed_logits.scatter("
                   "-1, processed_logits.argmax(-1, keepdim=True), float(\"-inf\"))\n")},
    # the same idea in the older sampler module, which the V2 runner never calls
    "old_sampler": {
        "module": "vllm.v1.sample.sampler", "on": ["worker"],
        "old": "        return logits.argmax(dim=-1).view(-1)\n",
        "new": "        return logits.topk(2, dim=-1).indices[:, 1].view(-1)\n"},
}
PROMPTS = [
    "The capital of France is", "Water boils at a temperature of", "def fibonacci(n):",
    "The three primary colors are", "In 1969, humans first", "The derivative of x squared is",
    "A haiku about autumn:", "The largest planet in the solar system is",
]


def say(step, **kw):
    print(json.dumps({"step": step, **kw}), flush=True)


class Run:
    def __init__(self, a):
        self.a = a
        self.work = os.path.abspath(a.work)
        self.src = os.path.join(self.work, "src")
        self.ctl = os.path.join(self.work, "ctl")
        self.proc = None
        self.full = {}

    # -- files --------------------------------------------------------------------------
    def setup(self):
        pkg = os.path.dirname(importlib.util.find_spec("vllm").origin)
        dst = os.path.join(self.src, "vllm")
        if not os.path.isdir(dst):
            shutil.copytree(pkg, dst, symlinks=True)
        site = os.path.join(self.work, "site")
        os.makedirs(site, exist_ok=True)
        with open(os.path.join(site, "sitecustomize.py"), "w") as f:
            f.write("import os\nif os.environ.get('FASTRSI_KERNEL_DIR'):\n"
                    "    import adapter.host\n")
        self.pythonpath = os.pathsep.join([self.src, LIVELAB, site])
        for name, e in EDITS.items():
            e["path"] = os.path.join(self.src, *e["module"].split(".")) + ".py"
            with open(e["path"]) as f:
                e["base"] = f.read()
            old = (e["old"] if e["base"].count(e["old"]) == 1
                   else e.get("old_alt", e["old"]))
            assert e["base"].count(old) == 1, f"{name}: the base line is not unique"
            new = e["new"] if "new" in e else old + e["suffix"]
            e["patched"] = e["base"].replace(old, new)
            rel = os.path.relpath(e["path"], self.src)
            e["diff"] = "".join(difflib.unified_diff(
                e["base"].splitlines(True), e["patched"].splitlines(True),
                "a/" + rel, "b/" + rel, n=0))

    def write(self, name, which):
        e = EDITS[name]
        with open(e["path"], "w") as f:
            f.write(e[which])

    # -- server -------------------------------------------------------------------------
    def serve(self, adapter, tag):
        """Start `vllm serve`; return seconds from launch to the first 200 on /health."""
        env = dict(os.environ, PYTHONPATH=self.pythonpath)
        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.a.model,
               "--served-model-name", "model", "--port", str(self.a.port),
               "--distributed-executor-backend", "mp"] + self.a.serve_args.split()
        if adapter:
            shutil.rmtree(self.ctl, ignore_errors=True)
            os.makedirs(self.ctl)
            env["FASTRSI_KERNEL_DIR"] = self.ctl
            cmd += ["--worker-extension-cls", "adapter.worker_ext.KernelWorkerExtension"]
        log = open(os.path.join(self.work, f"serve-{tag}.log"), "w")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                     start_new_session=True)
        url = f"http://127.0.0.1:{self.a.port}/health"
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"vllm serve exited {self.proc.returncode}: see {log.name}")
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    return round(time.monotonic() - t0, 2)
            except OSError:
                pass
            if time.monotonic() - t0 > 1800:
                raise SystemExit("no /health in 1800 s")
            time.sleep(0.1)

    def stop(self):
        if self.proc is None:
            return 0.0
        t0 = time.monotonic()
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        while True:                      # the GPU is free again before the next start
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"] + (
                ["-i", os.environ["CUDA_VISIBLE_DEVICES"]]
                if os.environ.get("CUDA_VISIBLE_DEVICES") else []),
                capture_output=True, text=True).stdout.split()
            if all(int(x) < 500 for x in used) or time.monotonic() - t0 > 120:
                break
            time.sleep(0.2)
        self.proc = None
        return round(time.monotonic() - t0, 2)

    def check(self):
        """The workload: 8 prompts, one request at a time, greedy, 32 tokens each. One at a
        time, because the output of a batch depends on which requests share a step."""
        t0, texts = time.monotonic(), []
        for prompt in PROMPTS:
            body = json.dumps({"model": "model", "prompt": prompt, "max_tokens": 32,
                               "temperature": 0, "seed": 0}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.a.port}/v1/completions", body,
                {"Content-Type": "application/json"})
            texts.append(json.load(urllib.request.urlopen(req, timeout=600))
                         ["choices"][0]["text"])
        return {"seconds": round(time.monotonic() - t0, 3), "texts": texts,
                "output": hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:16]}

    def checks(self, n):
        rs = [self.check() for _ in range(n)]
        return {"seconds": [r["seconds"] for r in rs], "outputs": [r["output"] for r in rs],
                "first_text": rs[0]["texts"][0]}, rs[0]

    def op(self, **cmd):
        from adapter.client import send
        return send(self.ctl, cmd)

    # -- phases -------------------------------------------------------------------------
    def warm(self):
        ready = self.serve(False, "warm")
        c, first = self.checks(3)
        self.base = first["output"]
        self.full["base_texts"] = first["texts"]
        say("warm", healthy_after_s=ready, check=c, stop_s=self.stop())

    def restart_check(self, name):
        """What a developer does without the lab: save the edit, start the server, check."""
        if name:
            self.write(name, "patched")
        t0 = time.monotonic()
        ready = self.serve(False, f"fresh-{name or 'base'}")
        c = self.check()
        total = round(time.monotonic() - t0, 2)
        again = self.check()["output"]
        stop = self.stop()
        if name:
            self.write(name, "base")
        self.full.setdefault("fresh", {})[name or "base"] = c
        say("restart_check", edit=name, healthy_after_s=ready, check_s=c["seconds"],
            edit_to_result_s=total, output=c["output"], second_check_output=again,
            first_text=c["texts"][0], stop_s=stop)
        return c["output"], total

    def live(self, restart_s):
        from kernel import locate as L
        ready = self.serve(True, "live")
        say("live_start", healthy_after_s=ready, ping=self.op(op="ping").get("parallel"))
        c, _ = self.checks(3)
        say("live_base_check", check=c)
        mods = [e["module"] for e in EDITS.values()]
        fp0 = self.fingerprint(mods)
        live = {}
        for name, e in EDITS.items():
            t_edit, m0 = time.time(), time.monotonic()
            r = self.op(op="txn", edit_time=t_edit, baseline_restart={
                "seconds": restart_s.get(name), "source": "restart_check of this run"},
                plan={"modules": {e["module"]: {
                    "source": e["patched"], "base_hash": L.text_hash(e["base"]),
                    "on": e["on"]}}})
            rec = r.get("receipt") or {}
            say("install", edit=name, ok=r.get("ok"), outcome=r.get("outcome"), txn=r.get("txn"),
                mechanisms=rec.get("actual_recipes"), targets=(rec.get("plan") or {}).get(
                    "targets"), prepare_s=rec.get("prepare_s"), gap_s=rec.get("gap_s"),
                edit_to_commit_s=rec.get("edit_to_commit_s"),
                client_round_trip_s=r.get("client_round_trip_s"),
                findings=[f.get("check") for f in rec.get("findings") or []],
                error=r.get("error"))
            first = self.check()
            first_done = round(time.monotonic() - m0, 3)
            more, _ = self.checks(2)
            p = self.op(op="prove", txn=r.get("txn"), golden={"output": first["output"]})
            prec = p.get("receipt") or {}
            say("check", edit=name, edit_to_result_s=first_done,
                seconds=[first["seconds"]] + more["seconds"],
                outputs=[first["output"]] + more["outputs"], first_text=first["texts"][0],
                proven=p.get("ok"), witness=prec.get("witness"),
                edit_to_run_s=prec.get("edit_to_run_s"),
                findings=[f.get("check") for f in prec.get("findings") or []])
            u = self.op(op="undo", txn=r.get("txn"))
            urec = u.get("receipt") or {}
            after, _ = self.checks(2)
            ab = self.op(op="at_base")
            say("uninstall", edit=name, ok=u.get("ok"), status=urec.get("status"),
                undo_s=urec.get("undo_s"), client_round_trip_s=u.get("client_round_trip_s"),
                check=after, output_is_base=set(after["outputs"]) == {self.base},
                at_base=ab.get("ok"), fingerprint_is_base=self.fingerprint(mods) == fp0)
            live[name] = first["output"]
            self.full.setdefault("live", {})[name] = {
                "txn": r, "prove": p, "undo": u, "at_base": ab, "check": first}
        say("live_stop", stop_s=self.stop())
        return live

    def fingerprint(self, mods):
        r = self.op(op="fingerprint", modules=mods)
        return {"workers": [x.get("code") for x in (r.get("workers") or {}).get("ranks", [])],
                "core": (r.get("core") or {}).get("code")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out")
    ap.add_argument("--serve-args", default="--no-enable-prefix-caching",
                    help="extra `vllm serve` flags for every start (see README: with the "
                         "prefix cache on, a repeated check does not repeat its output)")
    ap.add_argument("--phases", default="warm,fresh,live")
    a = ap.parse_args()
    sys.path.insert(0, LIVELAB)
    import adapter  # noqa: F401  (puts core/livelab on sys.path)
    run = Run(a)
    run.setup()
    for name, e in EDITS.items():
        print(e["diff"], flush=True)
    phases = a.phases.split(",")
    fresh, restart_s, live = {}, {}, {}
    try:
        run.warm()
        if "fresh" in phases:
            fresh["base"], restart_s["base"] = run.restart_check(None)
            for name in EDITS:
                fresh[name], restart_s[name] = run.restart_check(name)
        if "live" in phases:
            live = run.live(restart_s)
        say("agreement", base=run.base, fresh=fresh, live=live,
            fresh_base_is_base=fresh.get("base") == run.base if fresh else None,
            live_equals_fresh={n: live[n] == fresh[n] for n in live if n in fresh})
    finally:
        if run.proc is not None:
            run.stop()
        for name in EDITS:
            run.write(name, "base")
        if a.out:
            with open(a.out, "w") as f:
                json.dump(run.full, f, indent=1, default=str)


if __name__ == "__main__":
    main()
