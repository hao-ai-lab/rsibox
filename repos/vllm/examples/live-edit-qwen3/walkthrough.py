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
METHODS = {
    "scheduler": "Scheduler.schedule",
    "sampler": "Sampler._sample_random",
    "old_sampler": "Sampler.greedy_sample",
}
PHASES = ("warm", "fresh", "live")
PROMPTS = [
    "The capital of France is", "Water boils at a temperature of", "def fibonacci(n):",
    "The three primary colors are", "In 1969, humans first", "The derivative of x squared is",
    "A haiku about autumn:", "The largest planet in the solar system is",
]


def say(step, **kw):
    print(json.dumps({"step": step, **kw}), flush=True)


class VerificationError(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise VerificationError(reason)


def checked_texts(checks, label):
    """Verify actual text from every prompt and repeat."""
    texts = [check["texts"] for check in checks]
    require(texts and all(len(batch) == len(PROMPTS) and
                          all(isinstance(text, str) and text.strip() for text in batch)
                          for batch in texts),
            f"{label}: expected nonempty completion text for every prompt")
    require(all(batch == texts[0] for batch in texts),
            f"{label}: repeated checks disagree")
    return texts[0]


def verify_outputs(warm, fresh, live_base, live):
    base = checked_texts(warm, "warm")
    require(checked_texts(fresh["base"], "fresh base") == base,
            "fresh base disagrees with warm base")
    require(checked_texts(live_base, "live base") == base,
            "live base disagrees with warm base")
    for name in EDITS:
        fresh_texts = checked_texts(fresh[name], f"{name} fresh")
        live_texts = checked_texts(live[name]["checks"], f"{name} live")
        require(live_texts == fresh_texts, f"{name}: live output disagrees with fresh")
        if name == "sampler":
            require(live_texts != base, "sampler: output did not visibly change")
        else:
            require(live_texts == base, f"{name}: output changed from base")


def verify_install(name, reply):
    receipt = reply.get("receipt") or {}
    require(reply.get("ok") is True and reply.get("outcome") == "committed"
            and reply.get("txn"),
            f"{name}: transaction did not commit ({reply.get('error') or reply.get('outcome')})")
    require(receipt.get("actual_recipes") == ["R0 swap"],
            f"{name}: expected R0 swap, got {receipt.get('actual_recipes')}")


def verify_proof(name, reply, fingerprint):
    receipt = reply.get("receipt") or {}
    witness = receipt.get("witness") or {}
    edit = EDITS[name]
    prefix = f"{edit['module']}:{METHODS[name]}:"
    roles = (["core"] if edit["on"] == ["core"] else
             [f"workers/rank{rank}" for rank in fingerprint["workers"]])
    findings = [f.get("check") for f in receipt.get("findings") or []]
    unreached = name == "old_sampler"
    if unreached:
        require(reply.get("ok") is False and set(findings) == {"zero_witness"},
                f"{name}: expected an unproven zero_witness result, got {findings}")
    else:
        require(reply.get("ok") is True and not findings,
                f"{name}: proof failed ({findings})")
    for role in roles:
        counts = [count for key, count in witness.get(role, {}).items()
                  if key.startswith(prefix)]
        require(counts and all(type(count) is int and
                               (count == 0 if unreached else count > 0) for count in counts),
                f"{name}: expected {'zero' if unreached else 'positive'} calls to "
                f"{METHODS[name]} on {role}, got {counts}")
    require(not any(key.startswith(prefix) for role, counts in witness.items()
                    if role not in roles for key in counts),
            f"{name}: witness for {METHODS[name]} appeared on an unexpected role")


def verify_restoration(name, observation, warm, fingerprint):
    undo = observation["undo"]
    require(undo.get("ok") is True and (undo.get("receipt") or {}).get("status") == "undone",
            f"{name}: undo failed ({undo.get('error') or undo.get('status')})")
    require(observation["at_base"].get("ok") is True, f"{name}: adapter did not return to base")
    require(observation["fingerprint"] == fingerprint,
            f"{name}: code fingerprint was not restored")
    require(checked_texts(observation["after"], f"{name} after undo") ==
            checked_texts(warm, "warm"), f"{name}: undo did not restore base output")


class Run:
    def __init__(self, a):
        self.a = a
        self.work = os.path.abspath(a.work)
        self.src = os.path.join(self.work, "src")
        self.ctl = os.path.join(self.work, "ctl")
        self.proc = None
        self.full = {}
        self.edits = {name: dict(edit) for name, edit in EDITS.items()}
        self.warm_checks = []
        self.fresh_checks = {}
        self.live_base_checks = []
        self.phases = set(a.phases.split(","))
        self.strict = self.phases == set(PHASES)

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
        if not self.phases.intersection(("fresh", "live")):
            self.edits.clear()
        for name, e in self.edits.items():
            e["path"] = os.path.join(self.src, *e["module"].split(".")) + ".py"
            with open(e["path"]) as f:
                e["base"] = f.read()
            old = (e["old"] if e["base"].count(e["old"]) == 1
                   else e.get("old_alt", e["old"]))
            require(e["base"].count(old) == 1, f"{name}: expected one matching base source fragment")
            new = e["new"] if "new" in e else old + e["suffix"]
            e["patched"] = e["base"].replace(old, new)
            rel = os.path.relpath(e["path"], self.src)
            e["diff"] = "".join(difflib.unified_diff(
                e["base"].splitlines(True), e["patched"].splitlines(True),
                "a/" + rel, "b/" + rel, n=0))

    def write(self, name, which):
        e = self.edits[name]
        with open(e["path"], "w") as f:
            f.write(e[which])

    # -- server -------------------------------------------------------------------------
    def serve(self, adapter, tag):
        """Start `vllm serve`; return seconds from launch to the first 200 on /health."""
        self.server_tag = tag
        env = dict(os.environ, PYTHONPATH=self.pythonpath)
        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.a.model,
               "--served-model-name", "model", "--port", str(self.a.port),
               "--distributed-executor-backend", "mp"] + self.a.serve_args.split()
        if adapter:
            shutil.rmtree(self.ctl, ignore_errors=True)
            os.makedirs(self.ctl)
            env["FASTRSI_KERNEL_DIR"] = self.ctl
            env["PYTHONPATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))
            cmd += ["--worker-extension-cls", "walkthrough_worker.KernelWorkerExtension"]
        else:
            env.pop("FASTRSI_KERNEL_DIR", None)
        t0 = time.monotonic()
        with open(os.path.join(self.work, f"serve-{tag}.log"), "w") as log:
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
        t0, texts, ttft = time.monotonic(), [], []
        for prompt in PROMPTS:
            body = json.dumps({"model": "model", "prompt": prompt, "max_tokens": 32,
                               "temperature": 0, "seed": 0, "stream": True, "return_token_ids": True}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.a.port}/v1/completions", body,
                {"Content-Type": "application/json"})
            started = time.monotonic()
            pieces, first = [], None
            with urllib.request.urlopen(req, timeout=600) as response:
                for line in response:
                    if line.strip() == b"data: [DONE]":
                        break
                    if line.startswith(b"data: "):
                        for choice in json.loads(line[6:])["choices"]:
                            if first is None and choice.get("token_ids"):
                                first = time.monotonic() - started
                            pieces.append(choice["text"])
                else:
                    raise VerificationError("completion stream ended before [DONE]")
            require(first is not None, "completion returned no token IDs")
            texts.append("".join(pieces))
            ttft.append(first)
            self.full.setdefault("requests", []).append({
                "server": self.server_tag, "prompt": prompt, "ttft_s": first,
                "wall_s": time.monotonic() - started})
        return {"seconds": round(time.monotonic() - t0, 3), "texts": texts, "ttft_s": ttft,
                "output": hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:16]}

    def checks(self, n):
        rs = [self.check() for _ in range(n)]
        return {"seconds": [r["seconds"] for r in rs], "outputs": [r["output"] for r in rs],
                "first_text": next(iter(rs[0]["texts"]), None)}, rs

    def op(self, **cmd):
        from adapter.client import send
        return send(self.ctl, cmd)

    # -- phases -------------------------------------------------------------------------
    def warm(self):
        ready = self.serve(False, "warm")
        c, self.warm_checks = self.checks(3)
        first = self.warm_checks[0]
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
        again = self.check()
        stop = self.stop()
        if name:
            self.write(name, "base")
        self.full.setdefault("fresh", {})[name or "base"] = c
        self.fresh_checks[name or "base"] = [c, again]
        say("restart_check", edit=name, healthy_after_s=ready, check_s=c["seconds"],
            edit_to_result_s=total, output=c["output"], second_check_output=again["output"],
            first_text=next(iter(c["texts"]), None), stop_s=stop)
        return c["output"], total

    def live(self, restart_s):
        from kernel import locate as L
        ready = self.serve(True, "live")
        parallel = self.op(op="ping").get("parallel") or {}
        say("live_start", healthy_after_s=ready, ping=parallel)
        c, self.live_base_checks = self.checks(3)
        say("live_base_check", check=c)
        world = parallel.get("world")
        fp0 = self.fingerprint(world)
        if self.strict:
            require(fp0 is not None, "live base: could not read code fingerprint")
        live = {}
        for name, e in self.edits.items():
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
            observed = {"txn": r}
            self.full.setdefault("live", {})[name] = observed
            if self.strict:
                verify_install(name, r)
            first = self.check()
            first_done = round(time.monotonic() - m0, 3)
            more, repeats = self.checks(2)
            p = self.op(op="prove", txn=r.get("txn"), golden={"output": first["output"]})
            prec = p.get("receipt") or {}
            say("check", edit=name, edit_to_result_s=first_done,
                seconds=[first["seconds"]] + more["seconds"],
                outputs=[first["output"]] + more["outputs"], first_text=next(iter(first["texts"]), None),
                proven=p.get("ok"), witness=prec.get("witness"),
                edit_to_run_s=prec.get("edit_to_run_s"),
                findings=[f.get("check") for f in prec.get("findings") or []])
            u = self.op(op="undo", txn=r.get("txn"))
            urec = u.get("receipt") or {}
            after, after_checks = self.checks(2)
            ab = self.op(op="at_base")
            fingerprint = self.fingerprint(world)
            baseline = (self.warm_checks or self.live_base_checks)[0]["texts"]
            output_is_base = all(check["texts"] == baseline for check in after_checks)
            say("uninstall", edit=name, ok=u.get("ok"), status=urec.get("status"),
                undo_s=urec.get("undo_s"), client_round_trip_s=u.get("client_round_trip_s"),
                check=after, output_is_base=output_is_base,
                at_base=ab.get("ok"), fingerprint_is_base=fp0 is not None and fingerprint == fp0)
            live[name] = first["output"]
            observed.update(prove=p, undo=u, at_base=ab, check=first,
                            checks=[first, *repeats], after=after_checks, fingerprint=fingerprint)
            if self.strict:
                verify_restoration(name, observed, self.warm_checks, fp0)
                verify_proof(name, p, fp0)
        say("live_stop", stop_s=self.stop())
        return live

    def fingerprint(self, world):
        """Read each edited method from the process that owns it."""
        if type(world) is not int or world < 1:
            return None
        result = {}
        for role in ("core", "worker"):
            edits = [name for name, spec in EDITS.items() if role in spec["on"]]
            reply = self.op(op="fingerprint", modules=[EDITS[name]["module"] for name in edits])
            group = reply.get("core" if role == "core" else "workers") or {}
            if reply.get("ok") is not True or group.get("ok") is not True:
                return None
            participants = [group] if role == "core" else group.get("ranks", [])
            if role == "worker":
                ranks = [part.get("rank") for part in participants]
                if (any(type(rank) is not int for rank in ranks)
                        or sorted(ranks) != list(range(world))):
                    return None
            for part in participants:
                code = part.get("code") or {}
                if part.get("ok") is not True or not isinstance(code, dict):
                    return None
                for name in edits:
                    prefix = f"{EDITS[name]['module']}:{METHODS[name]}:"
                    if not any(key.startswith(prefix) for key in code):
                        return None
                if role == "core":
                    result["core"] = code
                else:
                    result.setdefault("workers", {})[part["rank"]] = code
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out")
    ap.add_argument("--serve-args", default="--no-enable-prefix-caching",
                    help="extra `vllm serve` flags for every start (see README: with the "
                         "prefix cache on, a repeated check does not repeat its output)")
    ap.add_argument("--phases", default=",".join(PHASES),
                    help="comma-separated phases to run; a subset is diagnostic")
    a = ap.parse_args()
    phases = a.phases.split(",")
    if not set(phases) <= set(PHASES) or len(set(phases)) != len(phases):
        ap.error("--phases must list distinct phases from warm,fresh,live")
    sys.path.insert(0, LIVELAB)
    import adapter  # noqa: F401  (puts core/livelab on sys.path)
    started = time.monotonic()
    run = Run(a)
    run.setup()
    for e in run.edits.values():
        print(e["diff"], flush=True)
    fresh, restart_s, live = {}, {}, {}
    try:
        if "warm" in phases:
            run.warm()
        if "fresh" in phases:
            fresh["base"], restart_s["base"] = run.restart_check(None)
            for name in EDITS:
                fresh[name], restart_s[name] = run.restart_check(name)
        if "live" in phases:
            live = run.live(restart_s)
        warm = run.warm_checks[0] if run.warm_checks else None
        live_observations = run.full.get("live", {})
        say("agreement", base=warm["output"] if warm else None, fresh=fresh, live=live,
            fresh_base_is_base=(run.fresh_checks["base"][0]["texts"] == warm["texts"]
                                if fresh and warm else None),
            live_equals_fresh={
                name: live_observations[name]["check"]["texts"] == run.fresh_checks[name][0]["texts"]
                for name in live if name in fresh
            })
        if run.strict:
            verify_outputs(run.warm_checks, run.fresh_checks, run.live_base_checks,
                           run.full["live"])
    finally:
        if run.proc is not None:
            run.stop()
        for name in run.edits:
            run.write(name, "base")
        run.full["total_s"] = time.monotonic() - started
        if a.out:
            with open(a.out, "w") as f:
                json.dump(run.full, f, indent=1, default=str)


if __name__ == "__main__":
    try:
        main()
    except VerificationError as exc:
        sys.exit(f"walkthrough verification failed: {exc}")
