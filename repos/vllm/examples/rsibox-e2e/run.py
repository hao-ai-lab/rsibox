"""Run the rsibox end-to-end cases (cases.json) against one real vLLM box, over HTTP only.

    RSIBOX=<box.json> python3 run.py --cases cases.json --out results.json [--only id,id] [--fresh-state DIR]

The box must be a `vllm --real` box that is `ready` and at `base`. Each case is a list of steps
(install, check, uninstall, status, reset, source, http, fresh). The runner saves the default
workload's output as `base` before each case and requires `base` again after it, so a case that
leaks an edit fails itself and is undone (uninstall to base, then `reset respawn`) before the
next one starts.

`fresh` steps start a second engine from a copy of the vLLM sources with the edit on disk (a
restart) through the local launcher, so they need `--fresh-state` and the same `RSIBOX_VLLM_*`
environment the box was started with. They are skipped, and reported as skipped, without it.

Prints one line per case and writes every reply it judged to `--out`.
"""
import argparse
import difflib
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "core", "livelab"), os.path.join(REPO, "repos", "vllm", "livelab")]

from rsibox import client as C  # noqa: E402
from rsibox import launch as L  # noqa: E402


class Failed(Exception):
    pass


def expect(cond, what, got=None):
    if not cond:
        raise Failed(f"{what}; got {json.dumps(got, default=str)[:400]}")


def apply_edits(text, edits, where):
    for edit in edits:
        count = text.count(edit["old"])
        expect(count == 1, f"{where}: anchor occurs {count} times in {edit['module']}, expected 1", edit["old"][:120])
        text = text.replace(edit["old"], edit["new"])
    return text


def module_path(module):
    return module.replace(".", "/") + ".py"


def unified(module, old, new):
    path = module_path(module)
    body = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))
    return f"diff --git a/{path} b/{path}\n{body}"


def dig(obj, path):
    """`a.b.-1.c` into nested dicts and lists; a final `#` is the length; `*` maps over a list."""
    parts = path.split(".") if path else []
    for i, part in enumerate(parts):
        if part == "#":
            return len(obj) if obj is not None else 0
        if part == "*":
            return [dig(item, ".".join(parts[i + 1:])) for item in obj]
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


class Runner:
    def __init__(self, box, fresh_state=None, log=print):
        self.box, self.fresh_state, self.log = box, fresh_state, log
        self.fresh_n = 0

    # -- one case ------------------------------------------------------------------------
    def run_case(self, case):
        self.labels, self.txns, self.before, self.saved = {}, {}, {}, {}       # per case
        self.trace, t0 = [], time.time()
        result = {"id": case["id"], "lens": case.get("lens"), "title": case.get("title")}
        try:
            self.labels["base"] = self.call("check")["output"]
            for i, step in enumerate(case["steps"]):
                self.where = f"step {i + 1} ({step['op']})"
                getattr(self, "op_" + step["op"])(step)
            self.where = "after the case"
            state = self.call("status")
            expect(state.get("state") == "base", "the case must end at base", state.get("state"))
            expect(self.call("check")["output"] == self.labels["base"], "base output must be back after the case")
            result["result"] = "pass"
        except Skip as skip:
            result.update(result="skip", why=str(skip))
        except Failed as failed:
            result.update(result="FAIL", why=f"{self.where}: {failed}")
        except Exception as exc:  # noqa: BLE001 - a runner bug must not hide the other cases
            result.update(result="ERROR", why=f"{self.where}: {type(exc).__name__}: {exc}")
        result["seconds"] = round(time.time() - t0, 2)
        result["trace"] = self.trace
        if result["result"] != "pass":
            result["cleanup"] = self.cleanup()
        return result

    def cleanup(self):
        """Back to base whatever the case left: undo every txn, respawn if still not base."""
        done = []
        status = self.box.status()
        for _ in range(len(status.get("txns") or [])):
            done.append(("uninstall", self.box.uninstall().get("ok")))
        status = self.box.status()
        if status.get("state") != "base":
            done.append(("reset respawn", self.box.reset("respawn").get("ok")))
            status = self.box.status()
        done.append(("state", status.get("state")))
        return done

    def call(self, name, *args):
        reply = getattr(self.box, name)(*args)
        keep = {k: v for k, v in reply.items() if k not in ("receipt", "text", "findings")}
        if reply.get("findings"):
            keep["finding_checks"] = sorted({f.get("check") for f in reply["findings"]})
        self.trace.append({"call": name, "reply": keep})
        return reply

    # -- steps ---------------------------------------------------------------------------
    def live(self, module):
        src = self.call("source", module)
        expect(src.get("ok"), f"{self.where}: source {module}", src)
        return src

    def build_patch(self, step):
        if "raw_files" in step:
            return {"files": dict(step["raw_files"])}
        files, base, diffs = {}, {}, []
        for module in dict.fromkeys(e["module"] for e in step["edits"]):
            src = self.live(module)
            start = self.before.get(module, src) if step.get("stale") else src
            new = apply_edits(start["text"], [e for e in step["edits"] if e["module"] == module], self.where)
            files[module], base[module] = new, start["hash"]
            diffs.append(unified(module, start["text"], new))
            self.pending = getattr(self, "pending", {})
            self.pending[module] = src
        if step.get("mode") == "diff":
            return {"diff": "".join(diffs)}                    # a stale diff is the lab's to refuse (it does not apply)
        patch = {"files": files}
        if step.get("stale") or step.get("send_base"):
            patch["base"] = base                               # full text from a stale read: the box refuses it
        return patch

    def op_install(self, step):
        self.pending = {}
        patch = self.build_patch(step)
        if step.get("accept"):
            patch["accept"] = list(step["accept"])
        reply = self.call("install", patch)
        want = (step.get("expect") or {}).get("outcome", "committed")
        if want == "committed":
            expect(reply.get("ok") and reply.get("outcome") == "committed", f"{self.where}: install commits",
                   {k: reply.get(k) for k in ("outcome", "codes", "error")})
            self.before.update(self.pending)                   # the text before this commit, for "stale"
            if step.get("as"):
                self.txns[step["as"]] = reply["id"]
        else:
            expect(not reply.get("ok") and reply.get("outcome") != "committed", f"{self.where}: install must not commit",
                   {k: reply.get(k) for k in ("outcome", "codes")})
        if step.get("as"):
            self.saved[step["as"]] = reply
        for code in (step.get("expect") or {}).get("codes_include", []):
            expect(code in (reply.get("codes") or []), f"{self.where}: codes include {code}", reply.get("codes"))

    def op_check(self, step):
        reply = self.call("check", step.get("workload"))
        expect(reply.get("ok"), f"{self.where}: check ok", {k: reply.get(k) for k in ("error", "findings")})
        if step.get("save"):
            self.labels[step["save"]] = reply["output"]
        want = step.get("expect") or {}
        if "output" in want:
            kind, _, label = want["output"].partition(":")
            expect(label in self.labels, f"{self.where}: label {label!r} was never saved")
            same = reply["output"] == self.labels[label]
            expect(same if kind == "eq" else not same, f"{self.where}: output {kind} {label}",
                   {"output": reply["output"], label: self.labels[label]})
        if "proven" in want:
            expect(reply.get("proven") is want["proven"], f"{self.where}: proven is {want['proven']}",
                   {"proven": reply.get("proven"), "witness": reply.get("witness")})
        if "witness_module" in want:
            hits = {k: n for part in (reply.get("witness") or {}).values() for k, n in part.items()
                    if k.startswith(want["witness_module"] + ":") and n > 0}
            expect(hits, f"{self.where}: a function of {want['witness_module']} ran", reply.get("witness"))

    def op_uninstall(self, step):
        target = step.get("to")
        reply = self.call("uninstall", self.txns.get(target, target))      # an unknown name is sent as a literal id
        want = step.get("expect") or {}
        expect(reply.get("ok") is want.get("ok", True), f"{self.where}: uninstall ok is {want.get('ok', True)}", reply)
        if "undone" in want:
            expect(len(reply.get("undone") or []) == want["undone"], f"{self.where}: {want['undone']} undone", reply.get("undone"))
        if "state" in want:
            expect(reply.get("state") == want["state"], f"{self.where}: state {want['state']}", reply.get("state"))

    def op_status(self, step):
        reply = self.call("status")
        want = step.get("expect") or {}
        if "state" in want:
            expect(reply.get("state") == want["state"], f"{self.where}: state {want['state']}", reply.get("state"))
        if "txns" in want:
            expect(len(reply.get("txns") or []) == want["txns"], f"{self.where}: {want['txns']} txns", reply.get("txns"))

    def op_reset(self, step):
        reply = self.call("reset", step.get("kind", "state"))
        expect(reply.get("ok"), f"{self.where}: reset ok", reply)

    def op_source(self, step):
        text = self.live(step["module"])["text"]
        want = step.get("expect") or {}
        if "contains" in want:
            expect(want["contains"] in text, f"{self.where}: source contains the text", want["contains"][:80])
        if "not_contains" in want:
            expect(want["not_contains"] not in text, f"{self.where}: source does not contain the text", want["not_contains"][:80])

    def op_http(self, step):
        token = {"good": self.box.token, "wrong": self.box.token + "x", "none": None}[step.get("token", "good")]
        data = None
        if step.get("method", "GET") == "POST":
            data = step["raw_body"].encode() if "raw_body" in step else json.dumps(step["body"] if "body" in step else {}).encode()
        path = step["path"]
        for label in self.saved:                               # "{label.path}" inside the request path
            while "{" + label + "." in path:
                ref = path[path.index("{" + label + "."):path.index("}", path.index("{" + label + ".")) + 1]
                path = path.replace(ref, str(self.ref("$" + ref[1:-1])))
        req = urllib.request.Request(self.box.url + path, data=data, method=step.get("method", "GET"),
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {}
        self.trace.append({"call": f"{step.get('method', 'GET')} {path}", "http": status})
        if step.get("save"):
            self.saved[step["save"]] = payload
        want = step.get("expect") or {}
        if "http" in want:
            expect(status == want["http"], f"{self.where}: HTTP {want['http']}", status)
        for key, val in (want.get("json") or {}).items():
            expect(dig(payload, key) == self.ref(val), f"{self.where}: reply {key} == {val!r}", dig(payload, key))
        for key, val in (want.get("json_ne") or {}).items():
            expect(dig(payload, key) != self.ref(val), f"{self.where}: reply {key} != {val!r}", dig(payload, key))
        for key, rule in (want.get("grew") or {}).items():
            was, now = len(dig(self.saved[rule["since"]], key) or []), len(dig(payload, key) or [])
            expect(now == was + rule["by"], f"{self.where}: {key} grew by {rule['by']} since {rule['since']}", [was, now])
        if "strictly_increasing" in want:
            seq = dig(payload, want["strictly_increasing"]) or []
            expect(all(a < b for a, b in zip(seq, seq[1:])), f"{self.where}: {want['strictly_increasing']} only grows", seq[-8:])

    def ref(self, value):
        """`$label.path` is a value from a saved reply; anything else is itself."""
        if isinstance(value, str) and value.startswith("$"):
            label, _, path = value[1:].partition(".")
            expect(label in self.saved, f"{self.where}: no saved reply {label!r}")
            return dig(self.saved[label], path)
        return value

    def op_fresh(self, step):
        """A restart with the edit on disk: a second box whose vLLM comes from a patched copy."""
        if not self.fresh_state:
            raise Skip("fresh steps need --fresh-state")
        from adapter import conformance_lab as CL

        self.fresh_n += 1
        root = os.path.join(self.fresh_state, f"fresh-{self.fresh_n}")
        shutil.rmtree(root, ignore_errors=True)
        farm = os.path.join(root, "farm")
        site = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
        CL.build_farm(os.path.join(site, "vllm"), os.path.join(farm, "vllm"))
        for module in dict.fromkeys(e["module"] for e in step["edits"]):
            path = os.path.join(farm, module_path(module))
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(apply_edits(text, [e for e in step["edits"] if e["module"] == module], self.where))
        env = {k: v for k, v in os.environ.items() if k.startswith(("RSIBOX_VLLM_", "CUDA_", "HF_", "VLLM_"))}
        env.update(PYTHONPATH=farm, RSIBOX_VLLM_WORK=os.path.join(root, "work"),
                   RSIBOX_VLLM_PORT=str(int(os.environ.get("RSIBOX_VLLM_PORT", "18500")) + 20))
        state = os.path.join(root, "box")
        record = L.up({"launcher": "local", "lab": "vllm", "real": True, "state": state, "env": env, "repo": REPO})
        try:
            expect(record.get("ok"), f"{self.where}: fresh box starts", record.get("error"))
            fresh = C.Client.from_box_file(record["box_file"])
            info = fresh.wait(timeout=1500)
            expect(info.get("ok"), f"{self.where}: fresh engine boots", info.get("boot_error"))
            reply = fresh.check(step.get("workload"))
            expect(reply.get("ok"), f"{self.where}: fresh check ok", reply.get("error"))
            self.labels[step["save"]] = reply["output"]
            self.trace.append({"call": "fresh", "boot_s": info.get("boot_s"), "output": reply["output"]})
        finally:
            L.down(state)
            shutil.rmtree(farm, ignore_errors=True)


class Skip(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--fresh-state", default=None)
    a = ap.parse_args(argv)
    with open(a.cases, encoding="utf-8") as handle:
        cases = json.load(handle)
    only = {x for x in a.only.split(",") if x}
    cases = [c for c in cases if not only or c["id"] in only]
    box = C.Client.from_env(timeout=900)
    info = box.info()
    if info.get("state") != "ready":
        print(f"the box is not ready: {info}", file=sys.stderr)
        return 2
    runner, results = Runner(box, a.fresh_state), []
    for case in cases:
        if any(s["op"] == "scenario" for s in case["steps"]):
            results.append({"id": case["id"], "result": "skip", "why": "scenario: run by scenarios.py", "seconds": 0})
        else:
            results.append(runner.run_case(case))
        r = results[-1]
        print(f"{r['result']:<5} {r['seconds']:>7.2f}s  {r['id']}" + (f"  -- {r['why'][:300]}" if r.get("why") else ""), flush=True)
        with open(a.out, "w", encoding="utf-8") as handle:
            json.dump({"box": info, "results": results}, handle, indent=1, default=str)
    count = lambda kind: sum(1 for r in results if r["result"] == kind)  # noqa: E731
    print(f"\n{count('pass')} pass, {count('FAIL')} fail, {count('ERROR')} error, {count('skip')} skip of {len(results)}")
    return 1 if count("FAIL") or count("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
