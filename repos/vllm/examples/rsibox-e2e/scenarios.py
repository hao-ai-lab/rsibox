"""Box-level scenarios on a real vLLM engine that step lists cannot express.

    python3 scenarios.py --state DIR [--only boot,busy,kill,down] [--out scenarios.json]

Starts its own box with the local launcher (the `FASTRSI_VLLM_*` environment must be set as
for any `vllm --real` box) and runs, in order:

- boot: verbs sent while the engine boots get 409 `booting`, `info` answers at once, then the
  box is `ready` with `boot_s` and a check works.
- busy: while a long check runs, `info` answers at once with the check in flight, a second verb
  gets 409 `busy`, and the long check still returns the same output as a quiet run.
- kill: the engine's worker process is killed from outside. The daemon stays up, the next
  check fails, the session is poisoned, an install is refused with `session_poisoned`,
  `reset respawn` returns it to base and the base output is back.
- down: `rsibox down` leaves no process of this box and none of its GPU memory.

Only processes below this box's own daemon are ever signalled.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "core", "livelab"), os.path.join(REPO, "repos", "vllm", "livelab")]

from rsibox import client as C  # noqa: E402
from rsibox import launch as L  # noqa: E402

SAMPLER = "vllm.v1.worker.gpu.sample.sampler"
ANCHOR = "        sampled, processed_logits = self.sample(\n"


class Failed(Exception):
    pass


def expect(cond, what, got=None):
    if not cond:
        raise Failed(f"{what}; got {json.dumps(got, default=str)[:500]}")


def descendants(root):
    """Every pid below `root` (children of children), read from /proc."""
    parents = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as handle:
                    parents[int(name)] = int(handle.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    out, frontier = [], [root]
    while frontier:
        kids = [pid for pid, parent in parents.items() if parent in frontier]
        out += kids
        frontier = kids
    return out


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def engine_pids(port):
    """This box's `vllm serve` on `port` and everything below it. The engine starts in its own
    session, so it is found by its port, which only this box uses."""
    roots = [int(p) for p in subprocess.run(["pgrep", "-f", f"vllm serve .*--port {port} "],
                                            capture_output=True, text=True).stdout.split()]
    pids = set(roots)
    for root in roots:
        pids.update(descendants(root))
    return sorted(pids)


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()}


def scenario_boot(ctx):
    record = L.up({"launcher": "local", "lab": "vllm", "real": True, "state": ctx["state"], "repo": REPO})
    expect(record.get("ok"), "the box starts", record.get("error"))
    box = ctx["box"] = C.Client.from_box_file(record["box_file"], timeout=900)
    t0 = time.time()
    info = box.info()
    answered = round(time.time() - t0, 3)
    expect(info.get("state") == "booting" and info.get("boot_s") is None, "info says booting at once", info)
    expect(answered < 2, "info answers within 2 s during boot", answered)
    early = box.check()
    expect((early.get("http"), early.get("state")) == (409, "booting"), "a check during boot gets 409 booting", early)
    refused = box.install({"files": {SAMPLER: "x = 1\n"}})
    expect(refused.get("http") == 409, "an install during boot gets 409", refused)
    ready = box.wait(timeout=1500)
    expect(ready.get("ok") and ready.get("state") == "ready" and ready.get("boot_s") > 1, "the box becomes ready", ready)
    ctx["base"] = box.check()
    expect(ctx["base"].get("ok"), "a check works after boot", ctx["base"])
    expect(box.status().get("state") == "base", "the session starts at base")
    return {"info_answered_s": answered, "boot_s": ready["boot_s"]}


def scenario_busy(ctx):
    box = ctx["box"]
    work = {"prompts": [f"Count from {i} to {i + 40}:" for i in range(48)], "max_tokens": 48}
    quiet = box.check(work)
    expect(quiet.get("ok"), "the long workload runs", quiet.get("error"))
    expect(quiet["cost_s"] > 8, "the long check outlasts the 5 s busy wait (raise the workload if not)", quiet["cost_s"])
    result = {}
    thread = threading.Thread(target=lambda: result.update(reply=box.check(work)), daemon=True)
    thread.start()
    deadline = time.time() + 5
    info = box.info()
    while info.get("in_flight") is None and time.time() < deadline:
        time.sleep(0.05)
        info = box.info()
    expect((info.get("in_flight") or {}).get("verb") == "check", "info shows the check in flight", info)
    t0 = time.time()
    second = box.status()
    waited = round(time.time() - t0, 2)
    expect((second.get("http"), second.get("state")) == (409, "busy"), "a second verb gets 409 busy", second)
    expect((second.get("in_flight") or {}).get("verb") == "check", "the 409 names the verb in flight", second)
    expect(4 < waited < 8, "the second verb waited about busy_wait_s (5 s)", waited)
    thread.join(timeout=300)
    long = result.get("reply") or {}
    expect(long.get("ok") and long.get("output") == quiet["output"], "the long check was not disturbed", long.get("error"))
    expect(box.status().get("state") == "base", "status works again afterwards")
    return {"long_check_s": quiet["cost_s"], "second_verb_waited_s": waited}


def scenario_kill(ctx):
    box, port = ctx["box"], int(os.environ.get("FASTRSI_VLLM_PORT", "18500"))
    before = engine_pids(port)
    workers = [p for p in before if "Worker" in cmdline(p)]
    expect(workers, "the engine has a worker process below this box", [(p, cmdline(p)[:60]) for p in before])
    text = box.source(SAMPLER)["text"]
    done = box.install({"files": {SAMPLER: text.replace(ANCHOR, "        logits = -logits\n" + ANCHOR)}})
    expect(done.get("ok"), "an edit is installed before the fault", done.get("codes"))
    for pid in workers:
        os.kill(pid, 9)
    time.sleep(3)
    broken = box.check()
    expect(broken.get("http") == 200 and not broken.get("ok"), "the check after the kill fails but is answered", broken)
    expect(box.info().get("state") == "ready", "the daemon is still up")
    status = box.status()
    expect(status.get("state") == "poisoned", "the session is poisoned", status)
    refused = box.install({"files": {SAMPLER: text.replace(ANCHOR, "        logits = logits * 2\n" + ANCHOR)}})
    expect(not refused.get("ok") and "session_poisoned" in (refused.get("codes") or []),
           "an install while poisoned is refused", {k: refused.get(k) for k in ("outcome", "codes")})
    t0 = time.time()
    reset = box.reset("respawn")
    respawn_s = round(time.time() - t0, 1)
    expect(reset.get("ok"), "reset respawn succeeds", reset)
    status = box.status()
    expect(status.get("state") == "base" and not status.get("txns"), "the session is at base with no edit", status)
    after = box.check()
    expect(after.get("ok") and after.get("output") == ctx["base"]["output"], "the base output is back", after.get("output"))
    expect("logits = -logits" not in box.source(SAMPLER)["text"], "the live text is the base text")
    expect(not set(workers) & set(engine_pids(port)), "the killed worker was replaced")
    return {"killed": len(workers), "respawn_s": respawn_s}


def scenario_down(ctx):
    box, port = ctx["box"], int(os.environ.get("FASTRSI_VLLM_PORT", "18500"))
    with open(os.path.join(ctx["state"], "box.json")) as handle:
        daemon = json.load(handle)["pid"]
    ours = set(engine_pids(port)) | {daemon} | set(descendants(daemon))
    expect(ours & gpu_pids(), "the box holds GPU memory before down", sorted(ours))
    reply = L.down(ctx["state"])
    expect(reply.get("ok"), "down succeeds", reply)
    deadline = time.time() + 60
    while time.time() < deadline and ({p for p in ours if os.path.exists(f"/proc/{p}")} or ours & gpu_pids()):
        time.sleep(1)
    left = sorted(p for p in ours if os.path.exists(f"/proc/{p}"))
    expect(not left, "no process of the box is left", [(p, cmdline(p)[:80]) for p in left])
    expect(not ours & gpu_pids(), "none of its processes holds GPU memory")
    expect(box.info().get("http") == 0, "the port no longer answers")
    ctx["box"] = None
    return {"processes_checked": len(ours)}


SCENARIOS = {"boot": scenario_boot, "busy": scenario_busy, "kill": scenario_kill, "down": scenario_down}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--only", default="boot,busy,kill,down")
    ap.add_argument("--out", default="scenarios.json")
    a = ap.parse_args(argv)
    ctx, results = {"state": os.path.abspath(a.state)}, []
    try:
        for name in a.only.split(","):
            t0 = time.time()
            try:
                results.append({"id": name, "result": "pass", **SCENARIOS[name](ctx)})
            except Failed as failed:
                results.append({"id": name, "result": "FAIL", "why": str(failed)})
            except Exception as exc:  # noqa: BLE001
                results.append({"id": name, "result": "ERROR", "why": f"{type(exc).__name__}: {exc}"})
            results[-1]["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(results[-1]), flush=True)
            if name == "boot" and results[-1]["result"] != "pass":
                break
    finally:
        if ctx.get("box") is not None:
            L.down(ctx["state"])
        with open(a.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
    return 0 if all(r["result"] == "pass" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
