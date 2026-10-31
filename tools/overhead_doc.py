"""Write docs/overhead.md from receipts/, so no number here is typed by hand."""

import json


def load(name):
    with open(f"receipts/{name}.json") as handle:
        return json.load(handle)


def rows(engine):
    r = load(f"overhead-{engine}")
    arms, session = r["stock_vs_agent"], r["session"]
    median = {arm: arms["exercise_s"][arm]["median"] for arm in ("stock", "agent")}
    memory = {arm: [m["rss_kib"] / 1024 for m in arms["memory"][arm]] for arm in median}
    gpu = {arm: {m["gpu_mib"] for m in arms["memory"][arm]} for arm in median}
    grow = session["growth"]
    cycles = grow["cycles"]
    host = (grow["last"]["rss_kib"] - grow["first"]["rss_kib"]) / 1024
    return {
        "Workload, stock median": f"{median['stock']:.4f} s",
        "Workload, agent loaded": f"{median['agent']:.4f} s ({arms['idle_cost_pct']:+.2f}%)",
        "Host memory, stock": f"{min(memory['stock']):,.0f}-{max(memory['stock']):,.0f} MiB",
        "Host memory, agent": f"{min(memory['agent']):,.0f}-{max(memory['agent']):,.0f} MiB",
        "GPU memory, stock vs agent": f"{sorted(gpu['stock'])} vs {sorted(gpu['agent'])} MiB",
        "Install, median / max": f"{session['install_s']['median'] * 1e3:.0f} / "
        f"{session['install_s']['max'] * 1e3:.0f} ms",
        "Undo, median / max": f"{session['undo_s']['median'] * 1e3:.0f} / "
        f"{session['undo_s']['max'] * 1e3:.0f} ms",
        "Host growth": f"{host:+.0f} MiB over {cycles} cycles ({host / cycles:+.2f} MiB/cycle)",
        "GPU growth over cycles": f"{grow['last']['gpu_mib'] - grow['first']['gpu_mib']:+d} MiB",
        "Base read, late vs early cycles": f"{session['base_read_drift']['percent']:+.2f}%",
        "Hang detection": f"{session['detect_hang_s']:.0f} s (the configured timeout)",
        "Restart to base": f"{session['recover_s']:.0f} s",
    }


vllm, sglang = rows("vllm"), rows("sglang")
micro = load("witness-microbench")["ns_per_call"]
control = load("control-repeatability")
lines = ["# Overhead book", "",
         "What FastRSI itself costs, generated from `receipts/` by `tools/overhead_doc.py`. "
         "One model, Qwen3-8B, on one GPU per engine. Measured on a shared host.", "",
         "| Cost | vLLM | SGLang |", "|---|---:|---:|"]
for (key, left), right in zip(vllm.items(), sglang.values()):
    lines.append(f"| {key} | {left} | {right} |")
lines += ["",
          "## How to read this", "",
          "- Stock and agent arms alternate in ABBA order, with a full engine start each. The "
          "idle difference is within run-to-run noise on both engines, so read it as not "
          "detected, not as zero.",
          "- SGLang runs one request at a time here, so its figures do not describe batched "
          "serving. See below.",
          "- Host growth is real and repeatable, but its cause is not isolated. It stays small "
          "next to a process of several GiB, and the install budget forces a restart first.",
          "- Hang detection equals the timeout you choose. A shorter one finds hangs sooner and "
          "risks calling a slow request a hang.",
          "- Only sampler edits were measured. Other routes will cost differently.", "",
          "## Witness cost per call", "",
          "The workloads above cannot resolve this, so it was timed directly.", "",
          "| Mode | ns per call |", "|---|---:|",
          f"| Off | {micro['off']:.0f} |",
          f"| First hit only, the default | {micro['first']:.0f} |",
          f"| Every call | {micro['count']:.0f} |", "",
          "## Repeatability", "",
          "Exact comparison only means something if base is repeatable. Each row is 60 identical "
          "runs, with a cache reset between them.", "",
          "| Engine | Distinct outputs |", "|---|---:|"]
for key in ("sglang-stock", "sglang-agent", "vllm-agent"):
    lines.append(f"| {key}, 32 prompts batched | {len(set(control[key]['reset_only_digests']))} |")
lines += ["",
          "SGLang varies with no agent present, so FastRSI does not cause it. Its plugin runs one "
          "request at a time to get a repeatable base. SGLang's deterministic inference mode "
          "failed to start here with a FlashInfer workspace overflow.", ""]
with open("docs/overhead.md", "w") as handle:
    handle.write("\n".join(lines))
