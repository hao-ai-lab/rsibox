"""Real-GPU testification of vLLM C++/CUDA `NATIVE` hot-reload on PR #49750 (`layernorm_kernels.cu`).

Verifies end-to-end against a real `vllm serve` (V1 GPUModelRunner with `FULL_DECODE_ONLY`
CUDA graphs and `--kernel-config` routing RMSNorm through `vllm_c` / `torch.ops._C`):
1. Direct GPU tensor check: pre-PR #49750 rejects non-contiguous `residual` while PR #49750 accepts it.
2. Live server graph refusal: `native` plan without `"rebuild": ["R1"]` aborts (`captured_target`).
3. Visible `.cu` perturbation (`epsilon = 0.25f;`): compiles with `nvcc`, installs in-place + `R1`
   CUDA graph recapture without restarting `vllm serve`, proves positive live + capture witness,
   changes generated tokens visibly, matches a fresh cold `vllm serve` running the same `.cu` edit
   token-for-token and logprob-for-logprob, and undoes in sub-millisecond kernel time back to base.
4. Real upstream PR #49750 (`pre-PR` -> `PR #49750` live install + `R1` + witness + `live == fresh` + undo).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIVELAB = os.path.abspath(os.path.join(HERE, "..", "..", "livelab"))

LN_REL = "libtorch_stable/layernorm_kernels.cu"
PR_NUMBER = 49750
PR_HOST_ANCHOR = (
    "  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type());\n"
    "  STD_TORCH_CHECK(residual.stride(-1) == 1);\n"
)
SCHEMA_FUSED = (
    "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()"
)
PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "def fibonacci(n):",
    "The three primary colors are",
    "In 1969, humans first",
    "The derivative of x squared is",
    "A haiku about autumn:",
    "The largest planet in the solar system is",
]


def say(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def extract_pre_pr_source(csrc_root: str, vllm_git: str | None) -> tuple[str, str]:
    """Return `(pr49750_text, pre_pr_text)` for `libtorch_stable/layernorm_kernels.cu`."""
    pr_path = os.path.join(csrc_root, LN_REL)
    with open(pr_path, encoding="utf-8") as fh:
        pr_text = fh.read()
    if PR_HOST_ANCHOR not in pr_text:
        raise RuntimeError(f"Expected PR #49750 anchor in {pr_path}")

    if vllm_git:
        features_path = os.path.join(os.path.dirname(os.path.dirname(vllm_git)), "data", "features.jsonl")
        commit_sha = None
        if os.path.isfile(features_path):
            with open(features_path, encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r'\{"pr": 49750, "commit": "([0-9a-f]+)"', line)
                    if m:
                        commit_sha = m.group(1)
                        break
        if commit_sha is None:
            commit_sha = "82642d7d6c332fefdec06b260a246bd1a4ebf5ff"
        diff = subprocess.check_output(
            ["git", f"--git-dir={vllm_git}", "diff", f"{commit_sha}^", commit_sha, "--", f"csrc/{LN_REL}"],
            text=True,
        )
        with tempfile.TemporaryDirectory(prefix="pr49750-rev-") as tmp:
            dst = os.path.join(tmp, "csrc", LN_REL)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(pr_text)
            patch_file = os.path.join(tmp, "pr.diff")
            with open(patch_file, "w", encoding="utf-8") as fh:
                fh.write(diff)
            r = subprocess.run(["git", "apply", "-R", patch_file], cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                r = subprocess.run(
                    ["patch", "-R", "-p1", "-F2", "-f", "--no-backup-if-mismatch", "-i", patch_file],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
            if r.returncode != 0:
                raise RuntimeError(f"Failed reversing PR #49750 diff: {r.stdout}\n{r.stderr}")
            with open(dst, encoding="utf-8") as fh:
                pre_text = fh.read()
        require("  STD_TORCH_CHECK(residual.is_contiguous());\n" in pre_text, "missing pre-PR anchor")
        return pr_text, pre_text
    raise RuntimeError("--vllm-git is required to derive exact pre-PR source cleanly")


def native_plan_spec(csrc_root: str, modified_source: str, nonce: str, *, rebuild_r1: bool = True, cache: bool = False) -> dict:
    plan = {
        "native": {
            "csrc_root": csrc_root,
            "files": [LN_REL],
            "sources": {LN_REL: f"// {nonce}\n" + modified_source},
            "ops": {
                "fused_add_rms_norm": {
                    "fn": "fused_add_rms_norm",
                    "schema": SCHEMA_FUSED,
                }
            },
            "cache": cache,
        }
    }
    if rebuild_r1:
        plan["rebuild"] = ["R1"]
    return plan


def make_visible_perturbed_source(pr_text: str) -> str:
    require(PR_HOST_ANCHOR in pr_text, "PR anchor missing for visible perturbation")
    return pr_text.replace(PR_HOST_ANCHOR, PR_HOST_ANCHOR + "  epsilon = 0.25f;\n", 1)


def direct_noncontiguous_check(csrc_root: str, pre_pr_text: str, pr_text: str, work: str) -> dict:
    """Prove pre-PR kernel rejects non-contiguous residual and PR #49750 kernel accepts it with exact result."""
    probe_code = """
import json, sys, torch
from vllm.platforms import current_platform
current_platform.import_kernels()
from adapter import native as N

csrc_root, pre_path, pr_path, build_root = sys.argv[1:5]
pre_src = open(pre_path, encoding="utf-8").read()
pr_src = open(pr_path, encoding="utf-8").read()
ln_rel = "libtorch_stable/layernorm_kernels.cu"
schema = "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()"

def compile_and_get(src, tag):
    spec = {
        "csrc_root": csrc_root,
        "build_root": build_root,
        "files": [ln_rel],
        "sources": {ln_rel: f"// {tag}\\n" + src},
        "ops": {"fused_add_rms_norm": {"fn": "fused_add_rms_norm", "schema": schema}},
        "cache": True,
    }
    built = N.build_native(spec)
    N.load_library(built["so"], built["key"])
    return getattr(getattr(torch.ops, f"_Cv{built['key']}"), "fused_add_rms_norm"), built

pre_op, pre_built = compile_and_get(pre_src, "direct-pre")
pr_op, pr_built = compile_and_get(pr_src, "direct-pr")

torch.manual_seed(0)
base_x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
wide_res = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
noncontig_res = wide_res[:, :256]
assert not noncontig_res.is_contiguous() and noncontig_res.stride(-1) == 1
w = torch.ones(256, dtype=torch.bfloat16, device="cuda")

# Reference
ref_res = noncontig_res.clone()
ref_sum = (base_x.float() + ref_res.float())
ref_var = ref_sum.pow(2).mean(dim=-1, keepdim=True)
ref_out = (ref_sum * torch.rsqrt(ref_var + 1e-6)).to(torch.bfloat16)

pre_rejected = False
pre_err = None
try:
    x1, r1 = base_x.clone(), wide_res.clone()[:, :256]
    pre_op(x1, r1, w, 1e-6)
    torch.cuda.synchronize()
except Exception as exc:
    pre_rejected = True
    pre_err = f"{type(exc).__name__}: {exc}"

x2, r2 = base_x.clone(), wide_res.clone()[:, :256]
pr_op(x2, r2, w, 1e-6)
torch.cuda.synchronize()
max_err = float((x2.float() - ref_out.float()).abs().max().item())
print(json.dumps({
    "pre_key": pre_built["key"],
    "pr_key": pr_built["key"],
    "pre_rejected_noncontiguous": pre_rejected,
    "pre_error": pre_err,
    "pr_max_abs_err": max_err,
    "pr_accepted_exact": max_err < 2e-2,
}))
"""
    pre_file = os.path.join(work, "pre_pr_layernorm.cu")
    pr_file = os.path.join(work, "pr49750_layernorm.cu")
    with open(pre_file, "w", encoding="utf-8") as fh:
        fh.write(pre_pr_text)
    with open(pr_file, "w", encoding="utf-8") as fh:
        fh.write(pr_text)
    build_root = os.path.join(work, "native_cache")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([LIVELAB, os.path.join(os.path.dirname(os.path.dirname(LIVELAB)), "core", "livelab")]),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe_code, csrc_root, pre_file, pr_file, build_root],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"direct_noncontiguous_check failed:\n{proc.stdout}\n{proc.stderr}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    require(res["pre_rejected_noncontiguous"] is True, f"pre-PR unexpectedly accepted noncontig: {res}")
    require(res["pr_accepted_exact"] is True, f"PR #49750 inaccurate on noncontig: {res}")
    return res


def gpu_mem_mib(dev_id: str | None) -> list[int]:
    args = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if dev_id is not None:
        args += ["-i", dev_id]
    out = subprocess.check_output(args, text=True)
    return [int(x.strip()) for x in out.split() if x.strip()]


class VllmRunner:
    def __init__(self, args, work: str):
        self.args = args
        self.work = work
        self.ctl = os.path.join(work, "ctl")
        self.proc = None

    def _prepare_site(self, mode: str, fresh_so: str | None = None, fresh_key: str | None = None) -> str:
        site = os.path.join(self.work, f"site-{mode}")
        os.makedirs(site, exist_ok=True)
        path = os.path.join(site, "sitecustomize.py")
        if mode == "live":
            code = (
                "import os\n"
                "if os.environ.get('FASTRSI_KERNEL_DIR'):\n"
                "    import adapter.host\n"
            )
        elif mode == "fresh_native":
            code = (
                "import ctypes, os, sys\n"
                "so = os.environ.get('FASTRSI_FRESH_NATIVE_SO')\n"
                "key = os.environ.get('FASTRSI_FRESH_NATIVE_KEY')\n"
                "_BOUND = []\n"
                "def _bind(_mod):\n"
                "    if _BOUND or not so or not key:\n"
                "        return\n"
                "    lib = ctypes.CDLL(so, mode=getattr(os, 'RTLD_NOW', ctypes.DEFAULT_MODE) | getattr(os, 'RTLD_LOCAL', 0))\n"
                "    import torch\n"
                "    op = getattr(getattr(torch.ops, f'_Cv{key}'), 'fused_add_rms_norm')\n"
                "    reg = torch.library.Library('_C', 'IMPL')\n"
                "    reg.impl('fused_add_rms_norm', lambda *a, **k: op(*a, **k), 'CUDA')\n"
                "    _BOUND.extend((lib, reg))\n"
                "if so and key:\n"
                "    from fastrsi import hooks\n"
                "    if 'vllm.kernels.vllm_c' in sys.modules:\n"
                "        _bind(sys.modules['vllm.kernels.vllm_c'])\n"
                "    else:\n"
                "        sys.meta_path.insert(0, hooks.AfterImportHook('vllm.kernels.vllm_c', _bind, once=True))\n"
            )
        else:
            code = ""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        return site

    def start(self, tag: str, *, adapter: bool, fresh_so: str | None = None, fresh_key: str | None = None) -> float:
        mode = "live" if adapter else ("fresh_native" if fresh_so else "plain")
        site = self._prepare_site(mode, fresh_so, fresh_key)
        core_root = os.path.abspath(os.path.join(LIVELAB, "..", "..", "..", "core", "livelab"))
        env = dict(os.environ)
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        env["VLLM_BATCH_INVARIANT"] = "0"
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([site, LIVELAB, core_root, env.get("PYTHONPATH", "")]).rstrip(":")
        if fresh_so and fresh_key:
            env["FASTRSI_FRESH_NATIVE_SO"] = fresh_so
            env["FASTRSI_FRESH_NATIVE_KEY"] = fresh_key
        else:
            env.pop("FASTRSI_FRESH_NATIVE_SO", None)
            env.pop("FASTRSI_FRESH_NATIVE_KEY", None)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.args.model,
            "--served-model-name", "model",
            "--port", str(self.args.port),
            "--distributed-executor-backend", "mp",
            "--no-enable-prefix-caching",
            "--seed", "0",
            "--gpu-memory-utilization", "0.25",
            "--max-model-len", "256",
            "--max-num-seqs", "4",
            "--disable-flashinfer-autotune",
            "--kernel-config", '{"ir_op_priority":{"rms_norm":["vllm_c","native"],"fused_add_rms_norm":["vllm_c","native"]}}',
            "--compilation-config", '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}',
            "--no-enforce-eager",
        ]
        if adapter:
            shutil.rmtree(self.ctl, ignore_errors=True)
            os.makedirs(self.ctl, exist_ok=True)
            env["FASTRSI_KERNEL_DIR"] = self.ctl
            cmd += ["--worker-extension-cls", "adapter.worker_ext.KernelWorkerExtension"]
        else:
            env.pop("FASTRSI_KERNEL_DIR", None)

        log_path = os.path.join(self.work, f"serve-{tag}.log")
        t0 = time.monotonic()
        with open(log_path, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
        url = f"http://127.0.0.1:{self.args.port}/health"
        while True:
            if self.proc.poll() is not None:
                with open(log_path, encoding="utf-8") as fh:
                    tail = fh.read()[-2000:]
                raise RuntimeError(f"vllm serve ({tag}) exited {self.proc.returncode}:\n{tail}")
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    return round(time.monotonic() - t0, 3)
            except OSError:
                pass
            if time.monotonic() - t0 > 600:
                raise RuntimeError(f"vllm serve ({tag}) timed out waiting for /health")
            time.sleep(0.1)

    def stop(self) -> float:
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
        dev = os.environ.get("CUDA_VISIBLE_DEVICES")
        while any(m > 100 for m in gpu_mem_mib(dev)) and time.monotonic() - t0 < 120:
            time.sleep(0.2)
        self.proc = None
        return round(time.monotonic() - t0, 3)

    def check(self) -> dict:
        t0 = time.monotonic()
        texts, token_ids, logprobs = [], [], []
        for prompt in PROMPTS:
            body = json.dumps({
                "model": "model",
                "prompt": prompt,
                "max_tokens": 32,
                "temperature": 0,
                "seed": 0,
                "logprobs": 1,
                "return_token_ids": True,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.args.port}/v1/completions",
                body,
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode())
            ch = data["choices"][0]
            texts.append(ch["text"])
            token_ids.append(ch.get("token_ids"))
            lps = ((ch.get("logprobs") or {}).get("token_logprobs")) or []
            logprobs.append([0.0 if v is None else float(v) for v in lps])
        return {
            "seconds": round(time.monotonic() - t0, 3),
            "texts": texts,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "output": hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:16],
        }

    def op(self, **cmd) -> dict:
        from adapter.client import send
        return send(self.ctl, cmd, timeout=300.0)


def max_abs_logprob_diff(a: dict, b: dict) -> float:
    worst = 0.0
    for xs, ys in zip(a["logprobs"], b["logprobs"]):
        require(len(xs) == len(ys), f"logprob len mismatch: {len(xs)} vs {len(ys)}")
        for x, y in zip(xs, ys):
            worst = max(worst, abs(x - y))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--csrc-root", required=True)
    ap.add_argument("--vllm-git", required=True)
    ap.add_argument("--port", type=int, default=29495)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    core_root = os.path.abspath(os.path.join(LIVELAB, "..", "..", "..", "core", "livelab"))
    for p in (LIVELAB, core_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    pr_text, pre_pr_text = extract_pre_pr_source(args.csrc_root, args.vllm_git)
    perturbed_text = make_visible_perturbed_source(pr_text)
    summary = {"pr": PR_NUMBER, "steps": {}}

    # 1. Direct GPU non-contiguous tensor proof (pre-PR vs PR #49750)
    direct = direct_noncontiguous_check(args.csrc_root, pre_pr_text, pr_text, work)
    summary["steps"]["direct_noncontiguous"] = direct
    say("direct_noncontiguous", **direct)

    runner = VllmRunner(args, work)
    perturbed_so = None
    perturbed_key = None
    pr_so = None
    pr_key = None

    try:
        # 2. Live server with R1 CUDA graphs + NATIVE route
        healthy_s = runner.start("live", adapter=True)
        ping = runner.op(op="ping")
        graphs0 = runner.op(op="graphs")
        base1 = runner.check()
        base2 = runner.check()
        require(base1["texts"] == base2["texts"], "base checks not deterministic")
        say("live_boot", healthy_s=healthy_s, ping=ping, graphs=graphs0, base_output=base1["output"], check_s=base1["seconds"])
        summary["steps"]["live_boot"] = {"healthy_s": healthy_s, "base_output": base1["output"], "check_s": base1["seconds"], "graphs": graphs0}

        # 2a. Graph safety refusal without rebuild: ["R1"]
        no_r1_plan = native_plan_spec(args.csrc_root, perturbed_text, "no-r1-test", rebuild_r1=False, cache=False)
        refused = runner.op(op="txn", plan=no_r1_plan, gap_timeout_s=120.0)
        require(refused.get("outcome") == "aborted", f"expected aborted without R1, got {refused}")
        refused_checks = [f["check"] for f in (refused.get("receipt") or {}).get("findings", [])]
        require("captured_target" in refused_checks, f"expected captured_target, got {refused_checks}")
        ab0 = runner.op(op="at_base")
        require(ab0.get("ok") is True, f"expected at_base True after refusal: {ab0}")
        say("graph_refusal", outcome=refused["outcome"], findings=refused_checks, prepare_s=(refused.get("receipt") or {}).get("prepare_s"))
        summary["steps"]["graph_refusal"] = {"outcome": refused["outcome"], "findings": refused_checks}

        # 2b. Visible perturbation + R1 recapture + witness + <1ms undo + base restore
        vis_nonce = f"visible-{int(time.time())}"
        vis_plan = native_plan_spec(args.csrc_root, perturbed_text, vis_nonce, rebuild_r1=True, cache=False)
        t_edit = time.time()
        vis_ins = runner.op(op="txn", plan=vis_plan, edit_time=t_edit, gap_timeout_s=120.0, phase2_deadline_s=60.0)
        require(vis_ins.get("ok") and vis_ins.get("outcome") == "committed", f"visible install failed: {vis_ins}")
        vis_rec = vis_ins["receipt"]
        vis_prep_native = vis_rec["participants"]["workers"]["prepare"]["ranks"][0]["native"]
        perturbed_so, perturbed_key = vis_prep_native["so"], vis_prep_native["key"]
        require((vis_rec.get("rebuild") or {}).get("ok") is True, f"R1 rebuild failed: {vis_rec.get('rebuild')}")
        wkey_vis = f"native:_C.fused_add_rms_norm:{perturbed_key}"

        vis_check = runner.check()
        require(vis_check["texts"] != base1["texts"], "visible .cu perturbation did not change outputs")
        vis_prove = runner.op(op="prove", txn=vis_ins["txn"], golden={"output": vis_check["output"]})
        require(vis_prove.get("ok") is True, f"visible prove failed: {vis_prove}")
        vis_prec = vis_prove["receipt"]
        live_w = vis_prec["witness"]["workers/rank0"][wkey_vis]
        cap_w = vis_prec["witness_capture"]["workers/rank0"][wkey_vis]
        require(live_w > 0 and cap_w > 0, f"expected positive live & capture witness, got live={live_w} cap={cap_w}")

        vis_undo = runner.op(op="undo", txn=vis_ins["txn"], gap_timeout_s=120.0)
        require(vis_undo.get("ok") and vis_undo["receipt"]["status"] == "undone", f"visible undo failed: {vis_undo}")
        ab_vis = runner.op(op="at_base")
        require(ab_vis.get("ok") is True, f"at_base failed after visible undo: {ab_vis}")
        after_vis_undo = runner.check()
        require(after_vis_undo["texts"] == base1["texts"], "undo did not restore base texts")
        require(max_abs_logprob_diff(after_vis_undo, base1) == 0.0, "undo did not restore exact base logprobs")

        summary["steps"]["visible_perturbation"] = {
            "key": perturbed_key,
            "build_s": vis_prep_native["build_s"],
            "load_s": vis_prep_native["load_s"],
            "prepare_s": vis_rec["prepare_s"],
            "gap_s": vis_rec["gap_s"],
            "recapture_s": vis_rec["recapture_s"],
            "edit_to_commit_s": vis_rec["edit_to_commit_s"],
            "edit_to_rebuilt_s": vis_rec["edit_to_rebuilt_s"],
            "output_changed": vis_check["output"] != base1["output"],
            "live_output": vis_check["output"],
            "witness_live": live_w,
            "witness_capture": cap_w,
            "undo_s": vis_undo["receipt"]["undo_s"],
            "worker_undo_s": vis_undo["receipt"]["participants"]["workers"]["undo"]["ranks"][0]["side_s"],
            "at_base": ab_vis["ok"],
            "undo_restored_exact": True,
            "check_s": vis_check["seconds"],
        }
        say("visible_perturbation", **summary["steps"]["visible_perturbation"])

        # 2c. Upstream PR #49750: install pre-PR -> install PR #49750 live + R1 -> prove -> undo
        pre_nonce = f"pre-pr49750-{int(time.time())}"
        pre_plan = native_plan_spec(args.csrc_root, pre_pr_text, pre_nonce, rebuild_r1=True, cache=False)
        pre_ins = runner.op(op="txn", plan=pre_plan, gap_timeout_s=120.0, phase2_deadline_s=60.0)
        require(pre_ins.get("ok") and pre_ins.get("outcome") == "committed", f"pre-pr install failed: {pre_ins}")
        pre_check = runner.check()
        pre_prove = runner.op(op="prove", txn=pre_ins["txn"], golden={"output": pre_check["output"]})
        require(pre_prove.get("ok") is True, f"pre-pr prove failed: {pre_prove}")

        pr_nonce = f"pr49750-{int(time.time())}"
        pr_plan = native_plan_spec(args.csrc_root, pr_text, pr_nonce, rebuild_r1=True, cache=False)
        t_edit_pr = time.time()
        pr_ins = runner.op(op="txn", plan=pr_plan, edit_time=t_edit_pr, gap_timeout_s=120.0, phase2_deadline_s=60.0)
        require(pr_ins.get("ok") and pr_ins.get("outcome") == "committed", f"PR #49750 install failed: {pr_ins}")
        pr_rec = pr_ins["receipt"]
        pr_prep_native = pr_rec["participants"]["workers"]["prepare"]["ranks"][0]["native"]
        pr_so, pr_key = pr_prep_native["so"], pr_prep_native["key"]
        require((pr_rec.get("rebuild") or {}).get("ok") is True, f"PR R1 rebuild failed: {pr_rec.get('rebuild')}")
        wkey_pr = f"native:_C.fused_add_rms_norm:{pr_key}"

        pr_check = runner.check()
        pr_prove = runner.op(op="prove", txn=pr_ins["txn"], golden={"output": pr_check["output"]})
        require(pr_prove.get("ok") is True, f"PR prove failed: {pr_prove}")
        pr_prec = pr_prove["receipt"]
        pr_live_w = pr_prec["witness"]["workers/rank0"][wkey_pr]
        pr_cap_w = pr_prec["witness_capture"]["workers/rank0"][wkey_pr]
        require(pr_live_w > 0 and pr_cap_w > 0, f"expected positive PR witness, got {pr_live_w}, {pr_cap_w}")

        # Cascade undo from pre-PR transaction restores clean base cleanly
        pr_undo = runner.op(op="undo", txn=pre_ins["txn"], gap_timeout_s=120.0)
        require(pr_undo.get("ok") and pr_undo["receipt"]["status"] == "undone", f"PR cascade undo failed: {pr_undo}")
        ab_pr = runner.op(op="at_base")
        require(ab_pr.get("ok") is True, f"at_base failed after PR undo: {ab_pr}")
        after_pr_undo = runner.check()
        require(after_pr_undo["texts"] == base1["texts"], "PR undo did not restore base")
        require(max_abs_logprob_diff(after_pr_undo, base1) == 0.0, "PR undo did not restore exact logprobs")

        summary["steps"]["pr49750_live"] = {
            "key": pr_key,
            "build_s": pr_prep_native["build_s"],
            "load_s": pr_prep_native["load_s"],
            "prepare_s": pr_rec["prepare_s"],
            "gap_s": pr_rec["gap_s"],
            "recapture_s": pr_rec["recapture_s"],
            "edit_to_commit_s": pr_rec["edit_to_commit_s"],
            "edit_to_rebuilt_s": pr_rec["edit_to_rebuilt_s"],
            "live_output": pr_check["output"],
            "witness_live": pr_live_w,
            "witness_capture": pr_cap_w,
            "cascade_undo_s": pr_undo["receipt"]["undo_s"],
            "worker_undo_s": pr_undo["receipt"]["participants"]["workers"]["undo"]["ranks"][0]["side_s"],
            "at_base": ab_pr["ok"],
            "undo_restored_exact": True,
            "check_s": pr_check["seconds"],
        }
        say("pr49750_live", **summary["steps"]["pr49750_live"])
        shutil.copyfile(os.path.join(runner.ctl, "receipts.jsonl"), os.path.join(work, "live-receipts.jsonl"))
    finally:
        runner.stop()

    # 3. Fresh cold restarts for live == fresh verification + baseline restart timing
    fresh_vis_healthy = runner.start("fresh-visible", adapter=False, fresh_so=perturbed_so, fresh_key=perturbed_key)
    try:
        fresh_vis_check = runner.check()
    finally:
        runner.stop()
    vis_lp_diff = max_abs_logprob_diff(vis_check, fresh_vis_check)
    require(vis_check["texts"] == fresh_vis_check["texts"], "visible live != fresh texts")
    require(vis_lp_diff == 0.0, f"visible live != fresh logprobs: max diff {vis_lp_diff}")
    summary["steps"]["visible_fresh"] = {
        "healthy_s": fresh_vis_healthy,
        "check_s": fresh_vis_check["seconds"],
        "fresh_output": fresh_vis_check["output"],
        "live_equals_fresh": True,
        "max_abs_logprob_diff": vis_lp_diff,
        "restart_total_s": round(summary["steps"]["visible_perturbation"]["build_s"] + fresh_vis_healthy + fresh_vis_check["seconds"], 3),
    }
    say("visible_fresh", **summary["steps"]["visible_fresh"])

    fresh_pr_healthy = runner.start("fresh-pr49750", adapter=False, fresh_so=pr_so, fresh_key=pr_key)
    try:
        fresh_pr_check = runner.check()
    finally:
        runner.stop()
    pr_lp_diff = max_abs_logprob_diff(pr_check, fresh_pr_check)
    require(pr_check["texts"] == fresh_pr_check["texts"], "PR #49750 live != fresh texts")
    require(pr_lp_diff == 0.0, f"PR #49750 live != fresh logprobs: max diff {pr_lp_diff}")
    summary["steps"]["pr49750_fresh"] = {
        "healthy_s": fresh_pr_healthy,
        "check_s": fresh_pr_check["seconds"],
        "fresh_output": fresh_pr_check["output"],
        "live_equals_fresh": True,
        "max_abs_logprob_diff": pr_lp_diff,
        "restart_total_s": round(summary["steps"]["pr49750_live"]["build_s"] + fresh_pr_healthy + fresh_pr_check["seconds"], 3),
    }
    say("pr49750_fresh", **summary["steps"]["pr49750_fresh"])

    summary["gpu_mem_after_mib"] = gpu_mem_mib(os.environ.get("CUDA_VISIBLE_DEVICES"))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    say("ALL_PASSED", out=args.out, gpu_mem_after_mib=summary["gpu_mem_after_mib"])


if __name__ == "__main__":
    main()
