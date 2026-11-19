"""The H3 workload of kb/workload.md as a FastVideoLab config, and a conformance lab factory.

    h3_config(gpus)   the run config: FastVideo's FastH3-QAD-v2 NVFP4 example at the pin, with
                      FP4 attention off, num_gpus = sp_size = gpus, tp_size = 1, seed 1234
    H3Lab(receipts)   for `python3 -m conformance --lab adapter.h3:H3Lab`: one warm FastVideoLab
                      shared by every case (a load costs 60-150 s)

The machine-specific inputs come from the environment, nothing else:

    FASTH3_MODEL                     local snapshot of FastVideo/FastH3-QAD-v2 (required)
    FASTVIDEO_H3_VAE_NVFP4_DECODER   decoder-nvfp4.safetensors of FastVideo/FastH3-NVFP4-VAE-Decoder
                                     (fast-8L), required: the workload decodes with it
    H3_CONFIG                        the example yaml (default: the one next to the imported
                                     fastvideo package, examples/inference/basic/)
    FASTRSI_H3_GPUS                  ranks (default: the devices in CUDA_VISIBLE_DEVICES, else 1)
    FASTRSI_H3_GOLDEN                optional uint8 frames .npy: adds psnr_vs_golden to checks
    FASTRSI_H3_CHECK_TIMEOUT         seconds a check may take in the conformance lab (default 120
                                     when --zygote or --holder is enabled, else 600; c8's injected
                                     hang outlasts it by 30 s, and fresh_check gets at least 1200 s).
                                     On a cold shared filesystem without zygote/holder, the first
                                     video after a respawn took 131 s and a fresh load >720 s
    FASTRSI_H3_ZYGOTE                set to 1/true/yes to fork workers from a preloaded zygote
                                     (services/zygote.py) so respawns skip worker imports
    FASTRSI_H3_HOLDER                set to 1/true/yes to keep loaded weights in a CUDA-IPC holder
                                     process (services/holder.py; implies zygote)

`workload_env()` sets the workload's FastVideo flags (kb/workload.md) unless already set. Every
worker inherits them, so they must be set before the lab starts.
"""
import argparse
import atexit
import os

YAML = os.path.join("examples", "inference", "basic", "basic_fasth3_spark_qad_v2_nvfp4.yaml")

# The frozen workload flags (kb/workload.md): fused H3 helpers, BF16 VSA (no FP4 attention),
# the H3 VAE tile batch, Triton VSA. Paths are machine-specific and not set here.
WORKLOAD_ENV = {
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
    "FASTVIDEO_H3_VSA_FP4": "0",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "12",
    "FASTVIDEO_H3_VAE_INT8_OVERLAY": "0",
    "FASTVIDEO_FA4": "0",
    "FASTVIDEO_VSA_TRITON": "1",
    "FASTVIDEO_VSA_PTX_BLACKWELL": "0",
    "FASTVIDEO_STAGE_LOGGING": "1",
    "HF_HUB_OFFLINE": "1",
}


def workload_env():
    """Set the workload flags (unless set) and check the machine-specific paths. Returns the
    model path."""
    model = os.environ.get("FASTH3_MODEL")
    if not model or not os.path.isdir(model):
        raise SystemExit(f"FASTH3_MODEL must name a local FastH3-QAD-v2 snapshot; got {model!r}")
    dec = os.environ.get("FASTVIDEO_H3_VAE_NVFP4_DECODER")
    if not dec or not os.path.isfile(dec):
        raise SystemExit("FASTVIDEO_H3_VAE_NVFP4_DECODER must name decoder-nvfp4.safetensors; "
                         f"got {dec!r}")
    for k, v in WORKLOAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("FASTVIDEO_H3_ADALN_TABLE",
                          os.path.join(model, "transformer", "adaln_tables.pt"))
    return model


def default_gpus():
    if os.environ.get("FASTRSI_H3_GPUS"):
        return int(os.environ["FASTRSI_H3_GPUS"])
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return len([x for x in vis.split(",") if x.strip()]) or 1


def h3_config(gpus=None, model=None, yaml=None):
    """The workload's run config (`fastvideo generate --config` schema) for `gpus` ranks."""
    import fastvideo
    from fastvideo.api.parser import load_raw_config
    gpus = gpus or default_gpus()
    model = model or workload_env()
    yaml = yaml or os.environ.get("H3_CONFIG") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__))), YAML)
    cfg = load_raw_config(yaml)
    g = cfg["generator"]
    g["model_path"] = model
    g["engine"]["num_gpus"] = gpus
    g["engine"]["parallelism"].update(sp_size=gpus, tp_size=1)
    g["engine"].pop("execution_backend", None)
    g["pipeline"]["experimental"]["h3_vsa_fp4"]["enabled"] = False
    cfg["request"]["sampling"]["seed"] = 1234
    return cfg


class Shared:
    """One warm lab handed to every conformance case. `close` (the harness calls it after each
    case) puts the lab back at base for the next case instead of shutting it down: a poisoned
    lab is respawned, live edits are undone. The real shutdown runs at exit."""

    def __init__(self, lab):
        self.lab = lab

    def install(self, patch):
        return self.lab.install(patch)

    def check(self, workload=None):
        return self.lab.check(workload)

    def reset(self, kind="state"):
        return self.lab.reset(kind)

    def uninstall(self, id=None):
        return self.lab.uninstall(id)

    def status(self):
        return self.lab.status()

    def patches(self):
        return self.lab.patches()

    def fresh_check(self, patch, workload=None):
        return self.lab.fresh_check(patch, workload)

    def inject(self, fault, **kw):
        return self.lab.inject(fault, **kw)

    def __getattr__(self, k):
        return getattr(self.lab, k)

    def close(self):
        st = self.lab.status()
        if st.get("state") == "poisoned":
            self.lab.reset("respawn")
        elif st.get("txns"):
            self.lab.uninstall(st["txns"][0])


def shared_factory(make):
    """A `--lab` factory that builds the lab once (`make(receipts)`) and shares it."""
    box = {}

    def factory(receipts=None):
        if "lab" not in box:
            box["lab"] = make(receipts)
            atexit.register(box["lab"].close)
        return Shared(box["lab"])
    return factory


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _make_h3(receipts):
    from adapter.lab import FastVideoLab
    zygote = _env_flag("FASTRSI_H3_ZYGOTE")
    holder = _env_flag("FASTRSI_H3_HOLDER")
    raw_timeout = os.environ.get("FASTRSI_H3_CHECK_TIMEOUT")
    default_timeout = 120.0 if (zygote or holder) else 600.0
    return FastVideoLab(
        h3_config(),
        receipts=receipts,
        golden=os.environ.get("FASTRSI_H3_GOLDEN"),
        check_timeout=float(raw_timeout) if raw_timeout else default_timeout,
        zygote=zygote,
        holder=holder,
    )


class H3Lab(Shared):
    """Concrete `LabUnderTest` class sharing one warm H3 `FastVideoLab` across conformance cases."""

    _shared = {}

    def __init__(self, receipts=None):
        if "lab" not in self._shared:
            lab = _make_h3(receipts)
            self._shared["lab"] = lab
            atexit.register(lab.close)
        super().__init__(self._shared["lab"])

    @staticmethod
    def launch_environment(argv):
        """Translate `fastrsi conformance fastvideo --real ...` flags into H3 environment vars."""
        clean = list(argv)
        if clean[:1] == ["--"]:
            clean = clean[1:]
        ap = argparse.ArgumentParser(prog="fastrsi conformance fastvideo --real", allow_abbrev=False)
        ap.add_argument("--model", default=None)
        ap.add_argument("--vae", default=None)
        ap.add_argument("--config", default=None)
        ap.add_argument("--gpus", type=int, default=None)
        ap.add_argument("--golden", default=None)
        ap.add_argument("--check-timeout", type=float, default=None)
        ap.add_argument("--zygote", action="store_true")
        ap.add_argument("--holder", action="store_true")
        ns = ap.parse_args(clean)
        env = {}
        if ns.model:
            env["FASTH3_MODEL"] = os.path.abspath(ns.model)
        if ns.vae:
            env["FASTVIDEO_H3_VAE_NVFP4_DECODER"] = os.path.abspath(ns.vae)
        if ns.config:
            env["H3_CONFIG"] = os.path.abspath(ns.config)
        if ns.gpus is not None:
            env["FASTRSI_H3_GPUS"] = str(ns.gpus)
        if ns.golden:
            env["FASTRSI_H3_GOLDEN"] = os.path.abspath(ns.golden)
        if ns.check_timeout is not None:
            env["FASTRSI_H3_CHECK_TIMEOUT"] = str(ns.check_timeout)
        if ns.zygote:
            env["FASTRSI_H3_ZYGOTE"] = "1"
        if ns.holder:
            env["FASTRSI_H3_HOLDER"] = "1"
        return env
