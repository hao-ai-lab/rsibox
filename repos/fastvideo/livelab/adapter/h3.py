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
    FASTRSI_H3_CHECK_TIMEOUT         seconds a check may take in the conformance lab (default 600,
                                     the lab's own; c8's injected hang outlasts it by 30 s, and a
                                     fresh_check gets it + 600 s). On a cold shared filesystem the
                                     first video after a respawn took 131 s and a fresh load more
                                     than 720 s (2026-11-12), so 120 s fails c3 and c8 there

`workload_env()` sets the workload's FastVideo flags (kb/workload.md) unless already set. Every
worker inherits them, so they must be set before the lab starts.
"""
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


def _make_h3(receipts):
    from adapter.lab import FastVideoLab
    return FastVideoLab(h3_config(), receipts=receipts, golden=os.environ.get("FASTRSI_H3_GOLDEN"),
                        check_timeout=float(os.environ.get("FASTRSI_H3_CHECK_TIMEOUT", "600")))


H3Lab = shared_factory(_make_h3)
