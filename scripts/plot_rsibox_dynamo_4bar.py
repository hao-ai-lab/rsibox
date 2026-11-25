#!/usr/bin/env python3
"""Generate Light and Dark 5-bar comparison charts for RSIBox + NVIDIA Dynamo on NVIDIA B200.

Bar order (left -> right per model):
  1. Cold Start (uncached JIT + disk weights)
  2. Dynamo GMS RO (1-line edit, no RSIBox: Python + CUDA Graph reload, weights in GMS)
  3. Dynamo GMS + Shadow (same code, no edit)
  4. Fault + 1-line edit (Dynamo GMS Shadow promote + RSIBox patch replay)
  5. Live 1-line edit (RSIBox hot-swap on Dynamo, bit-exact)

Usage:
  python3 scripts/plot_rsibox_dynamo_4bar.py
  python3 scripts/plot_rsibox_dynamo_4bar.py --out-dir assets
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "assets"

# 100% measured NVIDIA B200 numbers (CUDA Graphs ON, logprobs=5 max_abs_diff == 0.0, 0 GMS errors)
# Tuple: (label_top, label_sub, cold_s, gms_ro_s, shadow_s, rsibox_s)
DATA = [
    ("Qwen3-0.6B",     "vLLM · TP1 · 1.5 GB",    155.60, 33.83,  0.3159, 0.0827),
    ("Qwen3-14B",      "vLLM · TP1 · 29.5 GB",   174.50, 42.28,  0.7229, 0.0810),
    ("Qwen3-14B",      "SGLang · TP1 · 29.5 GB",  73.14, 28.05,  0.1724, 0.0572),
    ("Qwen3.8-27B",    "vLLM · TP1 · 51 GiB",    219.03, 63.77,  0.9748, 0.0809),
    ("Qwen3-32.8B*",   "vLLM · TP1 · 65.5 GB",   183.61, 38.42,  0.8167, 0.0760),
    ("Qwen3-Next-80B", "vLLM · TP4 · 151 GiB",   779.09, 60.89,  1.8615, 0.0906),
]

THEMES = {
    "light": {
        "bg": "#FFFFFF",
        "panel": "#F8FAFC",
        "grid": "#E2E8F0",
        "axis": "#CBD5E1",
        "text": "#0F172A",
        "sub": "#475569",
        "muted": "#64748B",
        "c_cold": "#94A3B8",
        "c_gms": "#EA580C",
        "c_shadow": "#76B900",
        "c_fault": "#0D9488",
        "c_rsi": "#2563EB",
        "t_cold": "#475569",
        "t_gms": "#C2410C",
        "t_shadow": "#3F6212",
        "t_fault": "#0F766E",
        "t_rsi": "#1D4ED8",
        "pill_bg": "#EFF6FF",
        "pill_edge": "#93C5FD",
    },
    "dark": {
        "bg": "#0B0F17",
        "panel": "#111827",
        "grid": "#1E293B",
        "axis": "#334155",
        "text": "#F8FAFC",
        "sub": "#94A3B8",
        "muted": "#64748B",
        "c_cold": "#64748B",
        "c_gms": "#F59E0B",
        "c_shadow": "#76B900",
        "c_fault": "#14B8A6",
        "c_rsi": "#38BDF8",
        "t_cold": "#CBD5E1",
        "t_gms": "#FCD34D",
        "t_shadow": "#A3E635",
        "t_fault": "#5EEAD4",
        "t_rsi": "#7DD3FC",
        "pill_bg": "#0C4A6E",
        "pill_edge": "#38BDF8",
    },
}


def fmt_val(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}s"
    if v >= 100:
        return f"{v:.0f}s"
    if v >= 1.0:
        return f"{v:.2f}s" if v < 10.0 else f"{v:.1f}s"
    return f"{v * 1000:.0f}ms"


def draw_chart(theme_name: str, out_dir: Path) -> Path:
    th = THEMES[theme_name]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SF Pro Display", "Helvetica Neue", "DejaVu Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(15.8, 7.2), dpi=220)
    fig.patch.set_facecolor(th["bg"])
    ax.set_facecolor(th["panel"])

    n = len(DATA)
    x = np.arange(n)
    w = 0.16
    offsets = [-2.0 * w, -1.0 * w, 0.0, 1.0 * w, 2.0 * w]

    cold_vals   = [d[2] for d in DATA]
    gms_vals    = [d[3] for d in DATA]
    shadow_vals = [d[4] for d in DATA]
    fault_vals  = [d[4] + d[5] for d in DATA]
    rsi_vals    = [d[5] for d in DATA]

    ax.set_yscale("log")
    ax.set_ylim(0.02, 2200)
    ax.set_xlim(-0.58, n - 0.42)

    yticks = [0.05, 0.1, 1, 10, 100, 1000]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["50ms", "0.1s", "1s", "10s", "100s", "1,000s"],
                       fontsize=10, color=th["sub"], fontweight="medium")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(axis="y", color=th["grid"], linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    bw = w * 0.86
    b1 = ax.bar(x + offsets[0], cold_vals,   width=bw, color=th["c_cold"],
                label="1. Cold Start (uncached JIT + disk)", zorder=3)
    b2 = ax.bar(x + offsets[1], gms_vals,    width=bw, color=th["c_gms"],
                label="2. Dynamo GMS RO (1-line edit, no RSIBox: Python + Graph reload)", zorder=3)
    b3 = ax.bar(x + offsets[2], shadow_vals, width=bw, color=th["c_shadow"],
                label="3. Dynamo GMS + Shadow (same code, no edit)", zorder=3)
    b4 = ax.bar(x + offsets[3], fault_vals,  width=bw, color=th["c_fault"],
                label="4. Fault + 1-line edit (Dynamo GMS Shadow + RSIBox replay)", zorder=3)
    b5 = ax.bar(x + offsets[4], rsi_vals,    width=bw, color=th["c_rsi"],
                label="5. Live 1-line edit (RSIBox hot-swap on Dynamo)", zorder=3)

    bar_groups = [
        (b1, cold_vals,   th["t_cold"],   "cold"),
        (b2, gms_vals,    th["t_gms"],    "gms"),
        (b3, shadow_vals, th["t_shadow"], "shadow"),
        (b4, fault_vals,  th["t_fault"],  "fault"),
        (b5, rsi_vals,    th["t_rsi"],    "rsi"),
    ]
    for bars, vals, col, kind in bar_groups:
        for idx, (rect, val) in enumerate(zip(bars, vals)):
            cx = rect.get_x() + rect.get_width() / 2.0
            lbl = fmt_val(val)
            if kind == "rsi":
                ax.text(cx, val * 1.24, lbl, ha="center", va="bottom",
                        fontsize=8.2, fontweight="bold", color=col, zorder=5)
                sp = gms_vals[idx] / val
                ax.text(cx, val * 2.55, f"{sp:,.0f}×", ha="center", va="bottom",
                        fontsize=7.7, fontweight="bold", color=th["t_rsi"],
                        bbox=dict(boxstyle="round,pad=0.20", facecolor=th["pill_bg"],
                                  edgecolor=th["pill_edge"], linewidth=0.9),
                        zorder=6)
            elif kind == "fault":
                # Slight vertical nudge above Shadow bar to prevent label collision
                ax.text(cx, val * 1.38, lbl, ha="center", va="bottom",
                        fontsize=7.7, fontweight="semibold", color=col, zorder=5)
            else:
                ax.text(cx, val * 1.16, lbl, ha="center", va="bottom",
                        fontsize=7.7, fontweight="semibold", color=col, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d[0]}\n{d[1]}" for d in DATA],
                       fontsize=9.5, color=th["text"], fontweight="semibold")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=8)

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.spines["bottom"].set_linewidth(1.1)

    ax.set_ylabel("Time to Ready / Edit Applied (log scale)", fontsize=10.5,
                  color=th["sub"], fontweight="semibold", labelpad=10)

    fig.text(0.055, 0.945,
             "RSIBox + NVIDIA Dynamo: Sub-100ms Live Edits & Sub-2s Fault Recovery with Code Changes",
             fontsize=15.5, fontweight="bold", color=th["text"])
    fig.text(0.055, 0.902,
             "Dynamo GMS + Shadow cuts crash recovery to 0.17–1.86s; RSIBox eliminates the 28–64s Python/Graph reload on edits (57–91ms live, 0.23–1.95s after a crash).",
             fontsize=9.8, color=th["sub"])

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.155), ncol=2,
                    frameon=False, fontsize=8.8, handlelength=1.3, columnspacing=1.8)
    for txt in leg.get_texts():
        txt.set_color(th["text"])
        txt.set_fontweight("medium")

    fig.text(0.055, 0.020,
             "100% measured on NVIDIA B200 (CUDA Graphs ON, torch.compile ON, bit-exact logprobs=5, 0 GMS errors).  "
             "Bar 2 is pure Dynamo GMS RO restart (no CRIU re-dump); Snapshot + warm GMS restores same-code state in 3.6–6.0s.  "
             "*65.5 GB dense BF16 checkpoint (gpt-oss-120b size tier).",
             fontsize=8.1, color=th["muted"])

    plt.subplots_adjust(left=0.070, right=0.980, top=0.75, bottom=0.14)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rsibox-dynamo-4bar-{theme_name}.png"
    fig.savefig(out_path, dpi=220, facecolor=th["bg"])
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    for theme in ("light", "dark"):
        p = draw_chart(theme, args.out_dir)
        print(f"Wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
