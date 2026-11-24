#!/usr/bin/env python3
"""Generate Light and Dark 4-bar comparison charts for RSIBox + NVIDIA Dynamo on NVIDIA B200.

Bar order (left -> right per model):
  1. Cold Start (no Snapshot)
  2. Dynamo Snapshot (same code, no edit)
  3. Dynamo Snapshot + 1-line code edit (without RSIBox: warm/GMS restart + re-dump)
  4. Dynamo + RSIBox + 1-line code edit (live in-place swap, bit-exact)

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

# Exact audited NVIDIA B200 numbers (CUDA Graphs ON, logprobs=5 max_abs_diff == 0.0)
# Tuple: (label_top, label_sub, cold_s, snap_s, snap_is_ref, edit_snap_s, rsibox_s)
DATA = [
    ("Qwen3-0.6B",     "vLLM · TP1 · 1.5 GB",    155.60,  3.6213, False, 33.83 + 5.7545,  0.084713),
    ("Qwen3-14B",      "vLLM · TP1 · 29.5 GB",   174.50,  4.54,   False, 42.275 + 7.34,   0.0828),
    ("Qwen3-14B",      "SGLang · TP1 · 29.5 GB",  73.14,  4.18,   False, 28.05 + 6.11,    0.0579),
    ("Qwen3.8-27B",    "vLLM · TP1 · 52 GB",     219.03,  5.9571, False, 63.765 + 9.0823, 0.080945),
    ("65.5 GB Tier*",  "vLLM · TP1 · 65.5 GB",   183.61,  5.1584, False, 38.42 + 7.0297,  0.075993),
    ("DeepSeek-V4.1",  "vLLM · TP4 · 491 GiB",  1840.35, 30.96,   True,  296.393,         0.092029),
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
        "c_snap": "#76B900",
        "c_edit": "#EA580C",
        "c_rsi": "#2563EB",
        "t_cold": "#475569",
        "t_snap": "#3F6212",
        "t_edit": "#C2410C",
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
        "c_snap": "#76B900",
        "c_edit": "#F59E0B",
        "c_rsi": "#38BDF8",
        "t_cold": "#CBD5E1",
        "t_snap": "#A3E635",
        "t_edit": "#FCD34D",
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
        return f"{v:.1f}s"
    return f"{v * 1000:.0f}ms"


def draw_chart(theme_name: str, out_dir: Path) -> Path:
    th = THEMES[theme_name]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SF Pro Display", "Helvetica Neue", "DejaVu Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=220)
    fig.patch.set_facecolor(th["bg"])
    ax.set_facecolor(th["panel"])

    n = len(DATA)
    x = np.arange(n)
    w = 0.19
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    cold_vals = [d[2] for d in DATA]
    snap_vals = [d[3] for d in DATA]
    edit_vals = [d[5] for d in DATA]
    rsi_vals = [d[6] for d in DATA]

    ax.set_yscale("log")
    ax.set_ylim(0.02, 6500)
    ax.set_xlim(-0.58, n - 0.42)

    yticks = [0.05, 0.1, 1, 10, 100, 1000]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["50ms", "0.1s", "1s", "10s", "100s", "1,000s"],
                       fontsize=10, color=th["sub"], fontweight="medium")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(axis="y", color=th["grid"], linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    b1 = ax.bar(x + offsets[0], cold_vals, width=w * 0.88, color=th["c_cold"],
                label="1. Cold Start (no Snapshot)", zorder=3, edgecolor="none")
    b2 = ax.bar(x + offsets[1], snap_vals, width=w * 0.88, color=th["c_snap"],
                label="2. Dynamo Snapshot (same code, no edit)", zorder=3, edgecolor="none")
    b3 = ax.bar(x + offsets[2], edit_vals, width=w * 0.88, color=th["c_edit"],
                label="3. Dynamo Snapshot + 1-line edit (without RSIBox: restart + re-dump)", zorder=3, edgecolor="none")
    b4 = ax.bar(x + offsets[3], rsi_vals, width=w * 0.88, color=th["c_rsi"],
                label="4. Dynamo + RSIBox + 1-line edit (live hot-swap, bit-exact)", zorder=3, edgecolor="none")

    for idx, d in enumerate(DATA):
        if d[4]:
            b2[idx].set_hatch("//")
            b2[idx].set_edgecolor(th["bg"])
            b2[idx].set_linewidth(0.8)

    bar_groups = [
        (b1, cold_vals, th["t_cold"], False),
        (b2, snap_vals, th["t_snap"], False),
        (b3, edit_vals, th["t_edit"], False),
        (b4, rsi_vals, th["t_rsi"], True),
    ]
    for bars, vals, col, is_rsi in bar_groups:
        for idx, (rect, val) in enumerate(zip(bars, vals)):
            cx = rect.get_x() + rect.get_width() / 2.0
            lbl = fmt_val(val)
            if is_rsi:
                ax.text(cx, val * 1.24, lbl, ha="center", va="bottom",
                        fontsize=8.8, fontweight="bold", color=col, zorder=5)
                sp = edit_vals[idx] / val
                ax.text(cx, val * 2.55, f"{sp:,.0f}×", ha="center", va="bottom",
                        fontsize=8.2, fontweight="bold", color=th["t_rsi"],
                        bbox=dict(boxstyle="round,pad=0.22", facecolor=th["pill_bg"],
                                  edgecolor=th["pill_edge"], linewidth=0.9),
                        zorder=6)
            else:
                suffix = "†" if (bars is b2 and DATA[idx][4]) else ""
                ax.text(cx, val * 1.18, lbl + suffix, ha="center", va="bottom",
                        fontsize=8.3, fontweight="semibold", color=col, zorder=5)

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
             "Cold Start vs. Dynamo Snapshot vs. RSIBox Live Code Edit (NVIDIA B200)",
             fontsize=16.5, fontweight="bold", color=th["text"])
    fig.text(0.055, 0.902,
             "Dynamo Snapshot restores unchanged code in seconds; editing 1 line forces a restart + re-dump — RSIBox hot-swaps in 58–92 ms (badges = speedup vs. Bar 3).",
             fontsize=10.3, color=th["sub"])

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.125), ncol=2,
                    frameon=False, fontsize=9.6, handlelength=1.4, columnspacing=2.2)
    for txt in leg.get_texts():
        txt.set_color(th["text"])
        txt.set_fontweight("medium")

    fig.text(0.055, 0.022,
             "All runs on NVIDIA B200 with CUDA Graphs ON and bit-exact logprobs=5 verification.  "
             "*65.5 GB Tier uses a deterministic 32.8B BF16 checkpoint matching gpt-oss-120b size.  "
             "†DeepSeek-V4.1 Snapshot bar uses official Dynamo TEP8 ref (31.0s); Bar 3 excludes re-dump.",
             fontsize=8.2, color=th["muted"])

    plt.subplots_adjust(left=0.075, right=0.975, top=0.79, bottom=0.14)
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
