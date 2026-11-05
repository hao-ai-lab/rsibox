"""Stacked-bar figure of engine startup phases, one bar per start, from engine.py reports.

Usage: python3 plot_phases.py out.png NAME=report.json [NAME=report.json ...]
Prints the same numbers as a markdown table. Needs matplotlib.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = [  # (key in report, legend label) in fixed order; colors follow the order
    ("engine_init", "Import and config"), ("model_load", "Weights"),
    ("graphmem_profile", "Profiling run + first-use kernel compiles"), ("jit_warmup", "JIT warmup"),
    ("autotune", "FlashInfer autotune"), ("graph_capture", "CUDA graph capture"),
    ("ready", "Server ready"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
data = []
for name, path in runs:
    ph = json.load(open(path))["phases"]
    data.append((name, [max(0, (ph.get(k) or {}).get("seconds", 0) or 0) for k, _ in PHASES]))

fig, ax = plt.subplots(figsize=(10, 1.1 + 0.9 * len(data)), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
longest = max(sum(v) for _, v in data) or 1
for row, (name, vals) in enumerate(reversed(data)):
    left = 0
    for (k, label), v, c in zip(PHASES, vals, COLORS):
        if v <= 0:
            continue
        ax.barh(row, v, left=left, height=0.55, color=c, edgecolor=SURFACE, linewidth=2,
                label=label if row == len(data) - 1 else None)
        if v / longest > 0.03:       # direct label when the segment is wide enough
            ax.text(left + v / 2, row, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color=INK)
        left += v
    ax.text(left + longest * 0.01, row, f"{left:.0f} s", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(data)), [n for n, _ in reversed(data)], fontsize=10, color=INK)
ax.set_xlabel("seconds from first engine log line to server ready", color=MUTED)
ax.set_xlim(0, longest * 1.12)
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(colors=MUTED)
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURFACE)

print("| Start | " + " | ".join(l for _, l in PHASES) + " | Total |")
print("|---|" + "---:|" * (len(PHASES) + 1))
for name, vals in data:
    print(f"| {name} | " + " | ".join(f"{v:.0f}" for v in vals) + f" | {sum(vals):.0f} |")
