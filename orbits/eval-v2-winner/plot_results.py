"""Generate comparison figure for eval-v2 stacked optimization sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

# Colors
COLORS = {
    "baseline": "#888888",
    "ODE-20/0r": "#4C72B0",
    "ODE-20/0r+TF32": "#55A868",
    "ODE-20/0r+bf16": "#8172B3",
    "ODE-20/0r+TF32+bf16": "#C44E52",
    "ODE-10/0r": "#DD8452",
    "ODE-10/0r+TF32+bf16": "#937860",
}

# Data from sweep (median of 3 runs per complex, mean across complexes)
configs = [
    "baseline\n(200s/3r)",
    "ODE-20/0r",
    "ODE-20/0r\n+TF32",
    "ODE-20/0r\n+bf16",
    "ODE-20/0r\n+TF32+bf16",
    "ODE-10/0r",
    "ODE-10/0r\n+TF32+bf16",
]

# Mean wall time (s) from sweep
mean_times = [53.57, 41.6, 41.6, 43.9, 36.2, 39.1, 39.0]

# Speedup = baseline / time
speedups = [1.0, 1.29, 1.29, 1.22, 1.48, 1.37, 1.37]

# pLDDT
plddts = [0.7170, 0.7293, 0.7293, 0.7293, 0.7293, 0.7301, 0.7301]

# pLDDT delta (pp)
plddt_deltas = [0.0, 1.23, 1.23, 1.23, 1.23, 1.31, 1.31]

# Per-complex timing (median of 3 runs)
per_complex_times = {
    "baseline\n(200s/3r)": {"small": 42.8, "medium": 51.3, "large": 66.6},
    "ODE-20/0r": {"small": 37.7, "medium": 40.2, "large": 47.0},
    "ODE-20/0r\n+TF32": {"small": 39.4, "medium": 40.7, "large": 44.8},
    "ODE-20/0r\n+bf16": {"small": 39.5, "medium": 42.2, "large": 49.9},
    "ODE-20/0r\n+TF32+bf16": {"small": 32.5, "medium": 35.5, "large": 40.5},
    "ODE-10/0r": {"small": 34.5, "medium": 38.2, "large": 44.6},
    "ODE-10/0r\n+TF32+bf16": {"small": 35.1, "medium": 37.9, "large": 44.1},
}

color_list = [
    COLORS["baseline"],
    COLORS["ODE-20/0r"],
    COLORS["ODE-20/0r+TF32"],
    COLORS["ODE-20/0r+bf16"],
    COLORS["ODE-20/0r+TF32+bf16"],
    COLORS["ODE-10/0r"],
    COLORS["ODE-10/0r+TF32+bf16"],
]

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel (a): Speedup bar chart
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
bars = ax.barh(range(len(configs)), speedups, color=color_list, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=9)
ax.set_xlabel("Speedup (x)")
ax.set_title("Speedup vs eval-v2 baseline")
ax.axvline(x=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.5)
ax.invert_yaxis()
# Add speedup labels
for i, (s, t) in enumerate(zip(speedups, mean_times)):
    ax.text(s + 0.02, i, f"{s:.2f}x ({t:.0f}s)", va="center", fontsize=9)
ax.set_xlim(0, 1.8)

# Panel (b): Per-complex wall time
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
complex_names = ["small", "medium", "large"]
x = np.arange(len(complex_names))
width = 0.12
for i, cfg in enumerate(configs):
    times = [per_complex_times[cfg][c] for c in complex_names]
    offset = (i - len(configs) / 2) * width + width / 2
    ax.bar(x + offset, times, width, color=color_list[i], edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(["Small\n(~200 res)", "Medium\n(~400 res)", "Large\n(~600 res)"])
ax.set_ylabel("Wall time (s)")
ax.set_title("Per-complex timing (median of 3 runs)")
# Simplified legend for panel b
for i, cfg in enumerate(configs):
    label = cfg.replace("\n", " ")
    ax.plot([], [], color=color_list[i], linewidth=6, label=label)
ax.legend(fontsize=7, loc="upper left")

# Panel (c): pLDDT comparison
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
bars = ax.barh(range(len(configs)), plddts, color=color_list, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=9)
ax.set_xlabel("Mean pLDDT")
ax.set_title("Quality (pLDDT)")
# Quality gate line
baseline_plddt = 0.7170
gate_plddt = baseline_plddt - 0.02
ax.axvline(x=gate_plddt, color="#C44E52", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(x=baseline_plddt, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.5)
ax.invert_yaxis()
# Labels
for i, (p, d) in enumerate(zip(plddts, plddt_deltas)):
    ax.text(p + 0.001, i, f"{p:.4f} ({d:+.1f}pp)", va="center", fontsize=8)
ax.set_xlim(0.68, 0.76)

fig.suptitle("eval-v2 Stacked Optimizations: ODE + TF32 + bf16 trunk\n"
             "(L40S, torch 2.6.0, cuequivariance kernels enabled)",
             fontsize=12, fontweight="medium", y=1.06)

fig.savefig(
    "/home/liambai/code/boltz/.worktrees/eval-v2-winner/orbits/eval-v2-winner/figures/stacked_comparison.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
plt.close(fig)
print("Saved stacked_comparison.png")
