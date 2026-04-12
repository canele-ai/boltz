"""Generate comparison figure for torch-upgrade-kernels orbit.

Multi-panel figure showing:
(a) Per-complex timing comparison (200s/3r: kernels ON vs OFF)
(b) Per-complex timing comparison (ODE-20/0r: kernels ON vs OFF)
(c) Speedup bar chart for all configurations vs baseline
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style from research/style.md
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

# Color palette (seaborn muted)
COLORS = {
    "baseline": "#888888",
    "kernels_on": "#4C72B0",
    "kernels_off": "#DD8452",
    "ode_kernels_on": "#55A868",
    "ode_kernels_off": "#C44E52",
    "combined": "#8172B3",
}

complexes = ["small", "medium", "large"]
x = np.arange(len(complexes))
width = 0.35

# --- Data ---
# 200s/3r (torch 2.6)
baseline_kernels_on = {
    "times": [41.7, 49.2, 62.8],
    "plddts": [0.8345, 0.5095, 0.8070],
    "mean_time": 51.2,
    "mean_plddt": 0.7170,
    "speedup": 1.37,
}
baseline_kernels_off = {
    "times": [87.0, 56.0, 74.7],  # Note: small_complex has MSA contamination
    "plddts": [0.8350, 0.4743, 0.8044],
    "mean_time": 72.6,
    "mean_plddt": 0.7046,
    "speedup": 0.97,
}

# ODE-20/0r (torch 2.6)
ode_kernels_on = {
    "times": [49.5, 51.9, 57.7],
    "plddts": [0.8831, 0.4866, 0.8175],
    "mean_time": 53.0,
    "mean_plddt": 0.7291,
    "speedup": 1.33,
}
ode_kernels_off = {
    "times": [43.6, 50.5, 60.0],
    "plddts": [0.8861, 0.4883, 0.8179],
    "mean_time": 51.4,
    "mean_plddt": 0.7308,
    "speedup": 1.37,
}

# ODE-20/0r + kernels + TF32
ode_kernels_tf32 = {
    "times": [42.8, 40.5, 48.5],
    "plddts": [0.8831, 0.4868, 0.8180],
    "mean_time": 43.9,
    "mean_plddt": 0.7293,
    "speedup": 1.60,
}

# Reference: parent orbit ODE-20/0r (torch 2.5.1)
ode_parent = {
    "times": [33.2, 40.0, 44.8],
    "mean_time": 39.3,
    "mean_plddt": 0.7303,
    "speedup": 1.79,
}

# Original baseline (torch 2.5.1, 200s/3r)
original_baseline = {
    "times": [53.0, 70.5, 87.6],
    "mean_time": 70.37,
    "mean_plddt": 0.7107,
    "speedup": 1.00,
}


# --- Figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel (a): 200s/3r timing comparison
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
bars1 = ax.bar(x - width/2, baseline_kernels_on["times"], width,
               label="Kernels ON", color=COLORS["kernels_on"], alpha=0.85)
bars2 = ax.bar(x + width/2, baseline_kernels_off["times"], width,
               label="Kernels OFF", color=COLORS["kernels_off"], alpha=0.85)
ax.axhline(y=original_baseline["mean_time"], color=COLORS["baseline"],
           linestyle="--", linewidth=1, alpha=0.7, label=f"Baseline mean ({original_baseline['mean_time']:.1f}s)")
ax.set_xlabel("Test complex")
ax.set_ylabel("Wall time (s, median of 3)")
ax.set_title("200s / 3r: Kernel effect on trunk")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper left")
ax.set_ylim(0, 100)

# Add percentage labels
for i, (on, off) in enumerate(zip(baseline_kernels_on["times"], baseline_kernels_off["times"])):
    if off > 0:
        pct = (off - on) / off * 100
        if abs(pct) > 2:
            ax.annotate(f"{pct:+.0f}%", xy=(x[i], max(on, off) + 2),
                       fontsize=9, ha="center", color=COLORS["kernels_on"])

# Panel (b): ODE-20/0r timing comparison
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
bars3 = ax.bar(x - width/2, ode_kernels_on["times"], width,
               label="Kernels ON", color=COLORS["kernels_on"], alpha=0.85)
bars4 = ax.bar(x + width/2, ode_kernels_off["times"], width,
               label="Kernels OFF", color=COLORS["kernels_off"], alpha=0.85)
# Also show TF32 + kernels
ax.scatter(x, ode_kernels_tf32["times"], marker="D", s=60, zorder=5,
           color=COLORS["combined"], label="Kernels ON + TF32")
ax.axhline(y=ode_parent["mean_time"], color=COLORS["baseline"],
           linestyle="--", linewidth=1, alpha=0.7, label=f"ODE-20/0r best ({ode_parent['mean_time']:.1f}s)")
ax.set_xlabel("Test complex")
ax.set_ylabel("Wall time (s, median of 3)")
ax.set_title("ODE-20 / 0r: Kernel effect minimal")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, 80)

# Panel (c): Speedup summary bar chart
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

configs = [
    "Baseline (torch 2.5.1)",
    "200s/3r, no kernels (torch 2.6)",
    "200s/3r, kernels ON (torch 2.6)",
    "ODE-20/0r, no kernels (torch 2.6)",
    "ODE-20/0r, kernels ON (torch 2.6)",
    "ODE-20/0r, kernels+TF32 (torch 2.6)",
    "ODE-20/0r (torch 2.5.1, parent)",
]
speedups = [
    original_baseline["speedup"],
    baseline_kernels_off["speedup"],
    baseline_kernels_on["speedup"],
    ode_kernels_off["speedup"],
    ode_kernels_on["speedup"],
    ode_kernels_tf32["speedup"],
    ode_parent["speedup"],
]
colors = [
    COLORS["baseline"],
    COLORS["kernels_off"],
    COLORS["kernels_on"],
    COLORS["kernels_off"],
    COLORS["kernels_on"],
    COLORS["combined"],
    COLORS["baseline"],
]

bars = ax.barh(range(len(configs)), speedups, color=colors, alpha=0.85, height=0.7)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=9)
ax.set_xlabel("Speedup vs baseline (70.37s)")
ax.set_title("Overall speedup comparison")
ax.axvline(x=1.0, color="black", linewidth=0.5, alpha=0.3)

# Add speedup labels
for i, (bar, spd) in enumerate(zip(bars, speedups)):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{spd:.2f}x", va="center", fontsize=9, fontweight="medium")

ax.set_xlim(0, 2.1)
ax.invert_yaxis()

fig.suptitle("cuequivariance Kernel Impact on Boltz-2 Inference (L40S)",
             fontsize=14, fontweight="medium", y=1.02)

outpath = Path(__file__).parent / "figures" / "kernel_comparison.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {outpath}")
