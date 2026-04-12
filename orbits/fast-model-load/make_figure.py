"""Generate multi-panel figure for fast-model-load orbit results."""
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

COLORS = {
    "baseline": "#888888",
    "checkpoint": "#4C72B0",
    "pickle": "#DD8452",
    "pickle_warmup": "#55A868",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ---- Panel (a): Model Load Time Comparison ----
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

methods = ["load_from_\ncheckpoint", "torch.load\n(pickle)", "pickle +\nGPU transfer"]
load_times = [18.4, 1.6, 3.1]
colors = [COLORS["checkpoint"], COLORS["pickle"], COLORS["pickle_warmup"]]
bars = ax.bar(methods, load_times, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, load_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha='center', va='bottom', fontsize=10, fontweight='medium')
ax.set_ylabel("Time (seconds)")
ax.set_title("Model Load Time")
ax.set_ylim(0, 22)

# ---- Panel (b): Amortized Wall Time vs Number of Complexes ----
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

n_complexes = np.arange(1, 21)
baseline_wall = 53.57  # per complex, unchanged

# Checkpoint approach: one_time=21.7s (load) + 8.6s (warmup), per_complex=6.6s
one_time_ckpt = 21.7 + 8.6
per_complex = 6.6
amort_ckpt = one_time_ckpt / n_complexes + per_complex

# Pickle approach: one_time=3.1s (load) + 8.6s (warmup), per_complex=6.6s
one_time_pickle = 3.1 + 8.6
amort_pickle = one_time_pickle / n_complexes + per_complex

ax.axhline(baseline_wall, color=COLORS["baseline"], linestyle='--', linewidth=1.5, label="Baseline (SDE-200)")
ax.plot(n_complexes, amort_ckpt, '-o', color=COLORS["checkpoint"], markersize=4,
        label="Persistent (checkpoint)", linewidth=1.5)
ax.plot(n_complexes, amort_pickle, '-o', color=COLORS["pickle_warmup"], markersize=4,
        label="Persistent (pickle)", linewidth=1.5)
ax.axhline(per_complex, color=COLORS["pickle"], linestyle=':', linewidth=1, alpha=0.7,
           label=f"Predict-only ({per_complex}s)")
ax.set_xlabel("Number of Complexes")
ax.set_ylabel("Amortized Wall Time / Complex (s)")
ax.set_title("Amortization Curve")
ax.set_xlim(1, 20)
ax.set_ylim(0, 55)
ax.legend(loc='upper right', fontsize=9)

# ---- Panel (c): Speedup vs Number of Complexes ----
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

speedup_ckpt = baseline_wall / amort_ckpt
speedup_pickle = baseline_wall / amort_pickle

ax.plot(n_complexes, speedup_ckpt, '-o', color=COLORS["checkpoint"], markersize=4,
        label="Persistent (checkpoint)", linewidth=1.5)
ax.plot(n_complexes, speedup_pickle, '-o', color=COLORS["pickle_warmup"], markersize=4,
        label="Persistent (pickle)", linewidth=1.5)

# Mark the eval-v5 measurement point (3 complexes)
ax.plot(3, baseline_wall / (one_time_pickle / 3 + per_complex), 's',
        color=COLORS["pickle_warmup"], markersize=10, zorder=5)
ax.annotate(f"5.12x\n(eval-v5, n=3)", xy=(3, 5.12), xytext=(6, 5.5),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Reference lines
ax.axhline(3.5, color=COLORS["baseline"], linestyle='--', linewidth=1, alpha=0.5)
ax.text(19.5, 3.5, "bypass-lightning\n(3.5x, eval-v4)", ha='right', va='bottom',
        fontsize=8, color=COLORS["baseline"])

ax.set_xlabel("Number of Complexes")
ax.set_ylabel("Speedup (x)")
ax.set_title("Speedup vs Batch Size")
ax.set_xlim(1, 20)
ax.set_ylim(0, 9)
ax.legend(loc='lower right', fontsize=9)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "figures", "results.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Figure saved to {out_path}")
