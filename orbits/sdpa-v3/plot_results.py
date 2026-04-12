"""Generate comparison figure for SDPA vs control evaluation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
    "control": "#4C72B0",
    "sdpa": "#DD8452",
}

# Data (median of 3 runs, excluding first cold-start run)
complexes = ["small\n(~200 res)", "medium\n(~400 res)", "large\n(~600 res)"]
control_times = [32.2, 36.8, 40.4]
sdpa_times = [36.5, 39.7, 44.8]
baseline_times = [35.74, 47.08, 59.83]

control_plddt = [0.8831, 0.4868, 0.8180]
sdpa_plddt = [0.8817, 0.4857, 0.8183]
baseline_plddt = [0.8345, 0.5095, 0.8070]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

# Panel (a): Wall time comparison
ax = axes[0]
x = np.arange(len(complexes))
w = 0.25
ax.bar(x - w, baseline_times, w, color=COLORS["baseline"], label="Baseline (200-step)")
ax.bar(x, control_times, w, color=COLORS["control"], label="Control (ODE-20+TF32+bf16)")
ax.bar(x + w, sdpa_times, w, color=COLORS["sdpa"], label="SDPA (same + SDPA attn)")
ax.set_ylabel("Wall time (s)")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_title("Wall time per complex")
ax.legend(loc="upper left", fontsize=9)
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (b): Relative speedup
ax = axes[1]
control_speedups = [b / c for b, c in zip(baseline_times, control_times)]
sdpa_speedups = [b / s for b, s in zip(baseline_times, sdpa_times)]
ax.bar(x - 0.15, control_speedups, 0.3, color=COLORS["control"], label="Control")
ax.bar(x + 0.15, sdpa_speedups, 0.3, color=COLORS["sdpa"], label="SDPA")
ax.axhline(1.0, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.7)
ax.set_ylabel("Speedup (vs baseline)")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_title("Speedup factor")
ax.legend(loc="upper right", fontsize=9)
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (c): pLDDT comparison
ax = axes[2]
ax.bar(x - w, baseline_plddt, w, color=COLORS["baseline"], label="Baseline")
ax.bar(x, control_plddt, w, color=COLORS["control"], label="Control")
ax.bar(x + w, sdpa_plddt, w, color=COLORS["sdpa"], label="SDPA")
ax.set_ylabel("pLDDT")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_title("Quality (pLDDT)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0.3, 1.0)
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.suptitle(
    "SDPA Attention Replacement: No Speedup (0.90x vs Control)",
    fontsize=14, fontweight="medium", y=1.02,
)

out_path = Path(__file__).parent / "figures" / "sdpa_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
