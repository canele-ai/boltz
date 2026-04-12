"""Generate results figure for profile-and-fuse orbit."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

COLORS = {
    "baseline": "#888888",
    "parent": "#4C72B0",
    "ode20_full": "#DD8452",
    "ode10_full": "#55A868",
}

# Data from experiments
# Parent orbit (eval-v2-winner) cross-container results
parent_seeds = ["C1", "C2", "C3"]
parent_times = [36.2, 42.0, 41.4]
parent_speedups = [53.57/t for t in parent_times]

# ODE-20 + full stack (3 seeds)
ode20_seeds = [42, 123, 7]
ode20_times = [73.9, 73.5, 61.5]
ode20_speedups = [53.57/t for t in ode20_times]

# ODE-10 + full stack (3 seeds)
ode10_seeds = [42, 123, 7]
ode10_times = [62.5, 55.3, 78.3]
ode10_speedups = [53.57/t for t in ode10_times]

# Per-complex analysis for ODE-20 (same container comparisons)
# From within-container sweeps
configs = ["baseline\n(eval-v2-winner)", "SDPA+bf16\n(this orbit)"]
medium_times = [39.4, 42.0]  # representative within-container
large_times = [47.2, 44.5]   # representative within-container

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): Cross-container speedup comparison
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

x = np.arange(3)
w = 0.25
bars1 = ax.bar(x - w, parent_speedups, w, label="Parent orbit (ODE-20+TF32+bf16)",
               color=COLORS["parent"], alpha=0.8)
bars2 = ax.bar(x, ode20_speedups, w, label="This orbit (ODE-20+full stack)",
               color=COLORS["ode20_full"], alpha=0.8)
bars3 = ax.bar(x + w, ode10_speedups, w, label="This orbit (ODE-10+full stack)",
               color=COLORS["ode10_full"], alpha=0.8)

ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1.0, label="Baseline (1.0x)")
ax.set_xlabel("Independent container run")
ax.set_ylabel("Speedup (T_baseline / T_optimized)")
ax.set_title("Cross-container speedup variance")
ax.set_xticks(x)
ax.set_xticklabels(["Run 1", "Run 2", "Run 3"])
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 1.8)

# Panel (b): Within-container per-complex timing
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Data from within-container sweep (run 3 of the sweep comparison)
complexes = ["Medium\n(~400 res)", "Large\n(~600 res)"]
baseline_stacked = [39.4, 47.2]  # ODE-20+TF32+bf16 (parent config)
full_stack = [42.3, 44.5]  # ODE-20+SDPA+bf16_opm+TF32fix

x = np.arange(2)
w = 0.3
ax.bar(x - w/2, baseline_stacked, w, label="Parent config",
       color=COLORS["parent"], alpha=0.8)
ax.bar(x + w/2, full_stack, w, label="+ SDPA + bf16 OPM",
       color=COLORS["ode20_full"], alpha=0.8)

ax.set_xlabel("Complex size")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("Within-container comparison\n(same Modal run)")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper right", fontsize=9)

# Panel (c): Quality vs speedup scatter
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# All 3-seed results
parent_plddts = [0.7293, 0.7293, 0.7293]  # parent reported same across containers
ode20_plddts = [0.7335, 0.7272, 0.7367]
ode10_plddts = [0.7256, 0.7267, 0.7138]

ax.scatter(parent_speedups, [p*100 for p in parent_plddts],
           color=COLORS["parent"], s=60, label="Parent (ODE-20+TF32+bf16)", zorder=3)
ax.scatter(ode20_speedups, [p*100 for p in ode20_plddts],
           color=COLORS["ode20_full"], s=60, label="This (ODE-20+full stack)", zorder=3)
ax.scatter(ode10_speedups, [p*100 for p in ode10_plddts],
           color=COLORS["ode10_full"], s=60, label="This (ODE-10+full stack)", zorder=3)

# Quality gate
ax.axhline(y=71.70-2.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Quality floor (69.7)")
ax.axvline(x=1.0, color=COLORS["baseline"], linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xlabel("Speedup (higher = better)")
ax.set_ylabel("pLDDT (higher = better)")
ax.set_title("Quality vs speed tradeoff")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(0.5, 1.7)
ax.set_ylim(69, 75)

fig.suptitle("Profile-and-Fuse Orbit: SDPA + bf16 OPM + TF32 Fix", y=1.02)

out = "orbits/profile-and-fuse/figures/results.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out}")
