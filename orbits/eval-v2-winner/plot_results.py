"""Generate comparison figure for eval-v2 stacked optimization sweep.

Two panels:
(a) Within-sweep comparison (same container, reliable relative ordering)
(b) Cross-container replication showing variance
"""

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
    "ODE-20/0r": "#4C72B0",
    "ODE-20/0r+TF32": "#55A868",
    "ODE-20/0r+bf16": "#8172B3",
    "ODE-20/0r+TF32+bf16": "#C44E52",
    "ODE-10/0r": "#DD8452",
    "ODE-10/0r+TF32+bf16": "#937860",
}

# =========================================================================
# Panel (a): Within-sweep comparison (same container)
# =========================================================================

configs_sweep = [
    "baseline (200s/3r)",
    "ODE-20/0r",
    "ODE-20/0r+TF32",
    "ODE-20/0r+bf16",
    "ODE-20/0r+TF32+bf16",
    "ODE-10/0r",
    "ODE-10/0r+TF32+bf16",
]

# Per-complex times (median of 3 runs within sweep)
small_times =  [42.8, 37.7, 39.4, 39.5, 32.5, 34.5, 35.1]
medium_times = [51.3, 40.2, 40.7, 42.2, 35.5, 38.2, 37.9]
large_times =  [66.6, 47.0, 44.8, 49.9, 40.5, 44.6, 44.1]
mean_times =   [53.57, 41.6, 41.6, 43.9, 36.2, 39.1, 39.0]
speedups =     [1.0, 1.29, 1.29, 1.22, 1.48, 1.37, 1.37]

color_list = [
    COLORS["baseline"],
    COLORS["ODE-20/0r"],
    COLORS["ODE-20/0r+TF32"],
    COLORS["ODE-20/0r+bf16"],
    COLORS["ODE-20/0r+TF32+bf16"],
    COLORS["ODE-10/0r"],
    COLORS["ODE-10/0r+TF32+bf16"],
]

# =========================================================================
# Panel (b): Cross-container replication of ODE-20/0r+TF32+bf16
# =========================================================================

# Three independent container runs of ODE-20/0r+TF32+bf16
replica_labels = ["Sweep\n(container 1)", "Replication\n(container 2)", "Replication\n(container 3)"]
replica_small = [32.5, 36.7, 36.8]
replica_medium = [35.5, 40.7, 40.9]
replica_large = [40.5, 48.4, 46.5]
replica_means = [36.2, 42.0, 41.4]
replica_speedups = [1.48, 1.28, 1.29]

# ODE-20/0r without TF32/bf16 cross-container
ode_only_labels = ["Sweep\n(container 1)", "Replication\n(container 4)"]
ode_only_means = [41.6, 53.5]
ode_only_speedups = [1.29, 1.00]

# =========================================================================
# Create figure
# =========================================================================

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

# Panel (a): Speedup bar chart
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.12, 1.05, "(a)", transform=ax_a.transAxes, fontsize=14, fontweight="bold")
bars = ax_a.barh(range(len(configs_sweep)), speedups, color=color_list, edgecolor="white", linewidth=0.5)
ax_a.set_yticks(range(len(configs_sweep)))
ax_a.set_yticklabels(configs_sweep, fontsize=9)
ax_a.set_xlabel("Speedup (x)")
ax_a.set_title("Speedup vs eval-v2 baseline (same container)")
ax_a.axvline(x=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.5)
ax_a.invert_yaxis()
for i, (s, t) in enumerate(zip(speedups, mean_times)):
    ax_a.text(s + 0.02, i, f"{s:.2f}x ({t:.0f}s)", va="center", fontsize=9)
ax_a.set_xlim(0, 1.8)

# Panel (b): Per-complex timing comparison
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.12, 1.05, "(b)", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
complex_names = ["Small\n(~200 res)", "Medium\n(~400 res)", "Large\n(~600 res)"]
x = np.arange(len(complex_names))
width = 0.12
for i, (cfg, s_t, m_t, l_t) in enumerate(zip(configs_sweep, small_times, medium_times, large_times)):
    times = [s_t, m_t, l_t]
    offset = (i - len(configs_sweep) / 2) * width + width / 2
    ax_b.bar(x + offset, times, width, color=color_list[i], edgecolor="white", linewidth=0.5,
             label=cfg)
ax_b.set_xticks(x)
ax_b.set_xticklabels(complex_names)
ax_b.set_ylabel("Wall time (s)")
ax_b.set_title("Per-complex timing (median of 3, same container)")
ax_b.legend(fontsize=7, loc="upper left", ncol=1)

# Panel (c): Cross-container replication of best config
ax_c = fig.add_subplot(gs[1, 0])
ax_c.text(-0.12, 1.05, "(c)", transform=ax_c.transAxes, fontsize=14, fontweight="bold")
x_rep = np.arange(len(replica_labels))
width_rep = 0.2
ax_c.bar(x_rep - width_rep, replica_small, width_rep, color="#4C72B0", label="Small", alpha=0.8)
ax_c.bar(x_rep, replica_medium, width_rep, color="#DD8452", label="Medium", alpha=0.8)
ax_c.bar(x_rep + width_rep, replica_large, width_rep, color="#55A868", label="Large", alpha=0.8)
# Mean line
for i, m in enumerate(replica_means):
    ax_c.plot([i - 0.3, i + 0.3], [m, m], "k-", linewidth=2, alpha=0.8)
    ax_c.text(i + 0.32, m, f"{m:.1f}s\n({replica_speedups[i]:.2f}x)", fontsize=9, va="center")
ax_c.set_xticks(x_rep)
ax_c.set_xticklabels(replica_labels)
ax_c.set_ylabel("Wall time (s)")
ax_c.set_title("Cross-container variance: ODE-20/0r+TF32+bf16")
ax_c.legend(fontsize=9)
ax_c.set_ylim(0, 60)

# Panel (d): Summary — honest range
ax_d = fig.add_subplot(gs[1, 1])
ax_d.text(-0.12, 1.05, "(d)", transform=ax_d.transAxes, fontsize=14, fontweight="bold")

# Bar chart comparing two configs across multiple containers
config_labels = ["ODE-20/0r\n(no TF32/bf16)", "ODE-20/0r\n+TF32+bf16"]
# ODE-20/0r: sweep=41.6, replication=53.5
# TF32+bf16: sweep=36.2, replication mean=41.7
ode_means = [47.6, 39.9]  # cross-container means
ode_stds = [8.4, 3.2]     # std from container spread
ode_best = [41.6, 36.2]
ode_worst = [53.5, 42.0]

x_d = np.arange(len(config_labels))
bars_d = ax_d.bar(x_d, ode_means, 0.5, color=[COLORS["ODE-20/0r"], COLORS["ODE-20/0r+TF32+bf16"]],
                  edgecolor="white", linewidth=0.5)
ax_d.errorbar(x_d, ode_means, yerr=ode_stds, fmt="none", ecolor="black", capsize=5, linewidth=1.5)

# Annotate best and worst
for i, (best, worst, mean) in enumerate(zip(ode_best, ode_worst, ode_means)):
    ax_d.text(i, mean + ode_stds[i] + 1, f"best: {best:.0f}s\nworst: {worst:.0f}s",
              ha="center", fontsize=8, color="#555555")

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(config_labels, fontsize=10)
ax_d.set_ylabel("Mean wall time (s)")
ax_d.set_title("Cross-container mean (honest range)")
# Add baseline reference
ax_d.axhline(y=53.57, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.5)
ax_d.text(1.5, 54.5, "baseline\n(53.6s)", fontsize=8, color=COLORS["baseline"], ha="center")
ax_d.set_ylim(0, 65)

fig.suptitle("eval-v2 Stacked Optimizations: ODE + TF32 + bf16 trunk\n"
             "L40S, torch 2.6.0, cuequivariance kernels enabled",
             fontsize=13, fontweight="medium", y=1.05)

fig.savefig(
    "/home/liambai/code/boltz/.worktrees/eval-v2-winner/orbits/eval-v2-winner/figures/stacked_comparison.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
plt.close(fig)
print("Saved stacked_comparison.png")
