"""Generate comparison figure for recycle-warmstart orbit."""
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
    "recycle=0": "#4C72B0",
    "recycle=1": "#DD8452",
}

# Data from evaluation (3 seeds each)
test_cases = ["small_complex", "medium_complex", "large_complex"]

# pLDDT per seed
plddt_r0 = {
    "small_complex": [0.8594, 0.8876, 0.9079],
    "medium_complex": [0.4769, 0.4647, 0.4772],
    "large_complex": [0.8196, 0.8187, 0.8202],
}
plddt_r1 = {
    "small_complex": [0.8728, 0.8448, 0.8443],
    "medium_complex": [0.4935, 0.4833, 0.4871],
    "large_complex": [0.8123, 0.8153, 0.8154],
}
plddt_baseline = {
    "small_complex": 0.8345,
    "medium_complex": 0.5095,
    "large_complex": 0.8070,
}

# Predict-only times per seed
pred_r0 = {
    "small_complex": [2.93, 3.05, 3.52],
    "medium_complex": [6.13, 6.25, 7.63],
    "large_complex": [12.37, 11.95, 12.21],
}
pred_r1 = {
    "small_complex": [3.44, 3.45, 3.63],
    "medium_complex": [10.39, 9.61, 8.76],
    "large_complex": [18.89, 17.11, 16.73],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): pLDDT comparison
ax = axes[0]
x = np.arange(len(test_cases))
width = 0.3

r0_means = [np.mean(plddt_r0[tc]) for tc in test_cases]
r0_stds = [np.std(plddt_r0[tc]) for tc in test_cases]
r1_means = [np.mean(plddt_r1[tc]) for tc in test_cases]
r1_stds = [np.std(plddt_r1[tc]) for tc in test_cases]
bl_vals = [plddt_baseline[tc] for tc in test_cases]

ax.bar(x - width/2, r0_means, width, yerr=r0_stds, label="recycle=0",
       color=COLORS["recycle=0"], alpha=0.85, capsize=3)
ax.bar(x + width/2, r1_means, width, yerr=r1_stds, label="recycle=1",
       color=COLORS["recycle=1"], alpha=0.85, capsize=3)
ax.scatter(x, bl_vals, marker='_', color=COLORS["baseline"], s=200, linewidths=2,
           zorder=5, label="baseline (SDE-200, 3r)")

ax.set_xticks(x)
ax.set_xticklabels(["small", "medium", "large"], fontsize=10)
ax.set_ylabel("pLDDT")
ax.set_title("Quality: pLDDT by complex")
ax.legend(loc="lower left", fontsize=9)
ax.set_ylim(0.35, 0.95)
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (b): Predict-only time comparison
ax = axes[1]
r0_time_means = [np.mean(pred_r0[tc]) for tc in test_cases]
r0_time_stds = [np.std(pred_r0[tc]) for tc in test_cases]
r1_time_means = [np.mean(pred_r1[tc]) for tc in test_cases]
r1_time_stds = [np.std(pred_r1[tc]) for tc in test_cases]

ax.bar(x - width/2, r0_time_means, width, yerr=r0_time_stds, label="recycle=0",
       color=COLORS["recycle=0"], alpha=0.85, capsize=3)
ax.bar(x + width/2, r1_time_means, width, yerr=r1_time_stds, label="recycle=1",
       color=COLORS["recycle=1"], alpha=0.85, capsize=3)

# Annotate overhead
for i, tc in enumerate(test_cases):
    overhead = np.mean(pred_r1[tc]) - np.mean(pred_r0[tc])
    pct = overhead / np.mean(pred_r0[tc]) * 100
    ax.annotate(f"+{overhead:.1f}s\n(+{pct:.0f}%)",
                xy=(i + width/2, np.mean(pred_r1[tc]) + np.std(pred_r1[tc]) + 0.3),
                fontsize=8, ha="center", color=COLORS["recycle=1"])

ax.set_xticks(x)
ax.set_xticklabels(["small", "medium", "large"], fontsize=10)
ax.set_ylabel("Predict-only time (s)")
ax.set_title("Cost: predict-only time")
ax.legend(loc="upper left", fontsize=9)
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (c): Quality-cost scatter
ax = axes[2]
for tc in test_cases:
    for seed_idx, (p0, t0) in enumerate(zip(plddt_r0[tc], pred_r0[tc])):
        ax.scatter(t0, p0, color=COLORS["recycle=0"], marker="o", s=40, alpha=0.7,
                   label="recycle=0" if tc == test_cases[0] and seed_idx == 0 else "")
    for seed_idx, (p1, t1) in enumerate(zip(plddt_r1[tc], pred_r1[tc])):
        ax.scatter(t1, p1, color=COLORS["recycle=1"], marker="s", s=40, alpha=0.7,
                   label="recycle=1" if tc == test_cases[0] and seed_idx == 0 else "")

# Draw arrows from r0 mean to r1 mean for each test case
for tc in test_cases:
    r0m_t = np.mean(pred_r0[tc])
    r0m_p = np.mean(plddt_r0[tc])
    r1m_t = np.mean(pred_r1[tc])
    r1m_p = np.mean(plddt_r1[tc])
    ax.annotate("", xy=(r1m_t, r1m_p), xytext=(r0m_t, r0m_p),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

ax.set_xlabel("Predict-only time (s)")
ax.set_ylabel("pLDDT")
ax.set_title("Quality vs cost tradeoff")
ax.legend(loc="lower right", fontsize=9)
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.suptitle("Recycling steps=0 vs 1 with ODE-12 sampling (3 seeds each)",
             fontsize=13, fontweight="medium", y=1.02)

out_path = "/home/liambai/code/boltz/.worktrees/recycle-warmstart/orbits/recycle-warmstart/figures/recycle_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved to {out_path}")
