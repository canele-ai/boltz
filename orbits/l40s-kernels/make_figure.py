"""Generate summary figure for L40S kernel investigation.

Multi-panel figure showing:
(a) Op-level TF32 speedup on isolated GPU operations
(b) End-to-end timing comparison showing MSA dominates
(c) Dependency conflict blocking cuequivariance kernels
"""
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
})

COLORS = {
    "baseline": "#888888",
    "highest": "#4C72B0",
    "tf32": "#DD8452",
    "bf16": "#55A868",
    "blocked": "#C44E52",
    "msa": "#8172B3",
}

fig = plt.figure(figsize=(18, 7), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.3])

# --- Panel (a): Op-level GPU speedup ---
ax = fig.add_subplot(gs[0])
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
ax.set_title("Triangular multiply speedup\n(L40S, seq_len=400)")

# From profiling data at N=400: tri_mul times in ms
categories = ["fp32\nhighest", "fp32\nTF32", "bf16"]
times_ms = [1.57, 1.32, 0.81]
speedups = [1.0, 1.57/1.32, 1.57/0.81]
colors_bar = [COLORS["highest"], COLORS["tf32"], COLORS["bf16"]]

bars = ax.bar(range(len(categories)), speedups, color=colors_bar, width=0.55,
              edgecolor="white", linewidth=1.5)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Relative speedup")
ax.set_ylim(0, 2.5)

for bar, sp, t in zip(bars, speedups, times_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
            f"{sp:.2f}x\n({t:.2f} ms)", ha="center", va="bottom", fontsize=9)

ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.5)

# --- Panel (b): End-to-end timing stacked bar ---
ax = fig.add_subplot(gs[1])
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
ax.set_title("End-to-end wall time (incl. MSA)")

# Data from validated runs (median across 3 runs)
configs = ["Baseline\n200s / 3r", "Best config\n20s / 0r"]
mean_times = [70.37, 41.10]

# Per-complex breakdown for 20s/0r (median times):
# small: 36.3s, medium: 39.7s, large: 47.3s (mean = 41.1s)
# Estimated GPU time from profiling: trunk ~20s + diffusion ~5s = 25s
# MSA + overhead: ~16s
gpu_est = [45, 25]
other_est = [t - g for t, g in zip(mean_times, gpu_est)]

x = np.arange(len(configs))
bar_width = 0.45

bars_gpu = ax.bar(x, gpu_est, bar_width, label="GPU compute (est.)",
                  color=COLORS["highest"], alpha=0.85)
bars_other = ax.bar(x, other_est, bar_width, bottom=gpu_est,
                    label="MSA + I/O overhead (est.)", color=COLORS["msa"], alpha=0.4)

ax.set_ylabel("Wall time (s)")
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 85)

speedups_e2e = [1.0, 70.37 / 41.10]
for i, (t, sp) in enumerate(zip(mean_times, speedups_e2e)):
    ax.text(i, t + 1.5, f"{t:.1f}s  ({sp:.2f}x)", ha="center", va="bottom",
            fontsize=10, fontweight="medium")

# Annotation showing TF32 would only affect GPU portion
ax.annotate("TF32: ~10% of this\n(~2.5s savings)",
            xy=(1, 12.5), fontsize=8, color=COLORS["tf32"], ha="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["tf32"], lw=1.0),
            xytext=(1.6, 5))

# --- Panel (c): Kernel blocker summary ---
ax = fig.add_subplot(gs[2])
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
ax.set_title("Optimization path status")
ax.axis("off")
ax.grid(False)

# Summary table as text
items = [
    ("cuequivariance kernels", "BLOCKED", COLORS["blocked"],
     "cublas-cu12: torch needs 12.4, cuequiv needs 12.5+"),
    ("torch.compile (pairformer)", "DISABLED", COLORS["blocked"],
     "Boltz reverts to uncompiled during inference"),
    ("TF32 matmul precision", "WORKS", COLORS["bf16"],
     "~10% GPU speedup, 0pp quality loss"),
    ("bf16 triangular multiply", "POTENTIAL", COLORS["tf32"],
     "~1.9x on isolated op, needs source change"),
]

y_start = 0.95
y_step = 0.22
for i, (name, status, color, detail) in enumerate(items):
    y = y_start - i * y_step
    ax.text(0.02, y, name, transform=ax.transAxes, fontsize=10.5, fontweight="medium",
            va="top")
    ax.text(0.7, y, status, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=color, va="top", ha="left")
    ax.text(0.02, y - 0.09, detail, transform=ax.transAxes, fontsize=8.5,
            color="#666666", va="top")

fig.suptitle("L40S Kernel Investigation: GPU optimizations masked by MSA latency",
             fontsize=14, fontweight="medium")

out_path = Path("orbits/l40s-kernels/figures/l40s_kernel_analysis.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved to {out_path}")
