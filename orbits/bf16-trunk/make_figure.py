"""Create comparison figure: bf16 trunk vs fp32 trunk quality and timing."""

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

COLORS = {
    "baseline": "#888888",
    "ode_fp32": "#4C72B0",
    "ode_bf16": "#DD8452",
}

complexes = ["small", "medium", "large"]

# Baseline (200s/3r) from config.yaml
baseline_plddt = [0.8350, 0.4906, 0.8064]
baseline_time = [53.0, 70.5, 87.6]

# ODE-20/0r fp32 (from parent orbit, validated)
ode_fp32_plddt = [0.8860, 0.4888, 0.8161]
ode_fp32_time = [33.2, 40.0, 44.8]

# ODE-20/0r bf16 tri_mult (this orbit, validated 3 runs)
ode_bf16_plddt = [0.8860, 0.4888, 0.8161]
ode_bf16_time = [36.4, 41.2, 48.8]  # median times from validation

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# --- Panel (a): Per-complex pLDDT ---
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

x = np.arange(len(complexes))
width = 0.25

bars1 = ax.bar(x - width, baseline_plddt, width, label="Baseline (200s/3r)",
               color=COLORS["baseline"], alpha=0.8)
bars2 = ax.bar(x, ode_fp32_plddt, width, label="ODE-20/0r fp32",
               color=COLORS["ode_fp32"], alpha=0.8)
bars3 = ax.bar(x + width, ode_bf16_plddt, width, label="ODE-20/0r bf16",
               color=COLORS["ode_bf16"], alpha=0.8)

ax.set_xlabel("Test complex")
ax.set_ylabel("pLDDT")
ax.set_title("Quality comparison")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_ylim(0.35, 1.0)
ax.legend(loc="lower right")

# Add baseline quality gate line
ax.axhline(y=np.mean(baseline_plddt) - 0.02, color="red", linestyle="--",
           alpha=0.5, linewidth=1)
ax.text(2.15, np.mean(baseline_plddt) - 0.02 - 0.02, "quality gate",
        color="red", fontsize=9, alpha=0.7, ha="right")

# --- Panel (b): Per-complex timing ---
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

bars1 = ax.bar(x - width, baseline_time, width, label="Baseline (200s/3r)",
               color=COLORS["baseline"], alpha=0.8)
bars2 = ax.bar(x, ode_fp32_time, width, label="ODE-20/0r fp32",
               color=COLORS["ode_fp32"], alpha=0.8)
bars3 = ax.bar(x + width, ode_bf16_time, width, label="ODE-20/0r bf16",
               color=COLORS["ode_bf16"], alpha=0.8)

ax.set_xlabel("Test complex")
ax.set_ylabel("Wall time (s)")
ax.set_title("Timing comparison")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper left")

# --- Panel (c): Summary - speedup vs quality delta ---
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

configs = {
    "Baseline\n(200s/3r)": (1.0, 0.0),
    "ODE-20/0r\nfp32 trunk": (1.79, 1.96),
    "ODE-20/0r\nbf16 trunk": (1.67, 1.96),
}

for name, (speedup, delta) in configs.items():
    color = COLORS["baseline"] if "Baseline" in name else (
        COLORS["ode_fp32"] if "fp32" in name else COLORS["ode_bf16"]
    )
    ax.scatter(speedup, delta, s=120, color=color, zorder=5, edgecolor="white", linewidth=1.5)
    offset_x = 0.03 if "bf16" in name else -0.03
    offset_y = -0.25 if "bf16" in name else 0.2
    ax.annotate(name, (speedup, delta),
                textcoords="offset points", xytext=(0, 20 if "Baseline" not in name else -25),
                ha="center", fontsize=9)

# Quality gate region
ax.axhspan(-2.0, 2.1, alpha=0.05, color="green")
ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
ax.text(0.5, -2.3, "quality gate (-2pp)", color="red", fontsize=9, alpha=0.7)

ax.set_xlabel("Speedup (x)")
ax.set_ylabel("pLDDT delta (pp)")
ax.set_title("Speedup vs quality")
ax.set_xlim(0.5, 2.2)
ax.set_ylim(-3.0, 3.5)

fig.suptitle("BF16 Trunk: Triangular Multiply Without FP32 Upcast", y=1.02)

out_path = Path(__file__).parent / "figures" / "bf16_trunk_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
