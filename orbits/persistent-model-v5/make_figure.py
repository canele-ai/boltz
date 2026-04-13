"""Generate results figure for persistent-model-v5 evaluation."""
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
    "persistent": "#4C72B0",
    "predict_only": "#55A868",
}

# --- Data from evaluation ---
complexes = ["small", "medium", "large"]

# Baseline (eval-v5, 200-step, 3 recycling, subprocess)
baseline_times = [42.8, 51.3, 66.6]  # per-complex wall time
baseline_plddts = [0.8345, 0.5095, 0.8070]

# Persistent model (12 ODE steps, 0 recycling, pickle load)
persistent_predict = [2.56, 6.15, 8.81]  # predict-only
persistent_total = [2.85, 6.49, 9.76]  # predict + process
persistent_plddts = [0.8854, 0.4767, 0.8222]

# Amortized wall times for different N
onetime_cost = 11.8  # model load + warmup
Ns = [1, 3, 5, 10, 20, 50, 100]
mean_per_complex = 6.36
baseline_mean = 53.57
amortized_times = [onetime_cost / n + mean_per_complex for n in Ns]
amortized_speedups = [baseline_mean / t for t in amortized_times]

# CA RMSD
ca_rmsds_seeds = [3.825, 2.616, 2.584]  # small_complex across 3 seeds
ca_rmsd_mean = np.mean(ca_rmsds_seeds)
ca_rmsd_std = np.std(ca_rmsds_seeds, ddof=1)

# Per-seed speedups (N=3)
seed_speedups = [5.11, 4.89, 5.69]
seed_labels = ["42", "123", "7"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- Panel (a): Per-complex wall time comparison ---
ax = axes[0, 0]
x = np.arange(len(complexes))
w = 0.25
bars1 = ax.bar(x - w, baseline_times, w, label="Baseline (200-step)", color=COLORS["baseline"])
bars2 = ax.bar(x, persistent_total, w, label="Persistent (predict+process)", color=COLORS["persistent"])
bars3 = ax.bar(x + w, persistent_predict, w, label="Persistent (predict only)", color=COLORS["predict_only"])
ax.set_ylabel("Wall time (s)")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Per-complex timing")
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# --- Panel (b): Amortization curve ---
ax = axes[0, 1]
ax.plot(Ns, amortized_speedups, "o-", color=COLORS["persistent"], label="Amortized speedup", markersize=6)
ax.axhline(y=baseline_mean / mean_per_complex, color=COLORS["predict_only"],
           linestyle="--", alpha=0.7, label=f"Asymptotic limit ({baseline_mean / mean_per_complex:.1f}x)")
ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", alpha=0.5, label="Baseline (1.0x)")
ax.set_xlabel("Number of complexes (N)")
ax.set_ylabel("Speedup vs baseline")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(fontsize=9)
ax.set_title("Speedup vs batch size")
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# --- Panel (c): pLDDT comparison ---
ax = axes[1, 0]
x = np.arange(len(complexes))
w = 0.3
bars1 = ax.bar(x - w/2, baseline_plddts, w, label="Baseline", color=COLORS["baseline"])
bars2 = ax.bar(x + w/2, persistent_plddts, w, label="Persistent (12 ODE / 0r)", color=COLORS["persistent"])
ax.set_ylabel("pLDDT")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_ylim(0.3, 1.0)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("Quality: pLDDT")
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# --- Panel (d): CA RMSD for small_complex across seeds ---
ax = axes[1, 1]
seed_x = np.arange(len(seed_labels))
bars = ax.bar(seed_x, ca_rmsds_seeds, 0.5, color=COLORS["persistent"], alpha=0.8)
ax.axhline(y=ca_rmsd_mean, color=COLORS["persistent"], linestyle="--", alpha=0.7,
           label=f"Mean: {ca_rmsd_mean:.2f} +/- {ca_rmsd_std:.2f} A")
ax.set_ylabel("CA RMSD (A)")
ax.set_xticks(seed_x)
ax.set_xticklabels([f"Seed {s}" for s in seed_labels])
ax.set_ylim(0, 5)
ax.set_title("Structural quality: small_complex CA RMSD")
ax.legend(fontsize=9)
ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontsize=14, fontweight="bold")
# Annotate: ground truth only available for this complex
ax.text(1, 0.3, "Only test case with\nPDB ground truth (1BRS)",
        fontsize=8, color="gray", ha="center", style="italic")

fig.suptitle("Persistent Model V5: Speedup with CA RMSD Validation", y=1.02,
             fontsize=14, fontweight="medium")

out_path = "orbits/persistent-model-v5/figures/results.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
