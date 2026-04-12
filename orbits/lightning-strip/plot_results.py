"""Generate results figure for lightning-strip orbit."""
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
    "parent": "#4C72B0",
    "nolightning_total": "#DD8452",
    "nolightning_gpu": "#55A868",
}

# Data from experiments
# Baseline (200-step, per-subprocess)
baseline_times = {"small": 42.81, "medium": 51.28, "large": 66.61}
baseline_mean = 53.57

# Parent orbit (ODE-20/0r+TF32+bf16, per-subprocess) - 1.34x
parent_times = {"small": 32.0, "medium": 38.0, "large": 48.0}  # approximate
parent_mean = baseline_mean / 1.34  # ~39.9s

# No-lightning single-load (v3 results, mean across 3 seeds)
nolightning_total = {"small": 12.9, "medium": 9.8, "large": 15.2}
nolightning_gpu = {"small": 7.7, "medium": 5.4, "large": 8.1}
nolightning_mean_total = 12.6
nolightning_mean_gpu = 7.0
model_load_time = 20.9

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel (a): Per-complex timing comparison
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
complexes = ["small", "medium", "large"]
x = np.arange(len(complexes))
width = 0.2

b1 = ax.bar(x - 1.5*width, [baseline_times[c] for c in complexes], width,
            label="Baseline (200-step)", color=COLORS["baseline"], alpha=0.8)
b2 = ax.bar(x - 0.5*width, [parent_mean]*3, width,
            label="Parent (ODE-20+TF32+bf16)", color=COLORS["parent"], alpha=0.8)
b3 = ax.bar(x + 0.5*width, [nolightning_total[c] for c in complexes], width,
            label="No-Lightning (total)", color=COLORS["nolightning_total"], alpha=0.8)
b4 = ax.bar(x + 1.5*width, [nolightning_gpu[c] for c in complexes], width,
            label="No-Lightning (GPU only)", color=COLORS["nolightning_gpu"], alpha=0.8)

ax.set_xlabel("Test complex")
ax.set_ylabel("Wall time (s)")
ax.set_title("Per-complex inference time")
ax.set_xticks(x)
ax.set_xticklabels(["Small\n(~200 res)", "Medium\n(~400 res)", "Large\n(~600 res)"])
ax.legend(loc="upper left", fontsize=9)

# Panel (b): Time breakdown (stacked bar)
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

categories = ["Baseline\n(per-case)", "Parent\n(per-case)", "No-Lightning\n(single load,\n3 cases)", "No-Lightning\n(single load,\n10 cases)"]

# Baseline: all in one block (per-case)
# Parent: model_load + inference (per-case), model load ~25s, inference ~15s
# No-lightning (3 cases): load_amort = 20.9/3, inference = 12.6
# No-lightning (10 cases): load_amort = 20.9/10, inference = 12.6

inference_times = [53.57, parent_mean, nolightning_mean_total, nolightning_mean_total]
load_amort = [0, 0, model_load_time / 3, model_load_time / 10]

x2 = np.arange(len(categories))
ax.bar(x2, inference_times, 0.5, label="Inference (incl. MSA)", color=COLORS["nolightning_total"], alpha=0.8)
ax.bar(x2, load_amort, 0.5, bottom=inference_times, label="Model load (amortized)", color="#C44E52", alpha=0.6)

for i, (inf, ld) in enumerate(zip(inference_times, load_amort)):
    total = inf + ld
    speedup = baseline_mean / total
    ax.annotate(f"{speedup:.1f}x", (i, total + 1), ha='center', fontsize=10, fontweight='medium')

ax.set_xlabel("")
ax.set_ylabel("Mean wall time (s)")
ax.set_title("Time breakdown: load vs inference")
ax.set_xticks(x2)
ax.set_xticklabels(categories, fontsize=9)
ax.legend(loc="upper right", fontsize=9)

# Panel (c): Speedup as function of batch size
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

n_cases = np.arange(1, 21)
speedup_nolightning = baseline_mean / (nolightning_mean_total + model_load_time / n_cases)
speedup_parent = np.full_like(n_cases, 1.34, dtype=float)

ax.plot(n_cases, speedup_nolightning, '-o', color=COLORS["nolightning_total"],
        label="No-Lightning (single load)", markersize=4, linewidth=2)
ax.axhline(y=1.34, color=COLORS["parent"], linestyle='--', linewidth=1.5,
           label="Parent orbit (per-case, 1.34x)")
ax.axhline(y=1.0, color=COLORS["baseline"], linestyle=':', linewidth=1,
           label="Baseline (1.0x)")

# Mark N=3 (our eval set)
ax.plot(3, baseline_mean / (nolightning_mean_total + model_load_time / 3),
        'D', color=COLORS["nolightning_total"], markersize=8, zorder=5)
ax.annotate(f"N=3: {baseline_mean / (nolightning_mean_total + model_load_time / 3):.1f}x",
            (3, baseline_mean / (nolightning_mean_total + model_load_time / 3)),
            textcoords="offset points", xytext=(10, -15), fontsize=9)

# Asymptotic speedup
ax.axhline(y=baseline_mean / nolightning_mean_total, color=COLORS["nolightning_gpu"],
           linestyle='-.', linewidth=1, alpha=0.5,
           label=f"Asymptotic ({baseline_mean / nolightning_mean_total:.1f}x)")

ax.set_xlabel("Number of predictions (batch size)")
ax.set_ylabel("Speedup vs baseline")
ax.set_title("Speedup vs number of predictions")
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 5)
ax.legend(loc="lower right", fontsize=9)

fig.suptitle("Lightning-stripped inference: eliminating per-case model loading overhead",
             fontsize=14, fontweight='medium')

outdir = Path(__file__).resolve().parent / "figures"
outdir.mkdir(exist_ok=True)
outpath = outdir / "results.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {outpath}")
