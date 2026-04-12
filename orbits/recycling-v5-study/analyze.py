"""Analyze recycling sweep results and create figures."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Style
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

ORBIT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ORBIT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    "small_complex": "#4C72B0",
    "medium_complex": "#DD8452",
    "large_complex": "#55A868",
    "mean": "#C44E52",
    "baseline": "#888888",
}

TC_LABELS = {
    "small_complex": "1BRS (199 res)",
    "medium_complex": "1DQJ (563 res)",
    "large_complex": "2DN2 (574 res)",
}

# Baseline from config.yaml
BASELINE = {
    "mean_wall_time_s": 53.567,
    "mean_plddt": 0.7170,
    "per_complex": {
        "small_complex": {"wall_time_s": 42.806, "complex_plddt": 0.8345},
        "medium_complex": {"wall_time_s": 51.279, "complex_plddt": 0.5095},
        "large_complex": {"wall_time_s": 66.615, "complex_plddt": 0.8070},
    },
}

# Load results
with open(ORBIT_DIR / "raw_results.json") as f:
    raw = json.load(f)

recycling_steps = [0, 1, 2, 3]
test_cases = ["small_complex", "medium_complex", "large_complex"]
seeds = [42, 123, 7]

# Organize data
data = {}  # data[r_steps][tc_name] = {"predict_times": [...], "plddts": [...], ...}
for r in recycling_steps:
    data[r] = {}
    for tc in test_cases:
        runs = [x for x in raw if x["recycling_steps"] == r and x["tc_name"] == tc]
        data[r][tc] = {
            "predict_times": [x["predict_only_s"] for x in runs if x["predict_only_s"] is not None],
            "wall_times": [x["wall_time_s"] for x in runs if x["wall_time_s"] is not None],
            "plddts": [x["quality"]["complex_plddt"] for x in runs if "complex_plddt" in x.get("quality", {})],
            "iptms": [x["quality"]["iptm"] for x in runs if "iptm" in x.get("quality", {})],
        }

# Create multi-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): Predict-only time vs recycling steps (per complex)
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

for tc in test_cases:
    means = [np.mean(data[r][tc]["predict_times"]) for r in recycling_steps]
    stds = [np.std(data[r][tc]["predict_times"]) for r in recycling_steps]
    ax.errorbar(recycling_steps, means, yerr=stds, marker='o', markersize=6,
                color=COLORS[tc], label=TC_LABELS[tc], linewidth=2, capsize=3)

# Mean across all complexes
grand_means = []
grand_stds = []
for r in recycling_steps:
    all_times = []
    for tc in test_cases:
        all_times.extend(data[r][tc]["predict_times"])
    grand_means.append(np.mean(all_times))
    grand_stds.append(np.std(all_times))
ax.errorbar(recycling_steps, grand_means, yerr=grand_stds, marker='s', markersize=8,
            color=COLORS["mean"], label="Mean", linewidth=2.5, capsize=3, linestyle='--')

ax.set_xlabel("Recycling steps")
ax.set_ylabel("Predict-only time (s)")
ax.set_title("Inference time vs recycling")
ax.set_xticks(recycling_steps)
ax.legend(loc='upper left')

# Panel (b): pLDDT vs recycling steps (per complex)
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

for tc in test_cases:
    means = [np.mean(data[r][tc]["plddts"]) for r in recycling_steps]
    stds = [np.std(data[r][tc]["plddts"]) for r in recycling_steps]
    # Show baseline as horizontal dashed line
    bl = BASELINE["per_complex"][tc]["complex_plddt"]
    ax.axhline(y=bl, color=COLORS[tc], linestyle=':', alpha=0.4, linewidth=1)
    ax.errorbar(recycling_steps, means, yerr=stds, marker='o', markersize=6,
                color=COLORS[tc], label=TC_LABELS[tc], linewidth=2, capsize=3)

# Mean pLDDT
grand_plddt_means = []
for r in recycling_steps:
    tc_means = [np.mean(data[r][tc]["plddts"]) for tc in test_cases]
    grand_plddt_means.append(np.mean(tc_means))
ax.plot(recycling_steps, grand_plddt_means, marker='s', markersize=8,
        color=COLORS["mean"], label="Mean", linewidth=2.5, linestyle='--')

# Baseline mean
ax.axhline(y=BASELINE["mean_plddt"], color=COLORS["baseline"], linestyle='--', alpha=0.6, linewidth=1.5,
           label="Baseline mean")

ax.set_xlabel("Recycling steps")
ax.set_ylabel("pLDDT")
ax.set_title("Quality vs recycling")
ax.set_xticks(recycling_steps)
ax.legend(loc='lower right', fontsize=9)

# Panel (c): Marginal time cost per recycling step by complex
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

bar_width = 0.2
x_pos = np.arange(len(test_cases))

for i, r in enumerate(recycling_steps):
    times = [np.mean(data[r][tc]["predict_times"]) for tc in test_cases]
    ax.bar(x_pos + i * bar_width, times, bar_width, label=f"r={r}",
           color=plt.cm.viridis(i / 3), alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_xlabel("Test case")
ax.set_ylabel("Predict-only time (s)")
ax.set_title("Time breakdown by complex")
ax.set_xticks(x_pos + 1.5 * bar_width)
ax.set_xticklabels([TC_LABELS[tc] for tc in test_cases], fontsize=9)
ax.legend(title="Recycling", loc='upper left')

fig.suptitle("Recycling sweep: ODE-12 + bypass wrapper + TF32 + bf16", y=1.02, fontsize=14, fontweight='medium')

fig.savefig(FIGURES_DIR / "recycling_sweep.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ---- Print summary table ----
print("\n=== PREDICT-ONLY TIME (seconds, mean +/- std across 3 seeds) ===")
print(f"{'Recycling':>10}", end="")
for tc in test_cases:
    print(f"  {TC_LABELS[tc]:>18}", end="")
print(f"  {'Mean':>12}")
print("-" * 75)
for r in recycling_steps:
    print(f"{r:>10}", end="")
    tc_means = []
    for tc in test_cases:
        m = np.mean(data[r][tc]["predict_times"])
        s = np.std(data[r][tc]["predict_times"])
        tc_means.append(m)
        print(f"  {m:>8.1f} +/- {s:.1f}", end="")
    gm = np.mean(tc_means)
    print(f"  {gm:>8.1f}s")

print("\n=== pLDDT (mean across 3 seeds) ===")
print(f"{'Recycling':>10}", end="")
for tc in test_cases:
    print(f"  {TC_LABELS[tc]:>18}", end="")
print(f"  {'Mean':>12}")
print("-" * 75)

# Baseline row
print(f"{'baseline':>10}", end="")
for tc in test_cases:
    bl = BASELINE["per_complex"][tc]["complex_plddt"]
    print(f"  {bl:>18.4f}", end="")
print(f"  {BASELINE['mean_plddt']:>12.4f}")

for r in recycling_steps:
    print(f"{r:>10}", end="")
    tc_means = []
    for tc in test_cases:
        m = np.mean(data[r][tc]["plddts"])
        bl = BASELINE["per_complex"][tc]["complex_plddt"]
        delta = (m - bl) * 100
        tc_means.append(m)
        print(f"  {m:>10.4f} ({delta:+.1f}pp)", end="")
    gm = np.mean(tc_means)
    gd = (gm - BASELINE["mean_plddt"]) * 100
    print(f"  {gm:>7.4f} ({gd:+.1f}pp)")

# Marginal cost per recycling step
print("\n=== MARGINAL COST PER RECYCLING STEP (predict-only seconds) ===")
for r in [1, 2, 3]:
    costs = []
    for tc in test_cases:
        prev = np.mean(data[r-1][tc]["predict_times"])
        curr = np.mean(data[r][tc]["predict_times"])
        costs.append(curr - prev)
    print(f"  Step {r-1} -> {r}: {np.mean(costs):+.1f}s mean "
          f"({', '.join(f'{TC_LABELS[tc]}: {c:+.1f}s' for tc, c in zip(test_cases, costs))})")

print("\nFigure saved to:", FIGURES_DIR / "recycling_sweep.png")
