"""Generate timing comparison figure: with-MSA vs cached-MSA baseline."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "baseline_msa": "#888888",       # gray for old baseline
    "baseline_cached": "#4C72B0",    # blue for cached baseline
    "winner_msa": "#C44E52",         # red for old winner
    "winner_cached": "#DD8452",      # orange for cached winner
}

ORBIT_DIR = Path(__file__).parent
FIGURES_DIR = ORBIT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load cached results
with open(ORBIT_DIR / "baseline_cached_results.json") as f:
    baseline_cached = json.load(f)

with open(ORBIT_DIR / "winner_cached_results.json") as f:
    winner_cached = json.load(f)

# Old baseline numbers (from config.yaml / parent orbits)
old_baseline = {
    "mean_wall_time_s": 70.37,
    "per_complex": {
        "small_complex": {"wall_time_s": 53.0, "plddt": 0.8350},
        "medium_complex": {"wall_time_s": 70.5, "plddt": 0.4906},
        "large_complex": {"wall_time_s": 87.6, "plddt": 0.8064},
    },
    "mean_plddt": 0.7107,
}

# Old winner numbers (from ode-sampler orbit)
old_winner = {
    "mean_wall_time_s": 39.3,
    "per_complex": {
        "small_complex": {"wall_time_s": 33.2, "plddt": 0.8860},
        "medium_complex": {"wall_time_s": 40.0, "plddt": 0.4888},
        "large_complex": {"wall_time_s": 44.8, "plddt": 0.8161},
    },
    "mean_plddt": 0.7303,
}

# Extract cached per-complex data
complexes = ["small_complex", "medium_complex", "large_complex"]
complex_labels = ["Small\n(~200 res)", "Medium\n(~400 res)", "Large\n(~600 res)"]

cached_baseline_times = {}
cached_baseline_runs = {}
cached_winner_times = {}
cached_winner_runs = {}
cached_baseline_plddts = {}
cached_winner_plddts = {}

for pc in baseline_cached["per_complex"]:
    name = pc["name"]
    cached_baseline_times[name] = pc["wall_time_s"]
    cached_baseline_runs[name] = pc["run_times"]
    cached_baseline_plddts[name] = pc["quality"]["complex_plddt"]

for pc in winner_cached["per_complex"]:
    name = pc["name"]
    cached_winner_times[name] = pc["wall_time_s"]
    cached_winner_runs[name] = pc["run_times"]
    cached_winner_plddts[name] = pc["quality"]["complex_plddt"]


# --- Figure: 3-panel comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel (a): Per-complex wall time comparison
ax = axes[0]
x = np.arange(len(complexes))
width = 0.2

bars1 = ax.bar(x - 1.5*width, [old_baseline["per_complex"][c]["wall_time_s"] for c in complexes],
               width, label="Baseline (MSA server)", color=COLORS["baseline_msa"], alpha=0.7)
bars2 = ax.bar(x - 0.5*width, [cached_baseline_times[c] for c in complexes],
               width, label="Baseline (cached MSA)", color=COLORS["baseline_cached"])
bars3 = ax.bar(x + 0.5*width, [old_winner["per_complex"][c]["wall_time_s"] for c in complexes],
               width, label="ODE-20/0r (MSA server)", color=COLORS["winner_msa"], alpha=0.7)
bars4 = ax.bar(x + 1.5*width, [cached_winner_times[c] for c in complexes],
               width, label="ODE-20/0r (cached MSA)", color=COLORS["winner_cached"])

ax.set_xticks(x)
ax.set_xticklabels(complex_labels)
ax.set_ylabel("Wall time (s)")
ax.set_title("Per-complex wall time")
ax.legend(loc="upper left", fontsize=8)
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (b): Run-to-run timing variance
ax = axes[1]

# For each complex, show individual run times as points + box
positions = []
data_groups = []
colors_groups = []
labels_shown = set()

for i, c in enumerate(complexes):
    # Old baseline/winner had high variance but we only have median values
    # Show cached runs as scatter
    base_runs = cached_baseline_runs[c]
    win_runs = cached_winner_runs[c]

    # Plot individual runs
    jitter_base = np.random.RandomState(42).uniform(-0.1, 0.1, len(base_runs))
    jitter_win = np.random.RandomState(7).uniform(-0.1, 0.1, len(win_runs))

    lbl_b = "Baseline (cached)" if c == complexes[0] else None
    lbl_w = "ODE-20/0r (cached)" if c == complexes[0] else None

    ax.scatter(np.full(len(base_runs), i - 0.15) + jitter_base, base_runs,
               color=COLORS["baseline_cached"], s=40, zorder=5, label=lbl_b)
    ax.scatter(np.full(len(win_runs), i + 0.15) + jitter_win, win_runs,
               color=COLORS["winner_cached"], s=40, zorder=5, marker="^", label=lbl_w)

    # Mark median
    ax.hlines(np.median(base_runs), i - 0.3, i - 0.0, color=COLORS["baseline_cached"],
              linewidth=2, alpha=0.7)
    ax.hlines(np.median(win_runs), i + 0.0, i + 0.3, color=COLORS["winner_cached"],
              linewidth=2, alpha=0.7)

ax.set_xticks(range(len(complexes)))
ax.set_xticklabels(complex_labels)
ax.set_ylabel("Wall time (s)")
ax.set_title("Run-to-run variance (3 runs each)")
ax.legend(loc="upper left", fontsize=9)
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Annotate the warmup outlier
ax.annotate("model\nwarmup",
            xy=(0 - 0.15, max(cached_baseline_runs["small_complex"])),
            xytext=(0.5, max(cached_baseline_runs["small_complex"]) - 5),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))


# Panel (c): Speedup and timing summary
ax = axes[2]

configs = ["Baseline\n(200s/3r)", "ODE-20/0r"]
old_times = [old_baseline["mean_wall_time_s"], old_winner["mean_wall_time_s"]]
new_times = [baseline_cached["aggregate"]["mean_wall_time_s"],
             winner_cached["aggregate"]["mean_wall_time_s"]]

x = np.arange(len(configs))
width = 0.3

bars_old = ax.bar(x - width/2, old_times, width,
                  label="With MSA server", color=COLORS["baseline_msa"], alpha=0.7)
bars_new = ax.bar(x + width/2, new_times, width,
                  label="Cached MSA", color=COLORS["baseline_cached"])

# Add time labels on bars
for bar_group in [bars_old, bars_new]:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{height:.1f}s", ha="center", va="bottom", fontsize=9)

# Add speedup annotation
old_speedup = old_baseline["mean_wall_time_s"] / old_winner["mean_wall_time_s"]
new_speedup = new_times[0] / new_times[1]

ax.annotate(f"Old speedup: {old_speedup:.2f}x",
            xy=(1 - width/2, old_times[1]),
            xytext=(-0.1, 85),
            fontsize=9, color=COLORS["baseline_msa"],
            arrowprops=dict(arrowstyle="->", color=COLORS["baseline_msa"], lw=0.8))

ax.annotate(f"Cached speedup: {new_speedup:.2f}x",
            xy=(1 + width/2, new_times[1]),
            xytext=(0.2, 75),
            fontsize=9, color=COLORS["baseline_cached"],
            arrowprops=dict(arrowstyle="->", color=COLORS["baseline_cached"], lw=0.8))

ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Mean wall time (s)")
ax.set_title("Speedup comparison")
ax.legend(loc="upper right", fontsize=9)
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.suptitle("MSA Caching: Isolating GPU-Only Inference Time", fontsize=14, fontweight="medium", y=1.02)

out_path = FIGURES_DIR / "msa_cache_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
