"""Generate quality-speed Pareto frontier figure from validation results.

Usage:
    python orbits/step-reduction/plot_pareto.py
"""
from __future__ import annotations

import json
from pathlib import Path

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
    "pass": "#55A868",
    "fail": "#C44E52",
    "pareto": "#4C72B0",
    "recycle_3": "#4C72B0",
    "recycle_1": "#DD8452",
    "recycle_0": "#55A868",
}

ORBIT_DIR = Path(__file__).parent
FIGURES_DIR = ORBIT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def make_combined_figure():
    """Three-panel figure using validated results.

    (a) Speedup vs configuration (bar chart, grouped by recycling)
    (b) Quality-Speed Pareto frontier
    (c) Per-complex pLDDT breakdown
    """

    # Validated results (3 runs each)
    configs = [
        {"label": "200s/3r\n(baseline)", "steps": 200, "recycle": 3,
         "time": 70.37, "plddt": 0.7107, "speedup": 1.00, "gate": True},
        # Phase 1: step sweep (1 run) - from sweep_results.json
        {"label": "100s/3r", "steps": 100, "recycle": 3,
         "time": 79.0, "plddt": 0.7050, "speedup": 0.89, "gate": True},
        {"label": "50s/3r", "steps": 50, "recycle": 3,
         "time": 71.7, "plddt": 0.7077, "speedup": 0.98, "gate": True},
        {"label": "20s/3r", "steps": 20, "recycle": 3,
         "time": 74.4, "plddt": 0.7062, "speedup": 0.95, "gate": True},
        {"label": "10s/3r", "steps": 10, "recycle": 3,
         "time": 84.5, "plddt": 0.4130, "speedup": 0.83, "gate": False},
        # Validated (3 runs each)
        {"label": "100s/0r", "steps": 100, "recycle": 0,
         "time": 41.5, "plddt": 0.7165, "speedup": 1.69, "gate": True},
        {"label": "50s/0r", "steps": 50, "recycle": 0,
         "time": 43.6, "plddt": 0.7350, "speedup": 1.61, "gate": True},
        {"label": "50s/1r", "steps": 50, "recycle": 1,
         "time": 107.4, "plddt": 0.7116, "speedup": 0.66, "gate": True},
        {"label": "20s/0r", "steps": 20, "recycle": 0,
         "time": 40.7, "plddt": 0.7263, "speedup": 1.73, "gate": True},
        {"label": "20s/1r", "steps": 20, "recycle": 1,
         "time": 47.6, "plddt": 0.7095, "speedup": 1.48, "gate": True},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # --- (a) Speedup vs recycling steps ---
    ax = axes[0]
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Speedup vs Configuration")

    # Group configs by key comparison: varying steps at fixed recycle
    # Show: recycle=3 sweep, recycle=0 sweep
    r3_configs = [c for c in configs if c["recycle"] == 3]
    r0_configs = [c for c in configs if c["recycle"] == 0]

    x_labels = ["10", "20", "50", "100", "200"]
    r3_speedups = []
    r0_speedups = []
    for s in [10, 20, 50, 100, 200]:
        r3_match = next((c for c in r3_configs if c["steps"] == s), None)
        r0_match = next((c for c in r0_configs if c["steps"] == s), None)
        r3_speedups.append(r3_match["speedup"] if r3_match else 0)
        r0_speedups.append(r0_match["speedup"] if r0_match else 0)

    x = np.arange(len(x_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, r3_speedups, width,
                   label="recycling=3", color=COLORS["recycle_3"], alpha=0.85)
    bars2 = ax.bar(x + width/2, r0_speedups, width,
                   label="recycling=0", color=COLORS["recycle_0"], alpha=0.85)

    # Annotate values
    for bar in bars1:
        if bar.get_height() > 0:
            ax.annotate(f"{bar.get_height():.2f}x",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 3), textcoords="offset points")
    for bar in bars2:
        if bar.get_height() > 0:
            ax.annotate(f"{bar.get_height():.2f}x",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 3), textcoords="offset points")

    ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", alpha=0.5,
               linewidth=1, label="Baseline (1.0x)")
    ax.set_xlabel("Sampling steps")
    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 2.0)

    # --- (b) Pareto frontier: speedup vs pLDDT ---
    ax = axes[1]
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Quality-Speed Pareto Frontier")

    # Color by recycling count
    recycle_colors = {3: COLORS["recycle_3"], 1: COLORS["recycle_1"], 0: COLORS["recycle_0"]}
    recycle_markers = {3: "o", 1: "^", 0: "s"}

    for c in configs:
        if c["steps"] == 10 and c["recycle"] == 3:
            # Skip 10s/3r - catastrophic quality failure, off-chart
            continue
        color = recycle_colors.get(c["recycle"], "#888888")
        marker = recycle_markers.get(c["recycle"], "o")
        if c["steps"] == 200 and c["recycle"] == 3:
            color = COLORS["baseline"]
            marker = "D"
        ax.scatter(c["plddt"], c["speedup"], c=color, s=80, zorder=5,
                   marker=marker, edgecolor="white", linewidth=0.5)
        # Offset labels to avoid overlap
        ha = "left"
        xoff = 6
        if c["label"] in ("50s/1r",):
            ha = "right"
            xoff = -6
        ax.annotate(c["label"].replace("\n", " "),
                    (c["plddt"], c["speedup"]),
                    fontsize=7.5, ha=ha,
                    xytext=(xoff, 5), textcoords="offset points")

    # Quality floor
    ax.axvline(x=0.7107 - 0.02, color=COLORS["fail"], linestyle="--",
               alpha=0.5, linewidth=1, label="Quality floor (-2pp)")

    # Legend for recycling counts
    import matplotlib.lines as mlines
    h_bl = mlines.Line2D([], [], color=COLORS["baseline"], marker="D",
                         linestyle="None", markersize=8, label="Baseline (200s/3r)")
    h_r3 = mlines.Line2D([], [], color=COLORS["recycle_3"], marker="o",
                         linestyle="None", markersize=8, label="recycle=3")
    h_r1 = mlines.Line2D([], [], color=COLORS["recycle_1"], marker="^",
                         linestyle="None", markersize=8, label="recycle=1")
    h_r0 = mlines.Line2D([], [], color=COLORS["recycle_0"], marker="s",
                         linestyle="None", markersize=8, label="recycle=0")
    ax.legend(handles=[h_bl, h_r3, h_r1, h_r0], loc="lower left", fontsize=8)

    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("Speedup")
    ax.set_xlim(0.68, 0.76)
    ax.set_ylim(0.5, 2.0)

    # --- (c) Wall time breakdown: MSA amortization insight ---
    ax = axes[2]
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Wall Time vs Step Count")

    # Plot time vs steps for each recycling level
    for recycle_val, color, label in [
        (3, COLORS["recycle_3"], "recycling=3"),
        (1, COLORS["recycle_1"], "recycling=1"),
        (0, COLORS["recycle_0"], "recycling=0"),
    ]:
        subset = [c for c in configs if c["recycle"] == recycle_val and c["gate"]]
        if not subset:
            continue
        subset.sort(key=lambda c: c["steps"])
        steps = [c["steps"] for c in subset]
        times = [c["time"] for c in subset]
        ax.plot(steps, times, "o-", color=color, label=label, alpha=0.8,
                markersize=6, linewidth=1.5)

    ax.axhline(y=70.37, color=COLORS["baseline"], linestyle="--", alpha=0.5,
               linewidth=1, label="Baseline (70.4s)")
    ax.set_xlabel("Sampling steps")
    ax.set_ylabel("Mean wall time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.set_xticklabels(["10", "20", "50", "100", "200"])

    fig.suptitle(
        "Boltz-2 Step Reduction: Recycling Steps Dominate Speedup, Not Diffusion Steps",
        fontsize=13,
        fontweight="medium",
        y=1.02,
    )

    outpath = FIGURES_DIR / "pareto_frontier.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")
    return outpath


if __name__ == "__main__":
    make_combined_figure()
