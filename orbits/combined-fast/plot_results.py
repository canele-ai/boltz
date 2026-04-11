"""Generate analysis figure for combined-fast orbit.

Multi-panel figure showing:
(a) Time breakdown: baseline vs optimized configs
(b) Quality-speed Pareto frontier (extending parent orbit data)
(c) MSA variance impact on measured speedup

Usage:
    python orbits/combined-fast/plot_results.py
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "step_reduction": "#4C72B0",
    "combined_fast": "#DD8452",
    "failed": "#C44E52",
    "quality_gate": "#55A868",
}

script_dir = Path(__file__).resolve().parent
fig_dir = script_dir / "figures"
fig_dir.mkdir(exist_ok=True)


def load_data():
    """Load all results data."""
    data = {}

    # Parent orbit data (step-reduction validation)
    parent_val_path = script_dir / ".." / "step-reduction" / "validation_results.json"
    if parent_val_path.exists():
        with open(parent_val_path) as f:
            data["parent_validation"] = json.load(f)

    # Parent orbit sweep
    parent_sweep_path = script_dir / ".." / "step-reduction" / "sweep_results.json"
    if parent_sweep_path.exists():
        with open(parent_sweep_path) as f:
            data["parent_sweep"] = json.load(f)

    # Our optimization results
    opt_path = script_dir / "optimization_results.json"
    if opt_path.exists():
        with open(opt_path) as f:
            data["optimization"] = json.load(f)

    # Our validation results
    val_path = script_dir / "validation_results.json"
    if val_path.exists():
        with open(val_path) as f:
            data["validation"] = json.load(f)

    # Official evaluator results (20s/0r)
    eval_20_path = script_dir / "eval_20s_0r.json"
    if eval_20_path.exists():
        with open(eval_20_path) as f:
            data["eval_20s_0r"] = json.load(f)

    # Official evaluator results (15s/0r)
    eval_15_path = script_dir / "eval_15s_0r.json"
    if eval_15_path.exists():
        with open(eval_15_path) as f:
            data["eval_15s_0r"] = json.load(f)

    return data


def plot_time_breakdown(ax, data):
    """Panel (a): Per-complex time comparison across configs."""
    complexes = ["small_complex", "medium_complex", "large_complex"]
    short_names = ["Small (~200 res)", "Medium (~400 res)", "Large (~600 res)"]

    # Baseline data from config.yaml
    baseline_times = {
        "small_complex": 53.0,
        "medium_complex": 70.5,
        "large_complex": 87.6,
    }
    baseline_mean = 70.37

    # Parent orbit 20s/0r validation data
    parent_20_0_times = {}
    if "parent_validation" in data:
        for result in data["parent_validation"]:
            cfg = result["config"]
            if cfg["sampling_steps"] == 20 and cfg["recycling_steps"] == 0:
                for pc in result["per_complex"]:
                    parent_20_0_times[pc["name"]] = pc["wall_time_s"]

    x = np.arange(len(complexes))
    width = 0.35

    bars_baseline = [baseline_times.get(c, 0) for c in complexes]
    bars_opt = [parent_20_0_times.get(c, 0) for c in complexes]

    rects1 = ax.bar(x - width/2, bars_baseline, width, label="Baseline (200s/3r)",
                    color=COLORS["baseline"], alpha=0.8)
    rects2 = ax.bar(x + width/2, bars_opt, width, label="Optimized (20s/0r)",
                    color=COLORS["step_reduction"], alpha=0.8)

    ax.set_ylabel("Wall time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.text(-0.15, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Per-complex wall time")

    # Add value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)


def plot_pareto(ax, data):
    """Panel (b): Quality-speed Pareto frontier."""
    # Collect all configs with speedup and pLDDT
    points = []

    # Baseline
    points.append({"label": "Baseline (200s/3r)", "speedup": 1.0,
                   "plddt": 0.7107, "gate": True, "source": "baseline"})

    # Parent orbit validation data
    if "parent_validation" in data:
        for result in data["parent_validation"]:
            cfg = result["config"]
            agg = result.get("aggregate", {})
            if agg.get("speedup") and agg.get("mean_plddt"):
                label = f"{cfg['sampling_steps']}s/{cfg['recycling_steps']}r"
                gate = agg.get("passes_quality_gate", False)
                points.append({
                    "label": label,
                    "speedup": agg["speedup"],
                    "plddt": agg["mean_plddt"],
                    "gate": gate,
                    "source": "parent",
                })

    # Parent orbit sweep data (phase 1)
    if "parent_sweep" in data:
        for result in data["parent_sweep"].get("phase1_step_sweep", []):
            cfg = result["config"]
            agg = result.get("aggregate", {})
            if agg.get("speedup") and agg.get("mean_plddt"):
                label = f"{cfg['sampling_steps']}s/{cfg['recycling_steps']}r"
                gate = agg.get("passes_quality_gate", False)
                points.append({
                    "label": label,
                    "speedup": agg["speedup"],
                    "plddt": agg["mean_plddt"],
                    "gate": gate,
                    "source": "parent_sweep",
                })

    # Our optimization results
    if "optimization" in data:
        for entry in data["optimization"]:
            result = entry["result"]
            agg = result.get("aggregate", {})
            if agg.get("speedup") and agg.get("mean_plddt"):
                label = entry["label"].split(":")[0].strip()
                gate = agg.get("passes_quality_gate", False)
                points.append({
                    "label": label,
                    "speedup": agg["speedup"],
                    "plddt": agg["mean_plddt"],
                    "gate": gate,
                    "source": "combined_fast",
                })

    # Official evaluator results
    for key, label_prefix in [("eval_20s_0r", "20s/0r-val"), ("eval_15s_0r", "15s/0r-val")]:
        if key in data:
            result = data[key]
            agg = result.get("aggregate", {})
            if agg.get("speedup") and agg.get("mean_plddt"):
                gate = agg.get("passes_quality_gate", False)
                points.append({
                    "label": label_prefix,
                    "speedup": agg["speedup"],
                    "plddt": agg["mean_plddt"],
                    "gate": gate,
                    "source": "combined_fast_validated",
                })

    # Plot
    for pt in points:
        color = COLORS["baseline"] if pt["source"] == "baseline" else (
            COLORS["failed"] if not pt["gate"] else (
            COLORS["step_reduction"] if "parent" in pt["source"] else
            COLORS["combined_fast"]))
        marker = "o" if pt["gate"] else "x"
        size = 100 if pt["source"] in ["combined_fast_validated", "baseline"] else 60
        ax.scatter(pt["speedup"], pt["plddt"], c=color, marker=marker, s=size,
                  zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(pt["label"], (pt["speedup"], pt["plddt"]),
                   fontsize=8, xytext=(5, 5), textcoords="offset points")

    # Quality gate line
    ax.axhline(y=0.7107 - 0.02, color=COLORS["quality_gate"], linestyle="--",
              alpha=0.5, label="Quality floor (-2pp)")

    ax.set_xlabel("Speedup (x)")
    ax.set_ylabel("Mean pLDDT")
    ax.legend(loc="lower left")
    ax.text(-0.15, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Quality vs speedup frontier")


def plot_variance(ax, data):
    """Panel (c): Run-to-run variance analysis for 20s/0r."""
    complexes = ["small_complex", "medium_complex", "large_complex"]
    short_names = ["Small", "Medium", "Large"]

    # Collect per-complex run times from parent orbit 20s/0r validation
    run_times_by_complex = {}
    if "parent_validation" in data:
        for result in data["parent_validation"]:
            cfg = result["config"]
            if cfg["sampling_steps"] == 20 and cfg["recycling_steps"] == 0:
                for pc in result["per_complex"]:
                    if pc.get("run_times"):
                        run_times_by_complex[pc["name"]] = pc["run_times"]

    # Also include official evaluator data if available
    if "eval_20s_0r" in data:
        for pc in data["eval_20s_0r"].get("per_complex", []):
            if pc.get("run_times"):
                key = f"{pc['name']}_new"
                run_times_by_complex[key] = pc["run_times"]

    if not run_times_by_complex:
        ax.text(0.5, 0.5, "No per-run data available",
               transform=ax.transAxes, ha="center")
        ax.text(-0.15, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
        return

    positions = []
    labels = []
    data_to_plot = []

    for i, c in enumerate(complexes):
        if c in run_times_by_complex:
            positions.append(i * 2)
            labels.append(f"{short_names[i]}\n(parent)")
            data_to_plot.append(run_times_by_complex[c])
        key_new = f"{c}_new"
        if key_new in run_times_by_complex:
            positions.append(i * 2 + 0.8)
            labels.append(f"{short_names[i]}\n(this orbit)")
            data_to_plot.append(run_times_by_complex[key_new])

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS["step_reduction"])
            patch.set_alpha(0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)

    ax.set_ylabel("Wall time (s)")
    ax.axhline(y=70.37 / 2, color=COLORS["quality_gate"], linestyle=":",
              alpha=0.5, label="2x target (35.2s)")
    ax.legend(loc="upper right")
    ax.text(-0.15, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Run-to-run variance (20s/0r)")


def main():
    data = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_time_breakdown(axes[0], data)
    plot_pareto(axes[1], data)
    plot_variance(axes[2], data)

    fig.suptitle("Boltz-2 Inference Speedup: Combined Optimization Analysis",
                fontsize=14, fontweight="medium", y=1.04)

    out_path = fig_dir / "analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
