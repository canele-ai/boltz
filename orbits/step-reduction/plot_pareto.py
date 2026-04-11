"""Generate quality-speed Pareto frontier figure from sweep results.

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
}

ORBIT_DIR = Path(__file__).parent
RESULTS_FILE = ORBIT_DIR / "sweep_results.json"
FIGURES_DIR = ORBIT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results():
    with RESULTS_FILE.open() as f:
        return json.load(f)


def make_pareto_figure(data):
    """Three-panel figure: (a) pLDDT vs steps, (b) speedup vs steps, (c) Pareto frontier."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Collect data points: baseline + phase1 + phase2
    # Baseline
    bl_steps = 200
    bl_recycle = 3
    bl_time = 70.37
    bl_plddt = 0.7107
    bl_speedup = 1.0

    points = [{
        "steps": bl_steps,
        "recycle": bl_recycle,
        "time": bl_time,
        "plddt": bl_plddt,
        "speedup": bl_speedup,
        "gate": True,
        "label": f"200s/3r",
    }]

    for r in data.get("phase1_step_sweep", []) + data.get("phase2_recycle_sweep", []):
        cfg = r["config"]
        agg = r.get("aggregate", {})
        if agg.get("mean_wall_time_s") is None:
            continue
        points.append({
            "steps": cfg["sampling_steps"],
            "recycle": cfg["recycling_steps"],
            "time": agg["mean_wall_time_s"],
            "plddt": agg.get("mean_plddt", 0),
            "speedup": agg.get("speedup", 0),
            "gate": agg.get("passes_quality_gate", False),
            "label": f"{cfg['sampling_steps']}s/{cfg['recycling_steps']}r",
        })

    # Sort by steps for line plotting
    step_sweep = [p for p in points if p["recycle"] == 3]
    step_sweep.sort(key=lambda p: p["steps"])

    steps_arr = [p["steps"] for p in step_sweep]
    plddt_arr = [p["plddt"] for p in step_sweep]
    speedup_arr = [p["speedup"] for p in step_sweep]
    gate_arr = [p["gate"] for p in step_sweep]

    # (a) pLDDT vs sampling steps
    ax = axes[0]
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Quality vs Sampling Steps")
    ax.set_xlabel("Sampling steps")
    ax.set_ylabel("Mean pLDDT")

    for i, p in enumerate(step_sweep):
        color = COLORS["pass"] if p["gate"] else COLORS["fail"]
        marker = "o" if p["steps"] != 200 else "s"
        ax.scatter(p["steps"], p["plddt"], c=color, s=80, zorder=5, marker=marker,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(p["label"], (p["steps"], p["plddt"]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.plot(steps_arr, plddt_arr, color="#4C72B0", alpha=0.4, linewidth=1.5, zorder=2)

    # Quality floor line
    ax.axhline(y=bl_plddt - 0.02, color=COLORS["fail"], linestyle="--",
               alpha=0.5, linewidth=1, label="Quality floor (-2pp)")
    ax.axhline(y=bl_plddt, color=COLORS["baseline"], linestyle="--",
               alpha=0.5, linewidth=1, label="Baseline pLDDT")
    ax.legend(loc="lower left")
    ax.set_xscale("log")
    ax.set_xticks(steps_arr)
    ax.set_xticklabels([str(s) for s in steps_arr])

    # (b) Speedup vs sampling steps
    ax = axes[1]
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Speedup vs Sampling Steps")
    ax.set_xlabel("Sampling steps")
    ax.set_ylabel("Speedup (T_baseline / T_optimized)")

    for i, p in enumerate(step_sweep):
        color = COLORS["pass"] if p["gate"] else COLORS["fail"]
        marker = "o" if p["steps"] != 200 else "s"
        ax.scatter(p["steps"], p["speedup"], c=color, s=80, zorder=5, marker=marker,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(f"{p['speedup']:.2f}x", (p["steps"], p["speedup"]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.plot(steps_arr, speedup_arr, color="#4C72B0", alpha=0.4, linewidth=1.5, zorder=2)
    ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(steps_arr)
    ax.set_xticklabels([str(s) for s in steps_arr])

    # (c) Pareto frontier: speedup vs pLDDT (all configs)
    ax = axes[2]
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_title("Quality-Speed Pareto Frontier")
    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("Speedup")

    for p in points:
        color = COLORS["pass"] if p["gate"] else COLORS["fail"]
        marker = "s" if p["steps"] == 200 and p["recycle"] == 3 else "o"
        ax.scatter(p["plddt"], p["speedup"], c=color, s=80, zorder=5, marker=marker,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(p["label"], (p["plddt"], p["speedup"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    # Quality floor
    ax.axvline(x=bl_plddt - 0.02, color=COLORS["fail"], linestyle="--",
               alpha=0.5, linewidth=1, label="Quality floor")

    # Pareto frontier (connect non-dominated points)
    passing_pts = [p for p in points if p["gate"]]
    if passing_pts:
        passing_pts.sort(key=lambda p: p["speedup"])
        pareto = []
        best_plddt = -1
        for p in passing_pts:
            if p["plddt"] >= best_plddt:
                pareto.append(p)
                best_plddt = p["plddt"]
        if len(pareto) > 1:
            ax.plot(
                [p["plddt"] for p in pareto],
                [p["speedup"] for p in pareto],
                color=COLORS["pareto"], linestyle="-", alpha=0.4, linewidth=1.5,
                label="Pareto frontier",
            )

    ax.legend(loc="upper left")

    fig.suptitle(
        "Boltz-2 Diffusion Step Reduction: Quality vs Speed",
        fontsize=14,
        fontweight="medium",
        y=1.02,
    )

    outpath = FIGURES_DIR / "pareto_frontier.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")
    return outpath


def make_per_complex_figure(data):
    """Per-complex breakdown: bar chart showing pLDDT per complex per config."""
    all_results = data.get("phase1_step_sweep", []) + data.get("phase2_recycle_sweep", [])
    if not all_results:
        return

    complex_names = ["small_complex", "medium_complex", "large_complex"]
    configs = []
    plddt_matrix = []  # rows=configs, cols=complexes
    time_matrix = []

    # Baseline
    configs.append("200s/3r\n(baseline)")
    plddt_matrix.append([0.8350, 0.4906, 0.8064])
    time_matrix.append([53.01, 70.53, 87.57])

    for r in all_results:
        cfg = r["config"]
        agg = r.get("aggregate", {})
        if agg.get("mean_wall_time_s") is None:
            continue
        label = f"{cfg['sampling_steps']}s/{cfg['recycling_steps']}r"
        configs.append(label)
        row_plddt = []
        row_time = []
        for cn in complex_names:
            pc = next((p for p in r["per_complex"] if p["name"] == cn), None)
            if pc and pc.get("quality", {}).get("complex_plddt") is not None:
                row_plddt.append(pc["quality"]["complex_plddt"])
            else:
                row_plddt.append(0)
            if pc and pc.get("wall_time_s") is not None:
                row_time.append(pc["wall_time_s"])
            else:
                row_time.append(0)
        plddt_matrix.append(row_plddt)
        time_matrix.append(row_time)

    n_configs = len(configs)
    n_complexes = len(complex_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # (a) pLDDT per complex
    ax = axes[0]
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    x = np.arange(n_configs)
    width = 0.25
    discrete_colors = ["#4C72B0", "#DD8452", "#55A868"]

    for j, cn in enumerate(complex_names):
        vals = [plddt_matrix[i][j] for i in range(n_configs)]
        ax.bar(x + j * width, vals, width, label=cn.replace("_", " ").title(),
               color=discrete_colors[j], alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("pLDDT")
    ax.set_title("Per-Complex pLDDT")
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs, fontsize=8)
    ax.legend(loc="upper right")

    # (b) Wall time per complex
    ax = axes[1]
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    for j, cn in enumerate(complex_names):
        vals = [time_matrix[i][j] for i in range(n_configs)]
        ax.bar(x + j * width, vals, width, label=cn.replace("_", " ").title(),
               color=discrete_colors[j], alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Per-Complex Wall Time")
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs, fontsize=8)
    ax.legend(loc="upper right")

    fig.suptitle(
        "Per-Complex Quality and Speed Across Configurations",
        fontsize=14,
        fontweight="medium",
        y=1.02,
    )

    outpath = FIGURES_DIR / "per_complex_breakdown.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")
    return outpath


if __name__ == "__main__":
    data = load_results()
    make_pareto_figure(data)
    make_per_complex_figure(data)
