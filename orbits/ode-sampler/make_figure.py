"""Generate comparison figure: ODE vs SDE sampler results.

Produces a multi-panel figure showing:
(a) pLDDT vs step count for ODE vs SDE
(b) Wall time vs step count
(c) Per-complex pLDDT comparison (validated configs)
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
    "sde": "#4C72B0",
    "ode": "#DD8452",
    "ode_ns1": "#55A868",
}

# Load sweep results (Phase 1)
orbit_dir = Path(__file__).parent
with open(orbit_dir / "sweep_results.json") as f:
    sweep_data = json.load(f)

# Load validation results (Phase 2)
with open(orbit_dir / "validate_results.json") as f:
    validate_data = json.load(f)

# Parse sweep data
sweep_configs = {}
for item in sweep_data:
    label = item["label"]
    r = item["result"]
    agg = r.get("aggregate", {})
    sweep_configs[label] = {
        "steps": r["config"].get("sampling_steps"),
        "gamma_0": r["config"].get("gamma_0", 0.8),
        "noise_scale": r["config"].get("noise_scale", 1.003),
        "plddt": agg.get("mean_plddt"),
        "time": agg.get("mean_wall_time_s"),
        "speedup": agg.get("speedup"),
        "delta_pp": agg.get("plddt_delta_pp"),
    }

# Parse validation data
val_configs = {}
for item in validate_data:
    label = item["label"]
    r = item["result"]
    agg = r.get("aggregate", {})
    per_complex = r.get("per_complex", [])
    val_configs[label] = {
        "steps": r["config"].get("sampling_steps"),
        "gamma_0": r["config"].get("gamma_0", 0.8),
        "plddt": agg.get("mean_plddt"),
        "time": agg.get("mean_wall_time_s"),
        "speedup": agg.get("speedup"),
        "delta_pp": agg.get("plddt_delta_pp"),
        "per_complex": per_complex,
    }

# Baseline values
baseline_plddt = 0.7107
baseline_time = 70.37

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Panel (a): pLDDT vs Steps ---
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# SDE data points
sde_steps = []
sde_plddts = []
for label in ["SDE-20", "SDE-10"]:
    if label in sweep_configs and sweep_configs[label]["plddt"] is not None:
        sde_steps.append(sweep_configs[label]["steps"])
        sde_plddts.append(sweep_configs[label]["plddt"])

# ODE data points
ode_steps = []
ode_plddts = []
for label in ["ODE-20", "ODE-10", "ODE-5"]:
    if label in sweep_configs and sweep_configs[label]["plddt"] is not None:
        ode_steps.append(sweep_configs[label]["steps"])
        ode_plddts.append(sweep_configs[label]["plddt"])

# ODE ns=1.0 data points
ode_ns1_steps = []
ode_ns1_plddts = []
for label in ["ODE-20-ns1", "ODE-10-ns1"]:
    if label in sweep_configs and sweep_configs[label]["plddt"] is not None:
        ode_ns1_steps.append(sweep_configs[label]["steps"])
        ode_ns1_plddts.append(sweep_configs[label]["plddt"])

ax.plot(sde_steps, sde_plddts, 'o-', color=COLORS["sde"], label="SDE (gamma_0=0.8)", markersize=8, linewidth=2)
ax.plot(ode_steps, ode_plddts, 's-', color=COLORS["ode"], label="ODE (gamma_0=0)", markersize=8, linewidth=2)
ax.plot(ode_ns1_steps, ode_ns1_plddts, '^--', color=COLORS["ode_ns1"], label="ODE (ns=1.0)", markersize=7, linewidth=1.5)

# Baseline reference
ax.axhline(y=baseline_plddt, color=COLORS["baseline"], linestyle='--', linewidth=1, label=f"Baseline (200s, 3r)")
# Quality floor
ax.axhline(y=baseline_plddt - 0.02, color='#C44E52', linestyle=':', linewidth=1, alpha=0.7, label="Quality floor (-2pp)")

ax.set_xlabel("Diffusion Steps")
ax.set_ylabel("Mean pLDDT")
ax.set_title("Quality vs Step Count")
ax.set_xlim(3, 25)
ax.set_ylim(0.35, 0.80)
ax.legend(loc='lower right', fontsize=9)

# --- Panel (b): Validated timing comparison ---
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

val_labels = ["SDE-20-r0", "ODE-20-r0", "ODE-10-r0"]
val_display = ["SDE 20s", "ODE 20s", "ODE 10s"]
val_colors_list = [COLORS["sde"], COLORS["ode"], COLORS["ode"]]
val_hatches = [None, None, '//']
val_times = [val_configs[l]["time"] for l in val_labels]
val_speedups = [val_configs[l]["speedup"] for l in val_labels]

x = np.arange(len(val_labels))
bars = ax.bar(x, val_times, color=val_colors_list, width=0.6, edgecolor='white', linewidth=0.5)
for i, hatch in enumerate(val_hatches):
    if hatch:
        bars[i].set_hatch(hatch)

# Add speedup labels
for i, (t, sp) in enumerate(zip(val_times, val_speedups)):
    ax.text(i, t + 1.5, f"{sp:.2f}x", ha='center', va='bottom', fontsize=10, fontweight='medium')

ax.axhline(y=baseline_time, color=COLORS["baseline"], linestyle='--', linewidth=1, label=f"Baseline ({baseline_time:.0f}s)")
ax.set_xticks(x)
ax.set_xticklabels(val_display)
ax.set_ylabel("Mean Wall Time (s)")
ax.set_title("Validated Timing (3 runs)")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 80)

# --- Panel (c): Per-complex pLDDT comparison ---
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

complexes = ["small_complex", "medium_complex", "large_complex"]
complex_short = ["Small", "Medium", "Large"]
x = np.arange(len(complexes))
width = 0.25

# Baseline per-complex pLDDTs
baseline_per_complex = {
    "small_complex": 0.8350,
    "medium_complex": 0.4906,
    "large_complex": 0.8064,
}

for idx, (vl, display, color) in enumerate(zip(val_labels, val_display, val_colors_list)):
    pc_data = val_configs[vl]["per_complex"]
    pc_plddts = []
    for c in complexes:
        for pc in pc_data:
            if pc["name"] == c:
                pc_plddts.append(pc.get("quality", {}).get("complex_plddt", 0))
                break
        else:
            pc_plddts.append(0)
    offset = (idx - 1) * width
    ax.bar(x + offset, pc_plddts, width, color=color, label=display,
           edgecolor='white', linewidth=0.5,
           hatch='//' if idx == 2 else None)

# Baseline bars
baseline_plddts = [baseline_per_complex[c] for c in complexes]
ax.bar(x + 1.5 * width, baseline_plddts, width, color=COLORS["baseline"],
       label="Baseline", edgecolor='white', linewidth=0.5, alpha=0.7)

ax.set_xticks(x + 0.125)
ax.set_xticklabels(complex_short)
ax.set_ylabel("Complex pLDDT")
ax.set_title("Per-Complex Quality")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1.0)

# Save
fig_path = orbit_dir / "figures" / "ode_vs_sde_comparison.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Figure saved to {fig_path}")
