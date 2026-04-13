#!/usr/bin/env python3
"""Plot recycling RMSD study results."""
import matplotlib
matplotlib.use('Agg')
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
    'baseline': '#888888',
    'small': '#4C72B0',
    'medium': '#DD8452',
    'large': '#55A868',
    'fail': '#C44E52',
}

# Data
recycling_steps = [0, 1, 2, 3]
baseline_recycling = 3  # baseline uses 3 recycling steps with 200 diffusion steps

# CA RMSD data (from evaluator runs)
ca_rmsd = {
    'small':  [0.328, 0.301, 0.299, 0.300],  # 1BRS
    'medium': [5.317, 5.341, 5.405, 5.404],  # 1DQJ
    'large':  [2.344, 1.917, 0.540, 0.528],  # 2DN2
}

# Baseline CA RMSD (200 steps, 3 recycles)
baseline_rmsd = {
    'small': 0.325,
    'medium': 5.243,
    'large': 0.474,
}

# pLDDT data
plddt = {
    'small':  [None, 0.9684, 0.9678, 0.9675],  # recycle=0 failed in first run
    'medium': [0.9639, 0.9651, 0.9648, 0.9649],
    'large':  [0.9502, 0.9564, 0.9640, 0.9654],
}

baseline_plddt = {
    'small': 0.9671,
    'medium': 0.9623,
    'large': 0.9655,
}

# Speedup data (predict_only_s based)
# Using wall times from successful configs
speedups = [None, 1.66, 1.49, 1.31]  # recycle=0 had a failure

# Timing data (median wall time per complex, seconds)
wall_times = {
    'small':  [None, 11.46, 10.00, 10.11],
    'medium': [15.49, 19.63, 22.22, 25.91],
    'large':  [17.73, 22.66, 27.58, 32.00],
}

# --- Figure: 3-panel ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): CA RMSD vs recycling steps
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

for name, label, color in [
    ('small', '1BRS (199 res)', COLORS['small']),
    ('medium', '1DQJ (563 res)', COLORS['medium']),
    ('large', '2DN2 (574 res)', COLORS['large']),
]:
    rmsd = ca_rmsd[name]
    ax.plot(recycling_steps, rmsd, 'o-', color=color, label=label, linewidth=2, markersize=8)
    # Baseline reference line
    ax.axhline(y=baseline_rmsd[name], color=color, linestyle='--', alpha=0.4, linewidth=1)

# Mark the quality gate threshold region
ax.axhspan(0, 1.474, alpha=0.03, color='green')  # baseline large + 1.0A
ax.set_xlabel('Recycling steps')
ax.set_ylabel('CA RMSD vs ground truth (A)')
ax.set_title('Structural accuracy')
ax.set_xticks([0, 1, 2, 3])
ax.legend(loc='upper left')

# Annotate the failure zone
ax.annotate('recycle=0: +1.87A\nregression on 2DN2',
           xy=(0, 2.344), fontsize=8, color=COLORS['fail'],
           xytext=(0.5, 3.5),
           arrowprops=dict(arrowstyle='->', color=COLORS['fail'], lw=0.8))

# Panel (b): pLDDT vs recycling steps
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

for name, label, color in [
    ('small', '1BRS', COLORS['small']),
    ('medium', '1DQJ', COLORS['medium']),
    ('large', '2DN2', COLORS['large']),
]:
    vals = plddt[name]
    valid_x = [x for x, v in zip(recycling_steps, vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    ax.plot(valid_x, valid_v, 'o-', color=color, label=label, linewidth=2, markersize=8)
    ax.axhline(y=baseline_plddt[name], color=color, linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('Recycling steps')
ax.set_ylabel('pLDDT')
ax.set_title('Confidence (pLDDT)')
ax.set_xticks([0, 1, 2, 3])
ax.legend(loc='lower right')

# Panel (c): Wall time per complex
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

for name, label, color in [
    ('small', '1BRS', COLORS['small']),
    ('medium', '1DQJ', COLORS['medium']),
    ('large', '2DN2', COLORS['large']),
]:
    vals = wall_times[name]
    valid_x = [x for x, v in zip(recycling_steps, vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    ax.plot(valid_x, valid_v, 'o-', color=color, label=label, linewidth=2, markersize=8)

ax.set_xlabel('Recycling steps')
ax.set_ylabel('Median wall time (s)')
ax.set_title('Inference time')
ax.set_xticks([0, 1, 2, 3])
ax.legend(loc='upper left')

fig.suptitle('Recycling steps vs structural accuracy, confidence, and speed (ODE-12 + TF32 + bf16)',
             fontsize=13, fontweight='medium', y=1.02)

fig.savefig('/home/liambai/code/boltz/.worktrees/recycling-rmsd-v5/orbits/recycling-rmsd-v5/figures/recycling_rmsd_study.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved figure.")
