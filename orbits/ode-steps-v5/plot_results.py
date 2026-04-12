"""Generate multi-panel figure for ODE step sweep results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# Data from sweep (mean +/- std across 3 seeds)
steps = np.array([6, 8, 10, 12, 15])

# Per-complex pLDDT (mean across seeds)
small_plddt = np.array([0.4291, 0.8593, 0.8505, 0.8853, 0.8722])
medium_plddt = np.array([0.3293, 0.4696, 0.4751, 0.4733, 0.4796])
large_plddt = np.array([0.4463, 0.8153, 0.8196, 0.8200, 0.8210])
mean_plddt = np.array([0.4015, 0.7147, 0.7151, 0.7262, 0.7243])

# pLDDT std across seeds
small_plddt_std = np.array([0.0025, 0.0139, 0.0046, 0.0195, 0.0283])
medium_plddt_std = np.array([0.0022, 0.0044, 0.0024, 0.0061, 0.0075])
large_plddt_std = np.array([0.0021, 0.0019, 0.0020, 0.0010, 0.0023])

# Predict-only time (mean across seeds, seconds)
small_pred = np.array([3.1, 3.2, 3.4, 3.3, 3.7])
medium_pred = np.array([7.0, 7.2, 7.5, 7.1, 7.6])
large_pred = np.array([11.7, 12.2, 12.6, 11.9, 12.0])
mean_pred = np.array([7.3, 7.5, 7.8, 7.4, 7.8])

# Baseline values
bl_mean_plddt = 0.7170
bl_small_plddt = 0.8345
bl_medium_plddt = 0.5095
bl_large_plddt = 0.8070
bl_pred_only = 30.0  # approximate baseline predict-only (200 steps SDE)

COLORS = {
    'baseline': '#888888',
    'small': '#4C72B0',
    'medium': '#DD8452',
    'large': '#55A868',
    'mean': '#C44E52',
    'fail': '#C44E52',
    'pass': '#55A868',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): pLDDT vs steps (per-complex + mean)
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

ax.errorbar(steps, small_plddt, yerr=small_plddt_std, marker='o', label='small (1BRS)',
            color=COLORS['small'], capsize=3, linewidth=1.5)
ax.errorbar(steps, medium_plddt, yerr=medium_plddt_std, marker='s', label='medium (1DQJ)',
            color=COLORS['medium'], capsize=3, linewidth=1.5)
ax.errorbar(steps, large_plddt, yerr=large_plddt_std, marker='^', label='large (2DN2)',
            color=COLORS['large'], capsize=3, linewidth=1.5)
ax.plot(steps, mean_plddt, 'D-', label='mean', color=COLORS['mean'], linewidth=2, markersize=7)

# Baseline reference lines
ax.axhline(bl_mean_plddt, color=COLORS['baseline'], linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(bl_mean_plddt - 0.02, color=COLORS['fail'], linestyle=':', linewidth=1, alpha=0.5)
ax.annotate('baseline mean', xy=(15.2, bl_mean_plddt), fontsize=8, color=COLORS['baseline'], va='bottom')
ax.annotate('-2pp gate', xy=(15.2, bl_mean_plddt - 0.02), fontsize=8, color=COLORS['fail'], va='top')

# Mark 6-step failure zone
ax.axvspan(5.5, 7.0, alpha=0.08, color=COLORS['fail'])
ax.annotate('fails quality gate', xy=(6, 0.35), fontsize=9, color=COLORS['fail'],
            ha='center', fontstyle='italic')

ax.set_xlabel('ODE sampling steps')
ax.set_ylabel('pLDDT')
ax.set_title('Quality vs step count')
ax.set_xticks(steps)
ax.set_ylim(0.25, 0.96)
ax.legend(loc='lower right', fontsize=9)

# Panel (b): Predict-only time vs steps
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

ax.plot(steps, small_pred, 'o-', label='small', color=COLORS['small'], linewidth=1.5)
ax.plot(steps, medium_pred, 's-', label='medium', color=COLORS['medium'], linewidth=1.5)
ax.plot(steps, large_pred, '^-', label='large', color=COLORS['large'], linewidth=1.5)
ax.plot(steps, mean_pred, 'D-', label='mean', color=COLORS['mean'], linewidth=2, markersize=7)

ax.axhline(bl_pred_only, color=COLORS['baseline'], linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('baseline (SDE-200)', xy=(6, bl_pred_only + 0.5), fontsize=8, color=COLORS['baseline'])

ax.set_xlabel('ODE sampling steps')
ax.set_ylabel('Predict-only time (s)')
ax.set_title('Predict-only time vs step count')
ax.set_xticks(steps)
ax.legend(loc='upper left', fontsize=9)

# Panel (c): Speedup at iso-quality (predict-only)
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

pred_speedup = bl_pred_only / mean_pred

# Color bars by pass/fail
bar_colors = [COLORS['fail'] if s == 6 else COLORS['pass'] for s in steps]
bars = ax.bar(steps, pred_speedup, width=1.5, color=bar_colors, alpha=0.7, edgecolor='white')

# Annotate bars
for i, (s, sp) in enumerate(zip(steps, pred_speedup)):
    label = f'{sp:.1f}x'
    if s == 6:
        label += '\nFAIL'
    ax.text(s, sp + 0.05, label, ha='center', va='bottom', fontsize=9,
            fontweight='bold' if s != 6 else 'normal',
            color=COLORS['fail'] if s == 6 else 'black')

ax.axhline(1.0, color=COLORS['baseline'], linestyle='--', linewidth=1, alpha=0.5)
ax.annotate('1.0x baseline', xy=(5.5, 1.05), fontsize=8, color=COLORS['baseline'])

ax.set_xlabel('ODE sampling steps')
ax.set_ylabel('Predict-only speedup')
ax.set_title('Speedup at iso-quality')
ax.set_xticks(steps)
ax.set_ylim(0, 5.0)

fig.suptitle('ODE Step Count Sweep (bypass + warmup, recycling=0, gamma_0=0.0)',
             fontsize=14, fontweight='medium', y=1.06)

out_path = Path(__file__).parent / 'figures' / 'ode_steps_sweep.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved to {out_path}")
