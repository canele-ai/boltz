"""Plot convergence profile and threshold sweep results."""

import matplotlib
matplotlib.use('Agg')
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

# Colors
COLORS = {
    'small': '#4C72B0',
    'medium': '#DD8452',
    'large': '#55A868',
    'baseline': '#888888',
    'best': '#C44E52',
}

# Convergence data from profiling (seed=42, 20 steps, 3 recycling)
convergence = {
    'small_complex': {
        'passes': [1, 2, 3],
        'cosine_sim': [0.9724, 0.9891, 0.9931],
        'rel_change': [0.2334, 0.1475, 0.1175],
    },
    'medium_complex': {
        'passes': [1, 2, 3],
        'cosine_sim': [0.9806, 0.9916, 0.9959],
        'rel_change': [0.1969, 0.1292, 0.0900],
    },
    'large_complex': {
        'passes': [1, 2, 3],
        'cosine_sim': [0.9764, 0.9965, 0.9989],
        'rel_change': [0.2169, 0.0838, 0.0475],
    },
}

# Sweep results (single seed=42)
sweep_results = {
    'threshold': ['0.95', '0.98', '0.99', '0.999', 'recycle=0'],
    'passes_small': [2, 3, 4, 4, 1],
    'passes_medium': [2, 2, 4, 4, 1],
    'passes_large': [2, 3, 3, 4, 1],
    'plddt': [0.7095, 0.7241, 0.7080, 0.7062, 0.7255],
    # Timings are noisy (MSA server) - mark as approximate
    'time_approx': [75.0, 74.3, 188.9, 88.5, 234.0],
}

# Baseline reference
baseline_plddt = 0.7107

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel (a): Cosine similarity convergence ---
ax = axes[0]
for name, color in [('small_complex', COLORS['small']),
                     ('medium_complex', COLORS['medium']),
                     ('large_complex', COLORS['large'])]:
    data = convergence[name]
    ax.plot(data['passes'], data['cosine_sim'],
            'o-', color=color, label=name.replace('_', ' '), linewidth=2, markersize=6)

# Threshold lines
for t, ls in [(0.95, ':'), (0.98, '--'), (0.99, '-.')]:
    ax.axhline(y=t, color='#888888', linestyle=ls, linewidth=0.8, alpha=0.6)
    ax.text(3.05, t, f'{t}', fontsize=8, va='center', color='#888888')

ax.set_xlabel('Recycling pass')
ax.set_ylabel('Cosine similarity (z vs z_prev)')
ax.set_title('Pair representation convergence')
ax.set_xlim(0.8, 3.2)
ax.set_ylim(0.96, 1.001)
ax.set_xticks([1, 2, 3])
ax.legend(loc='lower right')
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# --- Panel (b): Trunk passes per complex per threshold ---
ax = axes[1]
x = np.arange(len(sweep_results['threshold']))
w = 0.25
ax.bar(x - w, sweep_results['passes_small'], w, label='small', color=COLORS['small'], alpha=0.8)
ax.bar(x,     sweep_results['passes_medium'], w, label='medium', color=COLORS['medium'], alpha=0.8)
ax.bar(x + w, sweep_results['passes_large'],  w, label='large', color=COLORS['large'], alpha=0.8)

ax.set_xlabel('Configuration')
ax.set_ylabel('Trunk passes used')
ax.set_title('Recycling passes by threshold')
ax.set_xticks(x)
ax.set_xticklabels(sweep_results['threshold'], fontsize=9)
ax.set_ylim(0, 5)
ax.axhline(y=4, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(0.5, 4.15, 'baseline (3 recycle = 4 passes)', fontsize=8, color='#888888',
        ha='center')
ax.legend(loc='upper right')
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# --- Panel (c): pLDDT comparison ---
ax = axes[2]
colors_bar = [COLORS['small']] * 4 + [COLORS['best']]
x = np.arange(len(sweep_results['threshold']))
bars = ax.bar(x, sweep_results['plddt'], color=[COLORS['small'], COLORS['medium'],
              COLORS['large'], '#8172B3', COLORS['best']], alpha=0.8)

ax.axhline(y=baseline_plddt, color=COLORS['baseline'], linestyle='--', linewidth=1.5,
           label=f'baseline (pLDDT={baseline_plddt:.4f})')
ax.axhline(y=baseline_plddt - 0.02, color=COLORS['baseline'], linestyle=':',
           linewidth=1, alpha=0.5, label='quality floor (-2pp)')

ax.set_xlabel('Configuration')
ax.set_ylabel('Mean pLDDT')
ax.set_title('Quality vs threshold')
ax.set_xticks(x)
ax.set_xticklabels(sweep_results['threshold'], fontsize=9)
ax.set_ylim(0.68, 0.74)
ax.legend(loc='lower left', fontsize=9)
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Add value labels
for i, v in enumerate(sweep_results['plddt']):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)

fig.suptitle('Early-Exit Recycling: Convergence and Quality Analysis\n'
             '(20 sampling steps, seed=42, single run)', y=1.06, fontsize=14)

fig.savefig('/home/liambai/code/boltz/.worktrees/early-exit-recycling/orbits/early-exit-recycling/figures/convergence_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved convergence_analysis.png")
