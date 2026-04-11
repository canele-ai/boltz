"""Generate results figure for flash-sdpa orbit."""

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

COLORS = {
    'baseline': '#888888',
    'fp32_sdpa': '#4C72B0',
    'bf16_sdpa': '#DD8452',
}

# --- Data ---
# Baseline (from config.yaml, seed=42)
baseline = {
    'small': {'time': 53.01, 'plddt': 0.8350},
    'medium': {'time': 70.53, 'plddt': 0.4906},
    'large': {'time': 87.57, 'plddt': 0.8064},
}

# bf16 SDPA (3 seeds, mean)
bf16_per_complex = {
    'small': {
        'times': [103.4, 100.7, 96.2],
        'plddts': [0.8369, 0.8396, 0.8495],
    },
    'medium': {
        'times': [74.8, 78.7, 71.2],
        'plddts': [0.4773, 0.4878, 0.4617],
    },
    'large': {
        'times': [108.9, 112.8, 104.6],
        'plddts': [0.8048, 0.8089, 0.8118],
    },
}

# Aggregate across seeds
bf16_agg = {
    'times': [95.72, 97.42, 90.64],
    'plddts': [0.7063, 0.7121, 0.7076],
}

# fp32 SDPA (seeds 42, 123 only -- seed 7 had MSA outlier)
fp32_agg = {
    'times': [95.03, 106.00],  # excluding seed 7 (278s MSA outlier)
    'plddts': [0.7075, 0.7098],
}

# --- Figure: 3-panel ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): Wall time per complex
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

complexes = ['small\n(~200 res)', 'medium\n(~400 res)', 'large\n(~600 res)']
complex_keys = ['small', 'medium', 'large']
x = np.arange(len(complexes))
width = 0.25

# Baseline bars
bl_times = [baseline[k]['time'] for k in complex_keys]
ax.bar(x - width, bl_times, width, color=COLORS['baseline'], label='Baseline (einsum)')

# bf16 SDPA bars (mean + error)
bf16_means = [np.mean(bf16_per_complex[k]['times']) for k in complex_keys]
bf16_stds = [np.std(bf16_per_complex[k]['times']) for k in complex_keys]
ax.bar(x, bf16_means, width, color=COLORS['bf16_sdpa'], label='SDPA bf16',
       yerr=bf16_stds, capsize=3)

ax.set_xlabel('Test complex')
ax.set_ylabel('Wall time (s)')
ax.set_title('Per-complex wall time')
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc='upper left')

# Panel (b): Speedup comparison
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

methods = ['Baseline', 'SDPA fp32', 'SDPA bf16']
mean_times = [
    70.37,  # baseline
    np.mean(fp32_agg['times']),  # fp32 SDPA (excl. outlier)
    np.mean(bf16_agg['times']),  # bf16 SDPA
]
std_times = [
    0,  # baseline is a single reference point
    np.std(fp32_agg['times']),
    np.std(bf16_agg['times']),
]
speedups = [70.37 / t for t in mean_times]

colors = [COLORS['baseline'], COLORS['fp32_sdpa'], COLORS['bf16_sdpa']]
bars = ax.bar(methods, speedups, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Speedup (T_baseline / T_optimized)')
ax.set_title('Speedup at iso-quality')
ax.set_ylim(0, 1.4)

# Annotate values
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{sp:.2f}x', ha='center', va='bottom', fontsize=10)

# Panel (c): pLDDT comparison
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Scatter: baseline pLDDT vs SDPA pLDDT per complex
bl_plddts = [baseline[k]['plddt'] for k in complex_keys]
bf16_plddts_mean = [np.mean(bf16_per_complex[k]['plddts']) for k in complex_keys]
bf16_plddts_std = [np.std(bf16_per_complex[k]['plddts']) for k in complex_keys]

ax.errorbar(bl_plddts, bf16_plddts_mean, yerr=bf16_plddts_std,
            fmt='o', color=COLORS['bf16_sdpa'], markersize=8, capsize=4,
            label='SDPA bf16')

# Perfect agreement line
lims = [0.4, 0.9]
ax.plot(lims, lims, '--', color=COLORS['baseline'], linewidth=1, label='Perfect agreement')

# 2pp quality gate
ax.fill_between(lims, [l - 0.02 for l in lims], lims, alpha=0.1, color='green',
                label='Quality gate (2pp)')

ax.set_xlabel('Baseline pLDDT')
ax.set_ylabel('SDPA bf16 pLDDT')
ax.set_title('Quality preservation')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)

# Label points
for k, bx, by in zip(complex_keys, bl_plddts, bf16_plddts_mean):
    ax.annotate(k, (bx, by), textcoords="offset points", xytext=(8, -5),
                fontsize=9, color='gray')

fig.savefig('orbits/flash-sdpa/figures/results.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print("Saved orbits/flash-sdpa/figures/results.png")
