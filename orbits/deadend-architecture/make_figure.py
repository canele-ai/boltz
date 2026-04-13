"""Generate summary figure for dead-end architecture evaluation."""
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
    'control': '#4C72B0',
    'dt8': '#55A868',
    'pf48': '#DD8452',
    'tome10': '#C44E52',
}

# Data from eval-v5 run
configs = ['Baseline\n(200-step)', 'Control\n(ODE-12)', 'DT 24->8', 'PF 64->48', 'ToMe 10%']
colors = ['#888888', '#4C72B0', '#55A868', '#DD8452', '#C44E52']

# pLDDT per complex
plddt_data = {
    'small': [0.9671, 0.9674, 0.9674, 0.7390, 0.5161],
    'medium': [0.9623, 0.9644, 0.9644, 0.7048, 0.4527],
    'large': [0.9655, 0.9648, 0.9648, 0.6478, 0.7244],
}

# CA RMSD per complex
rmsd_data = {
    'small': [0.325, 0.297, 0.296, 0.986, 11.933],
    'medium': [5.243, 5.384, 5.385, 17.780, 26.504],
    'large': [0.474, 0.498, 0.498, 23.872, 20.122],
}

# Mean values
mean_plddt = [0.9650, 0.9655, 0.9655, 0.6972, 0.5644]
mean_rmsd = [2.014, 2.060, 2.060, 14.213, 19.520]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Mean pLDDT comparison
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
bars = ax.bar(range(len(configs)), mean_plddt, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=0.9650, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.9650 - 0.02, color='#C44E52', linestyle=':', alpha=0.5, linewidth=1, label='Quality gate (-2pp)')
ax.set_ylabel('Mean pLDDT')
ax.set_title('Quality: pLDDT')
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8)
# Annotate values on bars
for i, (bar, val) in enumerate(zip(bars, mean_plddt)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)

# Panel (b): Mean CA RMSD comparison
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
bars = ax.bar(range(len(configs)), mean_rmsd, color=colors, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Mean CA RMSD (A)')
ax.set_title('Structural quality: CA RMSD vs ground truth')
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylim(0, 25)
# Annotate values on bars
for i, (bar, val) in enumerate(zip(bars, mean_rmsd)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
            ha='center', va='bottom', fontsize=8)

# Panel (c): Per-complex CA RMSD heatmap
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

complexes = ['1BRS\n(small)', '1DQJ\n(medium)', '2DN2\n(large)']
rmsd_matrix = np.array([
    rmsd_data['small'],
    rmsd_data['medium'],
    rmsd_data['large'],
])

im = ax.imshow(rmsd_matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=30)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=8)
ax.set_yticks(range(len(complexes)))
ax.set_yticklabels(complexes, fontsize=9)
ax.set_title('Per-complex CA RMSD (A)')
ax.grid(False)

# Annotate cells
for i in range(len(complexes)):
    for j in range(len(configs)):
        val = rmsd_matrix[i, j]
        color = 'white' if val > 15 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('CA RMSD (A)', fontsize=9)

fig.suptitle('Dead-end architecture approaches: DT layer pruning, PF block pruning, Token Merging',
             fontsize=12, y=1.02)

fig.savefig('/home/liambai/code/boltz/.worktrees/deadend-architecture/orbits/deadend-architecture/figures/deadend_results.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Figure saved.")
