"""Generate figures for diffusion loop optimization orbit.

Multi-panel figure:
(a) Speedup comparison across configs (bar chart with error bars)
(b) Per-complex timing breakdown
(c) Quality (pLDDT) comparison
"""
import json
import sys
from pathlib import Path

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
    'ODE-20-r0': '#4C72B0',
    'ODE-10-r0': '#DD8452',
    'ODE-20-r0-tf32': '#55A868',
    'ODE-10-r0-tf32': '#C44E52',
    'ODE-20-r0-compile': '#8172B3',
}


def load_results():
    """Load results from clean_results_full.json."""
    results_path = Path(__file__).parent / "clean_results_full.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)
    with results_path.open() as f:
        return json.load(f)


def make_figure(results):
    """Create multi-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Baseline reference
    baseline_time = 53.57  # from config.yaml
    baseline_plddt = 0.7170

    labels = []
    speedups = []
    speedup_stds = []
    times = []
    time_stds = []
    plddts = []
    plddt_stds = []
    colors = []

    for r in results:
        s = r["summary"]
        if s["mean_speedup"] is None:
            continue
        labels.append(r["label"])
        speedups.append(s["mean_speedup"])
        speedup_stds.append(s["std_speedup"])
        times.append(s["mean_time"])
        time_stds.append(s["std_time"])
        plddts.append(s["mean_plddt"])
        plddt_stds.append(s["std_plddt"])
        colors.append(COLORS.get(r["label"], '#888888'))

    x = np.arange(len(labels))

    # --- Panel (a): Speedup ---
    ax = axes[0]
    bars = ax.bar(x, speedups, yerr=speedup_stds, capsize=4,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1, alpha=0.7,
               label='Baseline (200s/3r)')
    ax.set_ylabel('Speedup vs baseline')
    ax.set_title('Speedup at iso-quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.legend(loc='upper left')
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, val, std in zip(bars, speedups, speedup_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

    # --- Panel (b): Per-complex timing ---
    ax = axes[1]
    complex_names = ['small_complex', 'medium_complex', 'large_complex']
    complex_short = ['Small\n(~200 res)', 'Medium\n(~400 res)', 'Large\n(~600 res)']
    width = 0.18
    x_complex = np.arange(len(complex_names))

    # Baseline timing
    baseline_per_complex = [42.8, 51.3, 66.6]  # from config.yaml
    ax.bar(x_complex - 2*width, baseline_per_complex, width,
           color=COLORS['baseline'], alpha=0.6, label='Baseline (200s/3r)')

    for i, r in enumerate(results):
        if r["summary"]["mean_time"] is None:
            continue
        # Get per-complex times (average across seeds)
        per_complex_times = {}
        for seed_result in r["per_seed"]:
            for pc in seed_result["per_complex"]:
                name = pc["name"]
                if pc.get("error") is None and pc.get("wall_time_s"):
                    per_complex_times.setdefault(name, []).append(pc["wall_time_s"])

        means = [np.mean(per_complex_times.get(cn, [0])) for cn in complex_names]
        ax.bar(x_complex + (i-1)*width, means, width,
               color=colors[i], alpha=0.85, label=r["label"])

    ax.set_ylabel('Wall time (s)')
    ax.set_title('Per-complex timing')
    ax.set_xticks(x_complex)
    ax.set_xticklabels(complex_short)
    ax.legend(loc='upper left', fontsize=8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # --- Panel (c): Quality ---
    ax = axes[2]
    bars = ax.bar(x, plddts, yerr=plddt_stds, capsize=4,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=baseline_plddt, color='#888888', linestyle='--', linewidth=1,
               alpha=0.7, label=f'Baseline pLDDT={baseline_plddt:.4f}')
    ax.axhline(y=baseline_plddt - 0.02, color='#C44E52', linestyle=':',
               linewidth=1, alpha=0.5, label='Quality floor (-2pp)')
    ax.set_ylabel('Mean pLDDT')
    ax.set_title('Quality preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylim(baseline_plddt - 0.04, max(plddts) + max(plddt_stds) + 0.005)
    ax.legend(loc='lower left', fontsize=8)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.suptitle('Diffusion Loop Optimization: ODE Sampler + TF32 on eval-v2 (L40S)',
                 fontsize=14, fontweight='medium', y=1.02)

    # Save
    out_path = Path(__file__).parent / "figures" / "diffusion_loop_opt.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figure saved to {out_path}")
    return out_path


if __name__ == "__main__":
    results = load_results()
    make_figure(results)
