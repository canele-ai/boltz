---
alwaysApply: false
---
# Plotting Style Guide

When creating matplotlib figures, follow this consistent style:

## Layout Philosophy
- **Prefer grid layouts.** When presenting multiple related plots, arrange them as subplots in a grid rather than separate figures.
- Use `GridSpec` for complex layouts with custom width/height ratios.
- Use `fig, axes = plt.subplots(nrows, ncols, figsize=(...), constrained_layout=True)` as the default pattern.
- Aim for information density — one well-organized grid figure is better than 5 separate PNGs.
- For orbit comparisons: one row per strategy, columns for different metrics or views.

## Font & Typography
- `font.family`: "sans-serif"
- `font.sans-serif`: ["Helvetica", "Arial", "DejaVu Sans"]
- `font.size`: 11
- Titles: `fontsize=13, fontweight='medium'` — not bold, not shouty
- Axis labels: `fontsize=11` — clear but not competing with the data

## Axes & Spines
- `axes.spines.top`: False
- `axes.spines.right`: False
- `axes.grid`: True
- `grid.alpha`: 0.15
- `grid.linewidth`: 0.5
- `xtick.direction`: "out"
- `ytick.direction`: "out"
- `legend.frameon`: False
- `legend.borderpad`: 0.3
- `legend.handletextpad`: 0.5

## Annotation Style
- **NO boxed text.** Never use `bbox=dict(...)` on annotations, text boxes, or labels. No `FancyBboxPatch`, no `textprops=dict(bbox=...)`. Text should breathe — boxes add clutter.
- For callouts, use a simple arrow: `ax.annotate("label", xy=(...), fontsize=9, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))`
- For emphasis, use color or weight — not boxes or borders.

## Layout & Overlap Prevention
- **Always use `constrained_layout=True`** — never `tight_layout()` which often fails with complex grids.
- After generating any multi-panel figure, check for: overlapping titles/labels, clipped text, cramped legends.
- Use `fig.set_size_inches()` generously — too small is the #1 cause of overlap. Minimum 4 inches per subplot column, 3 inches per row.
- Rotate long tick labels: `ax.tick_params(axis='x', rotation=45)` with `ha='right'`
- Place legends outside the plot area when they overlap data: `ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))`
- Add `hspace` and `wspace` padding if `constrained_layout` isn't enough: `fig.subplots_adjust(hspace=0.35, wspace=0.3)`
- Suptitle needs room: `fig.suptitle(..., y=1.02)` to avoid colliding with subplot titles.

## Figure Setup
- Default DPI: 150 for research logs, 300 for publication/Issue embeds
- `figure.facecolor` and `savefig.facecolor`: "white"

## Color Palette

| Use case | Palette | Why |
|----------|---------|-----|
| Density/heatmaps (sequential) | `cmap='viridis'` | Perceptually uniform, colorblind-safe, grayscale-friendly |
| Density on light background | `cmap='inferno'` or `cmap='magma'` | Higher contrast for sparse data |
| 2D density with white background | `cmap='Blues'` or `cmap='GnBu'` | Subtle, doesn't fight the axes |
| Diverging (positive/negative) | `cmap='RdBu_r'` | Red=hot, blue=cold, white=zero — universal |
| Discrete methods (≤6) | `['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']` | Seaborn muted — distinguishable, not garish |
| Discrete methods (7-10) | `tab10` | Matplotlib standard, acceptable |
| Confidence bands | Same color as line, `alpha=0.15` | Visually linked to the curve |
| Baseline/reference | `color='#888888'`, `linestyle='--'` | Gray dashed — recedes visually |

**Avoid:** `jet`, `rainbow`, `hot`, `hsv` — perceptually misleading. `tab20` for ≤10 items — too many similar hues. Bright primary colors (`'r'`, `'b'`, `'g'`) — prefer muted tones.

**Consistency rule:** Pick method colors once per orbit, reuse across ALL panels:
```python
COLORS = {'baseline': '#888888', 'method_a': '#4C72B0', 'method_b': '#DD8452'}
```

## Output
- Save with `bbox_inches='tight'`, `facecolor='white'`
- Always `plt.close(fig)` after saving to free memory

## Standard rcParams Block

```python
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
```

## Comparison Panels

When comparing methods or experiments side-by-side:
- Always use `sharey=True` (and `sharex=True` where appropriate) so the reader's eye can scan across panels at the same scale.
- Set axis limits explicitly from the combined data range across ALL panels, not per-panel.
- Add panel labels: `ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')`
- Use identical bin edges, colormaps, and color scales across all panels in a comparison.

## Decluttering & Density Management

- **Too many points:** >1K points → scale alpha (`alpha = min(1.0, 500/N)`), use `rasterized=True`. >10K points → switch to hexbin or KDE, never raw scatter.
- **Too many lines:** >5 methods on one axis → split into small-multiples grid (one method per panel, shared axes, thin gray reference lines for others). Spaghetti plots are never acceptable.
- **Legends:** ≤3 entries → inside plot. 4-6 entries → outside right (`bbox_to_anchor=(1.02, 1)`). >6 → label lines directly at endpoints with `ax.text()`, no legend box.
- **Figure sizing:** Minimum 4 inches per column, 3.5 inches per row. If cramped, make it bigger — never shrink content to fit a figure size.
- **Tick density:** Use `ax.locator_params(nbins=5)` on crowded axes. Rotate long labels 45 degrees.
- **Colorbars:** Use `fraction=0.046, pad=0.04` to prevent colorbars from eating subplot space.
- **Grid:** Turn OFF grid for already-dense plots (scatter, heatmap, density). Grid is for line plots and bar charts.
- **Read the PNG.** After saving, use the Read tool to view the image. If any text overlaps, any legend covers data, or any region feels cramped — fix it before committing.

## Histogram & Density

Never use matplotlib's default `bins=10`. Use `bins='auto'` minimum, or `bins='fd'` (Freedman-Diaconis) for continuous data. For distribution comparisons, prefer KDE overlays (`sns.kdeplot` or `scipy.stats.gaussian_kde`) over raw histograms. When comparing distributions side-by-side, use identical bin edges and x/y limits across all panels.

## System & Schematic Diagrams

Every research problem has a natural visual representation — find it and produce it. The agent must identify what "seeing the system" means for their specific problem. A computation without visualization is an unverified claim.

**Schematic style:** Clean and minimal — thin lines (`linewidth=1`), open arrows, no heavy borders or filled boxes. Use `facecolor='none'` or very light fills (`alpha=0.05`). Mermaid for flowcharts. The goal is elegance, not PowerPoint. If a schematic looks like a corporate org chart, redesign it.

## Recommended Plot Types

Beyond basic line/bar charts, use these for richer insight:

- **Empirical 2D density** — `ax.hexbin(x, y, gridsize=30, cmap='GnBu')` or `sns.kdeplot(x=x, y=y, fill=True)`. Great for showing where the search concentrates in parameter space.
- **Trajectory overlay** — plot optimization paths through 2D solution space. Use `alpha=0.3` for individual runs, bold line for the best. Shows exploration vs exploitation.
- **Violin/box comparison** — `ax.violinplot()` for metric distributions across strategies. More informative than bar charts with error bars.
- **Heatmap** — `ax.imshow()` or `sns.heatmap()` for parameter sweep grids or correlation matrices.
- **Convergence with confidence bands** — `ax.fill_between(x, mean-std, mean+std, alpha=0.2)` when you have multiple runs.

Combine 2-4 of these as subplots in a single figure. One dense multi-panel PNG is always preferred over multiple separate files.

When writing orbit logs:
- Lead with the result, not the method.
- Include one key figure per orbit (multi-panel is fine — it still counts as one figure).
- Keep it under 500 words unless warranted.
