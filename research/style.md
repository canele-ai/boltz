# Plotting Style Guide

## Layout

- **Always multi-panel figures.** Never save a single isolated plot. Combine related views: `fig, axes = plt.subplots(1, N, figsize=(5*N, 4.5), constrained_layout=True)`.
- **`constrained_layout=True` always.** Never use `tight_layout()` — it fails on complex panels.
- **Minimum 5 inches per panel.** 3 panels = `figsize=(15, 4.5)`.
- **Panel labels `(a)`, `(b)`, `(c)`** via `ax.set_title('(a) ...')`.
- **`savefig(bbox_inches='tight', pad_inches=0.2)`** to prevent clipping.

## Typography

- Base font size: **14pt** (`mpl.rcParams['font.size'] = 14`). Never below 12pt for any text.
- Title: 16pt bold. Axis labels: 14pt. Tick labels: 12pt. Legend: 11pt.
- **No overlapping text.** If title + subtitle collide, use `fig.suptitle(..., y=1.02)` to lift it above panel titles.
- Captions below axes use `fig.text(x, -0.04, ...)`, not text inside the axes area.

## Colors

- **Baseline (200-step default)**: `#1f77b4` (blue)
- **Step-reduced variants**: `#ff7f0e` (orange) — lighter shades for more steps, darker for fewer
- **Compile/TF32 optimizations**: `#2ca02c` (green family)
- **Flash Attention variants**: `#9467bd` (purple)
- **Combined optimizations**: `#d62728` (red) — for the "best combined" configuration
- **Failed / below quality gate**: `#7f7f7f` (gray, dashed lines)
- Use the `tab10` colormap for >5 methods. Never use red-vs-green without shape/pattern distinction (colorblind safety).
- Error regions: `fill_between(..., alpha=0.2)`

## Axes

- **Shared axes on comparison panels:** `sharey=True` / `sharex=True` when comparing the same quantity.
- Remove top and right spines: `ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)`
- Log scale for timing plots when comparing across orders of magnitude.
- **X-axis**: sampling_steps or configuration name. **Y-axis**: wall-clock time (s) or pLDDT.
- Always label units: "Wall time (s)", "pLDDT", "Speedup (x)".

## Legends

- ≤3 entries: inside the plot, `frameon=False`
- >3 entries: below the figure, `fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=N, frameon=False)`
- Never let the legend overlap data points.

## Saving

```python
fig.savefig('orbits/<name>/figures/results.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
```
