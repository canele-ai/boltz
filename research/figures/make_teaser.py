"""
Teaser figure: Boltz Inference Speedup — Where the Time Goes
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Data — relative wall-clock fractions (must sum to 1 within each scenario)
# ---------------------------------------------------------------------------
stages = [
    "MSA\nProcessing",
    "Trunk /\nPairformer",
    "Diffusion\nSampling",
    "Confidence\nHead",
    "Output /\nPostproc",
]

# Baseline fractions (rough but illustrative)
baseline = [0.05, 0.18, 0.68, 0.06, 0.03]

# "Goal" scenario: diffusion 10x fewer steps → ~7x wall-clock reduction on that stage;
# trunk ~2x faster via compile + flash-attn; others unchanged.
# Raw unscaled: [0.05, 0.09, 0.10, 0.06, 0.03]  → total 0.33
raw_goal = [0.05, 0.09, 0.10, 0.06, 0.03]
total_goal = sum(raw_goal)
goal = [v / total_goal for v in raw_goal]

colors = [
    "#4C72B0",   # blue  — MSA / preprocessing
    "#DD8452",   # orange — trunk
    "#C44E52",   # red    — diffusion
    "#55A868",   # green  — confidence
    "#8172B2",   # purple — output
]

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
BAR_HEIGHT = 0.55
Y_POS = 0.0          # single row: y = 0
FONTSIZE = 14
TITLE_FONTSIZE = 16

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

def draw_pipeline(ax, fractions, title, annotations=None):
    """
    Draw stacked horizontal bar.
    annotations: list of (stage_index, label_text) to annotate with an arrow.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.80, 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Draw stacked bar segments
    left = 0.0
    centers = []
    for i, (frac, color) in enumerate(zip(fractions, colors)):
        rect = mpatches.FancyBboxPatch(
            (left, Y_POS - BAR_HEIGHT / 2),
            frac,
            BAR_HEIGHT,
            boxstyle="round,pad=0.008",
            linewidth=1.2,
            edgecolor="white",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(rect)
        cx = left + frac / 2
        centers.append(cx)

        # Stage label inside bar (only if wide enough)
        if frac > 0.06:
            ax.text(
                cx, Y_POS,
                stages[i],
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold",
                zorder=3,
            )
        left += frac

    # Percentage labels above each segment
    left = 0.0
    for i, frac in enumerate(fractions):
        cx = left + frac / 2
        pct = f"{frac * 100:.0f}%"
        ax.text(
            cx, Y_POS + BAR_HEIGHT / 2 + 0.07,
            pct,
            ha="center", va="bottom",
            fontsize=10, color="0.3",
        )
        left += frac

    # Annotations (arrows pointing to stages)
    if annotations:
        for stage_idx, label, y_offset_sign in annotations:
            cx = centers[stage_idx]
            y_tip = Y_POS + y_offset_sign * (BAR_HEIGHT / 2 + 0.04)
            y_text = Y_POS + y_offset_sign * (BAR_HEIGHT / 2 + 0.38)
            ax.annotate(
                label,
                xy=(cx, y_tip),
                xytext=(cx, y_text),
                fontsize=10,
                ha="center",
                va="center",
                color="#1a1a1a",
                fontweight="semibold",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#1a1a1a",
                    lw=1.4,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightyellow",
                    edgecolor="#bbbb88",
                    linewidth=1,
                    alpha=0.9,
                ),
                zorder=5,
            )

    # "Faster →" x-axis label
    ax.set_xlabel("Relative inference time  →", fontsize=FONTSIZE - 1, color="0.4",
                  labelpad=6)

# ---------------------------------------------------------------------------
# LEFT panel: baseline
# ---------------------------------------------------------------------------
draw_pipeline(
    axes[0],
    baseline,
    "Current Boltz Inference",
    annotations=None,
)

# Add a subtle time-axis arrow at the bottom
axes[0].annotate(
    "", xy=(1.0, -0.57), xytext=(0.0, -0.57),
    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.5),
    annotation_clip=False,
)
axes[0].text(0.5, -0.65, "100% wall-clock time", ha="center", va="top",
             fontsize=11, color="0.5", style="italic")

# ---------------------------------------------------------------------------
# RIGHT panel: goal with annotations
# ---------------------------------------------------------------------------
# Annotations: (stage_index, text, y_sign)
#  +1 = above bar, -1 = below bar
annotations_goal = [
    (1, "torch.compile\n+ Flash Attention",  +1),
    (2, "10x fewer\ndenoising steps",        -1),
]
draw_pipeline(
    axes[1],
    goal,
    "Target: Fast Boltz Inference",
    annotations=annotations_goal,
)

# Speedup badge
total_speedup = 1.0 / total_goal
axes[1].text(
    0.98, 0.82,
    f"~{total_speedup:.1f}x\nspeedup",
    transform=axes[1].transAxes,
    ha="right", va="top",
    fontsize=15, fontweight="bold",
    color="#C44E52",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff0f0",
              edgecolor="#C44E52", linewidth=1.8),
)

# Time-axis arrow
axes[1].annotate(
    "", xy=(1.0, -0.57), xytext=(0.0, -0.57),
    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.5),
    annotation_clip=False,
)
axes[1].text(0.5, -0.65,
             f"{total_goal * 100:.0f}% of original wall-clock time",
             ha="center", va="top",
             fontsize=11, color="0.5", style="italic")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=colors[0], label="MSA / Preprocessing"),
    mpatches.Patch(facecolor=colors[1], label="Trunk / Pairformer"),
    mpatches.Patch(facecolor=colors[2], label="Diffusion Sampling"),
    mpatches.Patch(facecolor=colors[3], label="Confidence Head"),
    mpatches.Patch(facecolor=colors[4], label="Output / Postproc"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

# ---------------------------------------------------------------------------
# Super-title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Boltz Inference Speedup: Where the Time Goes",
    fontsize=TITLE_FONTSIZE + 2,
    fontweight="bold",
    y=1.04,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/Users/liambai/code/boltz/research/figures/teaser.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
