"""Generate comparison figure for deadend-kernels evaluation."""
import matplotlib
matplotlib.use("Agg")
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
    "Control": "#4C72B0",
    "BMM": "#DD8452",
    "Sim INT8": "#55A868",
}

complexes = ["small\n(1BRS)", "medium\n(1DQJ)", "large\n(2DN2)"]

# predict_only_s data (GPU-only inference time)
control_predict = {
    "small": [5.461, 4.527, 5.397],
    "medium": [14.780, 13.903, 13.874],
    "large": [28.131, 26.460, 25.462],
}
bmm_predict = {
    "small": [4.852, 5.553, 4.929],
    "medium": [20.133, 22.893, 22.725],
    "large": [39.737, 39.654, 39.440],
}
sim_int8_predict = {
    "small": [5.352, 4.860, 5.457],
    "medium": [14.043, 13.553, 14.258],
    "large": [28.143, 26.829, 26.157],
}

# pLDDT data
control_plddt = {
    "small": [0.8623, 0.8411, 0.8535],
    "medium": [0.4904, 0.4796, 0.4797],
    "large": [0.8103, 0.8096, 0.8082],
}
bmm_plddt = {
    "small": [0.8620, 0.8411, 0.8536],
    "medium": [0.4896, 0.4800, 0.4789],
    "large": [0.8083, 0.8085, 0.8090],
}
sim_int8_plddt = {
    "small": [0.8609, 0.8347, 0.8614],
    "medium": [0.4899, 0.4796, 0.4822],
    "large": [0.8122, 0.8087, 0.8089],
}

# CA RMSD data
control_rmsd = {
    "small": [5.498, 5.534, 5.635],
    "medium": [25.899, 25.651, 25.908],
    "large": [20.047, 20.690, 22.316],
}
bmm_rmsd = {
    "small": [5.502, 5.534, 5.633],
    "medium": [25.923, 25.649, 25.929],
    "large": [20.069, 20.691, 22.300],
}
sim_int8_rmsd = {
    "small": [5.501, 5.526, 5.604],
    "medium": [25.895, 25.663, 25.987],
    "large": [19.871, 20.710, 22.390],
}


def get_means_stds(data):
    means = [np.mean(data[k]) for k in ["small", "medium", "large"]]
    stds = [np.std(data[k]) for k in ["small", "medium", "large"]]
    return means, stds


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Predict-only time
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")
x = np.arange(3)
w = 0.25

for i, (name, data) in enumerate([
    ("Control", control_predict),
    ("BMM", bmm_predict),
    ("Sim INT8", sim_int8_predict),
]):
    means, stds = get_means_stds(data)
    bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=name,
                  color=COLORS[name], capsize=3, edgecolor="white", linewidth=0.5)

ax.set_ylabel("Predict-only time (s)")
ax.set_title("Inference Time (GPU only)")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.legend(loc="upper left")

# Panel (b): pLDDT
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

for i, (name, data) in enumerate([
    ("Control", control_plddt),
    ("BMM", bmm_plddt),
    ("Sim INT8", sim_int8_plddt),
]):
    means, stds = get_means_stds(data)
    bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=name,
                  color=COLORS[name], capsize=3, edgecolor="white", linewidth=0.5)

ax.set_ylabel("pLDDT")
ax.set_title("Structure Quality")
ax.set_xticks(x)
ax.set_xticklabels(complexes)
ax.set_ylim(0.3, 1.0)

# Panel (c): CA RMSD
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

for i, (name, data) in enumerate([
    ("Control", control_rmsd),
    ("BMM", bmm_rmsd),
    ("Sim INT8", sim_int8_rmsd),
]):
    means, stds = get_means_stds(data)
    bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=name,
                  color=COLORS[name], capsize=3, edgecolor="white", linewidth=0.5)

ax.set_ylabel("CA RMSD (A)")
ax.set_title("Structural Accuracy vs Ground Truth")
ax.set_xticks(x)
ax.set_xticklabels(complexes)

fig.suptitle("Dead-End Approaches: BMM Triangle Mult & Simulated INT8 vs Control",
             fontsize=13, fontweight="medium", y=1.02)

fig.savefig("/home/liambai/code/boltz/.worktrees/deadend-kernels/orbits/deadend-kernels/figures/results.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure saved.")
