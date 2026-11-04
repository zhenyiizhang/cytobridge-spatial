#!/usr/bin/env python3
"""Draw AD population panels from saved cell states and cell-type labels."""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True,
                    help="Directory containing compat_base and whole_tissue")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--png-dpi", type=int, default=300)
args = parser.parse_args()
ROOT = args.run_dir.resolve()
STATE_DIR = ROOT / "compat_base/01_interpolation"
LABEL_DIR = ROOT / "whole_tissue/baseline_labels_k1"
OUT = args.output_dir.resolve()
if OUT == ROOT or ROOT in OUT.parents or STATE_DIR in OUT.parents or LABEL_DIR in OUT.parents:
    parser.error("Choose an output directory outside the source run.")
if args.png_dpi < 72:
    parser.error("--png-dpi must be at least 72")
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "ad_supplementary1"
CSV_OUT = OUT / "celltype_counts_and_proportions.csv"

ORDER = ["Astrocytes", "Excitatory neurons", "Fibroblast", "Inhibitory neurons",
         "Microglia", "OPC", "Oligodendrocytes", "Pericytes/Endothelial"]
COLORS = {"Astrocytes":"#1f77b4", "Excitatory neurons":"#ff7f0e",
          "Fibroblast":"#2ca02c", "Inhibitory neurons":"#d62728",
          "Microglia":"#9467bd", "OPC":"#8c564b",
          "Oligodendrocytes":"#e377c2", "Pericytes/Endothelial":"#7f7f7f"}
DISPLAY = {"Astrocytes":"Astrocytes", "Excitatory neurons":"Excitatory Neurons",
           "Fibroblast":"Fibroblast", "Inhibitory neurons":"Inhibitory Neurons",
           "Microglia":"Microglia", "OPC":"OPC",
           "Oligodendrocytes":"Oligodendrocytes",
           "Pericytes/Endothelial":"Pericytes/Endothelial"}
TIMES = np.round(np.arange(0, 2.41, .1), 1)  # first 25 saved nodes only
# Match the standalone true-3D notebook-style panel a exactly.
LAYERS = [0.0, 0.4, 0.7, 1.0, 1.2, 1.8, 2.0, 2.2]
OBSERVED_LAYERS = {0.0, 1.0, 2.0}

mpl.rcParams.update({"font.family":"Arial", "font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":9, "axes.linewidth":1.0, "pdf.fonttype":42,
                     "ps.fonttype":42, "figure.facecolor":"white", "savefig.facecolor":"white"})

def token(t): return f"{t:g}"
def load(t):
    s = np.load(STATE_DIR / f"generated_t{token(t)}.npy")
    lab = np.load(LABEL_DIR / f"labels_t{token(t)}.npy").astype(str)
    if len(s) != len(lab): raise ValueError(f"length mismatch at t={t}")
    return s[:, :2], lab

counts = np.zeros((len(TIMES), len(ORDER)), dtype=int)
for i, t in enumerate(TIMES):
    _, labels = load(float(t))
    for j, ct in enumerate(ORDER): counts[i, j] = np.count_nonzero(labels == ct)
props = counts / counts.sum(axis=1, keepdims=True) * 100

with CSV_OUT.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["model_time", "status", "celltype", "count", "proportion_percent", "total"])
    for i, t in enumerate(TIMES):
        for j, ct in enumerate(ORDER):
            writer.writerow([f"{t:.1f}", "extrapolated" if t > 2 else "generated",
                             ct, counts[i,j], props[i,j], counts[i].sum()])

fig = plt.figure(figsize=(11.4, 6.2), dpi=220)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1],
                      left=.045, right=.985, bottom=.16, top=.93, wspace=.16, hspace=.42)
ax3 = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
axp = fig.add_subplot(gs[0, 1])
axn = fig.add_subplot(gs[1, 1])

# a: true orthographic Figure3D-notebook style layers. All slices retain their
# per-particle colours; panel b/c remain the original 25-node summaries.
layer_data = [(t, *load(t)) for t in LAYERS]
all_x = np.concatenate([xy[:, 0] for _, xy, _ in layer_data])
all_y = np.concatenate([xy[:, 1] for _, xy, _ in layer_data])
xmin, xmax = np.quantile(all_x, [.003, .997])
ymin, ymax = np.quantile(all_y, [.003, .997])
def normalize(xy):
    q = xy.copy()
    q[:, 0] = (q[:, 0] - xmin) / (xmax - xmin)
    q[:, 1] = (q[:, 1] - ymin) / (ymax - ymin)
    return q

frame_min, frame_max = -.20, 1.10
z_spacing, point_size = 6.5, 1.15
for i, (t, xy, labels) in enumerate(layer_data):
    xy, z = normalize(xy), i * z_spacing
    opacity = 1.0 if t in OBSERVED_LAYERS else .60
    line_style = "-" if t in OBSERVED_LAYERS else (0, (6, 5))
    line_alpha, base_zorder = (.8 if t in OBSERVED_LAYERS else .6), 20 + 4 * i
    ax3.plot([frame_min, frame_min], [frame_min, frame_max], zs=z, color="black", lw=.8,
             ls=line_style, alpha=line_alpha, zorder=base_zorder)
    ax3.plot([frame_min, frame_max], [frame_min, frame_min], zs=z, color="black", lw=.8,
             ls=line_style, alpha=line_alpha, zorder=base_zorder)
    point_colors = np.asarray([COLORS.get(label, "#888888") for label in labels])
    ax3.scatter(xy[:, 0], xy[:, 1], zs=z, zdir="z", s=point_size, c=point_colors,
                alpha=opacity, linewidths=0, depthshade=False, rasterized=True,
                zorder=base_zorder + 1)
    ax3.plot([frame_max, frame_max], [frame_min, frame_max], zs=z, color="black", lw=.8,
             ls=line_style, alpha=line_alpha, zorder=base_zorder + 2)
    ax3.plot([frame_min, frame_max], [frame_max, frame_max], zs=z, color="black", lw=.8,
             ls=line_style, alpha=line_alpha, zorder=base_zorder + 2)
ax3.set_xlim(frame_min, frame_max); ax3.set_ylim(frame_min, frame_max)
ax3.set_zticks(np.arange(len(LAYERS)) * z_spacing, [f"{t:.1f}" for t in LAYERS])
ax3.set_zlabel("Model time", labelpad=8)
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_proj_type("ortho")
ax3.set_box_aspect((1.5, 1.0, 1.5)); ax3.view_init(elev=26, azim=34)
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis): axis.pane.fill=False; axis.pane.set_edgecolor("white")
ax3.grid(False); ax3.text2D(-.02, 1.0, "a", transform=ax3.transAxes, fontsize=16, fontweight="bold")
stage_handles = [Line2D([0], [0], color="black", lw=.9, ls="-", label="Observed stage"),
                 Line2D([0], [0], color="black", lw=.9, ls=(0, (6, 5)), label="Generated")]
stage_legend = ax3.legend(handles=stage_handles, loc="lower right", bbox_to_anchor=(1.0, .02),
                          frameon=False, fontsize=8, handlelength=2.8)
ax3.add_artist(stage_legend)

# b: baseline cell proportions at every saved time.
x = np.arange(len(TIMES)); bottom = np.zeros(len(TIMES))
for j, ct in enumerate(ORDER):
    axp.bar(x, props[:,j], bottom=bottom, width=.78, color=COLORS[ct],
            edgecolor="white", linewidth=.25)
    bottom += props[:,j]
axp.set_ylim(0,100); axp.set_xlim(-.7,len(TIMES)-.3)
ticks=[0,5,10,15,20,24]; axp.set_xticks(ticks,[f"{TIMES[k]:g}" for k in ticks])
axp.set_ylabel("Cell Proportion (%)"); axp.set_xlabel("Model time")
axp.text(22.5,103,"Extrapolated",ha="center",va="bottom",fontsize=8)
axp.spines[["top","right"]].set_visible(False); axp.tick_params(direction="out",length=3)
axp.text(-.08,1.05,"b",transform=axp.transAxes,fontsize=14,fontweight="bold")

# c: baseline cell numbers at every saved time.
for j, ct in enumerate(ORDER): axn.plot(TIMES, counts[:,j], color=COLORS[ct], lw=2.2)
axn.set_xlim(0,2.4); axn.set_ylim(bottom=0); axn.set_xlabel("Model time"); axn.set_ylabel("Cell Number")
axn.spines[["top","right"]].set_visible(False); axn.tick_params(direction="out",length=3)
axn.text(-.08,1.05,"c",transform=axn.transAxes,fontsize=14,fontweight="bold")

handles=[Line2D([0],[0],marker="o",ls="none",ms=7,markerfacecolor=COLORS[c],
                markeredgecolor="none",label=DISPLAY[c]) for c in ORDER]
fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.055,.015),ncol=4,
           frameon=False,columnspacing=1.4,handletextpad=.35,fontsize=9)
# Dense spatial layers are rasterized. Curves, bars and text remain vector.
fig.savefig(STEM.with_suffix(".png"),dpi=args.png_dpi,bbox_inches="tight")
fig.savefig(STEM.with_suffix(".pdf"),dpi=args.png_dpi,bbox_inches="tight")
plt.close(fig)
