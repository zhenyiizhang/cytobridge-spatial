#!/usr/bin/env python3
"""Draw the whole-tissue scale-2.5 Spp1 module response in panel-g style."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ad_supplementary2/whole_tissue_spp1_modules_endpoint.csv"
DATA = ROOT / "data/ad_supplementary2/spp1_whole_tissue_scale2p5_r2_module_contrasts_t2p5.csv"
OUTPUT = ROOT / "figures/ad_supplementary2"
DISPLAY_TIME = 2.5
MODULES = ["Myelination_Oligo", "Endothelial_BBB", "SPP1_CD44_axis",
           "DAM_microglia", "Antigen_Presentation_MHCII"]
LABELS = ["Myelination/\noligo.", "Endothelial/\nBBB", "SPP1-CD44",
          "DAM", "Antigen\npresentation"]

mpl.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 13, "axes.linewidth": 1.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

table = pd.read_csv(SOURCE)
selected = table[table.perturbed_gene.eq("Spp1") & np.isclose(table.time, DISPLAY_TIME)
                 & table.population.eq("all_particles") & table.module.isin(MODULES)].copy()
selected.to_csv(DATA, index=False)
activation = selected[selected.direction.eq("high")].set_index("module").reindex(MODULES).delta.to_numpy(float)
knockdown = selected[selected.direction.eq("low")].set_index("module").reindex(MODULES).delta.to_numpy(float)
if np.isnan(activation).any() or np.isnan(knockdown).any():
    raise ValueError("Missing requested whole-tissue Spp1 module values")

x = np.arange(len(MODULES), dtype=float)
width = 0.36
fig, ax = plt.subplots(figsize=(11.08, 5.48), dpi=180)
ax.bar(x-width/2, activation, width, color="#B5667A", edgecolor="white", linewidth=.7,
       label="In silico Spp1 Activation", zorder=3)
ax.bar(x+width/2, knockdown, width, color="#9EBED0", edgecolor="white", linewidth=.7,
       label="In silico Spp1 Knockdown", zorder=3)
ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=4)
for value in (-.5, .5): ax.axhline(value, color="#D9D9D9", linewidth=2, zorder=1)
maximum = max(float(np.max(np.abs(activation))), float(np.max(np.abs(knockdown))))
axis_limit = max(.82, np.ceil((maximum+.08)*10)/10)
ax.set(xlim=(-.5, len(MODULES)-.5), ylim=(-axis_limit, axis_limit), ylabel="Module Score Change")
ax.set_yticks([-.5, .5], ["−0.5", "0.5"])
ax.set_xticks(x, LABELS, rotation=48, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", direction="out", width=1.3, length=5)
for spine in ax.spines.values(): spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1.5)
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2, frameon=False,
          handlelength=1.5, handletextpad=.35, columnspacing=3, borderaxespad=0)
ax.text(-.095, 1.17, "g", transform=ax.transAxes, fontsize=22, fontweight="bold", ha="left", va="top")
ax.text(.99, 1.17, "Whole tissue | scale 2.5 | t=2.5 | formal r2", transform=ax.transAxes,
        fontsize=11, ha="right", va="top")
fig.subplots_adjust(left=.105, right=.985, bottom=.34, top=.79)
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
