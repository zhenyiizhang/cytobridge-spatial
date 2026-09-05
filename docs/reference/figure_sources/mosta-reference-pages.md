---
orphan: true
---

# MOSTA S11–S18: inputs and plotting code

The [MOSTA supplementary notebook](../../tutorials/paper_figures/mosta_figures.ipynb)
draws all eight figures from numerical cell states and analysis tables.

## Download and draw

```bash
python -m CytoBridge.datasets mosta --kind mosta_figure_data.zip --output-dir .
python -m reproduction.mosta.figures --data-dir data/mosta/paper --output-dir outputs/mosta_si
```

The first command downloads the saved paper states. The second writes the
figures and recalculated summary tables.

| Figure | Input under `data/mosta/paper/shared/` | Calculation |
| --- | --- | --- |
| S11 | `s4/observed_t0.h5ad`, `generated_states/` | Draw the spatial populations. |
| S12 | `s5_growth/growth_by_cell_fully_generated.csv` | Select brain cells and draw growth on a shared colour scale. |
| S13 | `s6_composition/celltype_composition_fully_generated.csv` | Calculate counts and proportions. |
| S14 | `s7_lineage/fixed_particle_labels.csv.gz` | Count transitions between the same simulated particles. |
| S15–S17 | Gene profiles, program assignments and GO tables | Draw expression programs and enrichment results. |
| S18 | LR score tables in the source archive | Normalize sampled scores and interpolate the displayed time courses. |

The plotting functions are in `reproduction/mosta/figures.py`. Smaller gene
and LR tables are included in the source archive. To generate populations
from a model first, follow the [MOSTA analysis tutorial](../../tutorials/dataset_workflows/mosta.ipynb).
