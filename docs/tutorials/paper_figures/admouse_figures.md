# AD mouse figures

The [AD population tutorial](../dataset_workflows/admouse.ipynb) starts from the
trained model and generates new populations. This page continues with the
gene-program, ligand–receptor, perturbation, and NicheNet analyses. It also
shows how to redraw S26 from the saved simulation used in the paper.

Run these commands from the source checkout described in
[Installation](../../installation.md). Each command writes new tables and
plots under `outputs/`.

## Cell populations: Figure 6b and S26

Extract `admouse_population_data.zip` from [Data and models](../../data_checkpoints.md).
It contains the saved cell states and classifier labels at 25 model times.

```bash
python reproduction/admouse/plot_population.py \
  --run-dir data/admouse/populations \
  --output-dir outputs/admouse_population
```

This counts the labels at each time, calculates cell-type proportions, and
draws the spatial populations from the cell coordinates. The output includes
`celltype_counts_and_proportions.csv` and `ad_supplementary1.pdf`.
To draw the selected ages in Figure 6b from the same arrays:

```bash
python reproduction/admouse/draw_figures.py \
  --data-dir data/admouse --panels b --output-dir outputs/admouse_main
```

## Gene programs and LR time courses: Figure 6c–e

The numerical inputs for these panels are included in
`reproduction/admouse/final_figures/main/data/`. The gene table contains
347 microglial gene profiles. The LR table contains scores across model times.

```bash
python reproduction/admouse/draw_figures.py \
  --panels cd e --output-dir outputs/admouse_programs
```

For panels c–d, the command clusters the gene profiles using weighted
hierarchical linkage, assigns four programs, and calculates their mean
curves. For panel e, it calculates a z-score within each LR pair before
drawing the time courses.

The output includes the heatmap, program curves, LR plot, and these tables:

```text
outputs/admouse_programs/tables/
├── gene_cluster_assignments_weighted.csv
├── gene_cluster_prototypes_weighted.csv
├── pattern_1_genes.csv
├── pattern_2_genes.csv
├── pattern_3_genes.csv
├── pattern_4_genes.csv
├── lr_scores.csv
└── lr_zscores.csv
```

## GO enrichment: S27–S28

Use the program 1 and program 4 gene lists produced above. These analyses
run in R, separately from the Python environment. They were tested with
R 4.3, clusterProfiler 4.10, enrichplot 1.22, and org.Mm.eg.db 3.18.
The plots also need ggupset. Install the dependencies once:

```r
install.packages(c("BiocManager", "ggupset"))
BiocManager::install(version = "3.18")
BiocManager::install(c("clusterProfiler", "enrichplot", "org.Mm.eg.db"))
```

Then run each analysis:

```bash
Rscript reproduction/admouse/go/enrich_program.R \
  outputs/admouse_programs/tables/pattern_1_genes.csv \
  outputs/admouse_GO_program1
```

```bash
Rscript reproduction/admouse/go/enrich_program.R \
  outputs/admouse_programs/tables/pattern_4_genes.csv \
  outputs/admouse_GO_program4
```

Each directory contains gene-ID mappings, the enrichment table, the fitted
enrichment object, and five plots. `R_packages.txt` records the package and
annotation versions. Changing the annotation database can change the GO results.

## Perturbation panels: Figure 6f–g and S30

Extract `admouse_perturbation_data.zip`. It contains the saved Trem2
perturbation states, attention edges, and module scores.

```bash
python reproduction/admouse/draw_figures.py \
  --data-dir data/admouse --panels f g --output-dir outputs/admouse_trem2
```

The command draws the spatial comparison, recalculates cell-type composition
from the saved labels, and plots the module-score changes. For S30, the Spp1
module-score table is already included with the code:

```bash
python reproduction/admouse/draw_figures.py \
  --panels s30 --output-dir outputs/admouse_spp1
```

These commands draw the results of saved perturbation simulations. They do
not run a new perturbation experiment.

## NicheNet analysis: S29

Extract `admouse_analysis_data.zip` and `admouse_nichenet_data.zip`.
The latter contains 51 saved cell populations, NicheNet reference networks,
and the LR-pair list. Install the R dependencies once:

```r
install.packages(c("dplyr", "tidyr", "tibble", "purrr", "magrittr", "ROCR",
                  "caTools", "Hmisc", "ggplot2", "circlize", "png"))
```

First reconstruct gene expression from the saved PCA states and prepare the
microglial gene sets for the 50 adjacent time intervals:

```bash
python reproduction/admouse/nichenet/prepare.py \
  --data-dir data/admouse --output-dir outputs/admouse_nichenet
```

This writes expression summaries and the input gene lists into
`outputs/admouse_nichenet/data/`. Use those files to calculate ligand activity:

```bash
Rscript reproduction/admouse/nichenet/score.R \
  outputs/admouse_nichenet outputs/admouse_nichenet data/admouse/nichenet
```

The scores are written to `outputs/admouse_nichenet/results/`.
Use that same directory for the activity plot and the six LR-network panels:

```bash
Rscript reproduction/admouse/nichenet/plot_activity.R \
  outputs/admouse_nichenet outputs/admouse_nichenet
```

```bash
Rscript reproduction/admouse/nichenet/plot_links.R \
  outputs/admouse_nichenet outputs/admouse_nichenet data/admouse/nichenet
```

The new plots are in `outputs/admouse_nichenet/figures/`, with their numerical
tables in `data/`. The scoring and network functions are included under the
NicheNet license in `reproduction/admouse/nichenet/nichenetr_R/`.

To generate a new set of 51 populations, `nichenet/interpolate.py` accepts
`--data-dir` and `--output-dir`. Pass its `slice_data/` output to
`prepare.py --states`. The 0.05 spacing here is the simulation output interval.
The population and NicheNet analyses use different output grids.
