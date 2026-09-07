# AD mouse figures

This page calculates the gene programs, ligand–receptor time courses and
perturbations in Figure 6, followed by GO and NicheNet analyses in S27–S30.
Start with the AD model and aligned data from [Data and models](../../data_checkpoints.md).
No other notebook needs to be run first.

Run these commands from the CytoBridge code folder described in
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

Generate 26 populations at model times 0–2.5, starting with the observed cells
at time zero. Cell types are assigned by the fitted classifier. For each time,
the calculation reconstructs expression from PCA, averages expression within
microglia, and standardizes each gene's time course. Genes with nonzero PCA
loadings are retained, giving 347 profiles with the paper inputs.

For LR scores, the model's cell-type communication weights are multiplied by
the corresponding sender-ligand and receiver-receptor expression. The command
writes both input tables needed by the plotting step below.

```bash
python -m reproduction.admouse.calculate_programs \
  --data-dir data/admouse --output-dir outputs/admouse_calculated --device cuda:0
```

The new populations are in `outputs/admouse_calculated/generated_states/`,
the gene profiles are in `gene_programs/`, and the LR scores are in
`lr_pair_timecourse.csv`. Draw Figure 6b–e from these new results:

```bash
python reproduction/admouse/draw_figures.py \
  --panels b cd e \
  --population-dir outputs/admouse_calculated/generated_states \
  --gene-input-dir outputs/admouse_calculated/gene_programs \
  --lr-input outputs/admouse_calculated/lr_pair_timecourse.csv \
  --output-dir outputs/admouse_programs
```

For panels c–d, the command clusters the gene profiles using weighted
hierarchical linkage, assigns four programs, and calculates their mean
curves. For panel e, it calculates a z-score within each LR pair before
drawing the time courses.

The output of the gene-program calculation above:

```{image} ../../_static/figures/admouse_calculated_gene_programs.png
:alt: Figure 6c–d gene profiles and temporal programs calculated from the AD model.
:width: 100%
```

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
enrichment object, and the four plot types in S27–S28: dot plot, enrichment
map, gene–term network, and UpSet plot. `R_packages.txt` records the package and
annotation versions. Changing the annotation database can change the GO results.

## Perturbation panels: Figure 6f–g and S30

Continue from `outputs/admouse_calculated/` above. These two commands change
the initial PCA states along the selected gene's loading, then simulate the
perturbed populations using the same model and random seed. The perturbation
scales are 1 for Trem2 and 2.5 for Spp1, as used in these panels.

```bash
python -m reproduction.admouse.perturbations \
  --data-dir data/admouse --baseline-dir outputs/admouse_calculated \
  --gene Trem2 --output-dir outputs/admouse_perturbations/trem2 --device cuda:0
```

```bash
python -m reproduction.admouse.perturbations \
  --data-dir data/admouse --baseline-dir outputs/admouse_calculated \
  --gene Spp1 --output-dir outputs/admouse_perturbations/spp1 --device cuda:0
```

Module scores use the gene sets in `reproduction/admouse/gene_sets.py`.
For each gene, expression is standardized with equally weighted moments of
the baseline and perturbed populations. The plot shows the mean score change
within each gene set. Draw the new results:

```bash
python reproduction/admouse/draw_figures.py \
  --data-dir outputs/admouse_perturbations --panels f g --output-dir outputs/admouse_trem2
```

The command draws the spatial comparison, recalculates cell-type composition
from the new labels, and plots the module-score changes. For S30:

```bash
python reproduction/admouse/draw_figures.py \
  --panels s30 --spp1-input outputs/admouse_perturbations/spp1/spp1_module_scores.csv \
  --output-dir outputs/admouse_spp1
```

To redraw the saved paper perturbation instead, extract
`admouse_perturbation_data.zip` and use `--data-dir data/admouse` for panels f–g.

S30 drawn from the Spp1 calculation above:

```{image} ../../_static/figures/admouse_calculated_spp1.png
:alt: S30 module-score changes calculated from the Spp1 perturbation.
:width: 100%
```

## NicheNet analysis: S29

Extract `admouse_analysis_data.zip` and `admouse_nichenet_data.zip`.
The latter contains 51 saved cell populations, NicheNet reference networks,
and the LR-pair list. Install the R dependencies once:

```r
install.packages(c("dplyr", "tidyr", "tibble", "purrr", "magrittr", "ROCR",
                  "caTools", "Hmisc", "ggplot2", "circlize", "png"))
```

First generate the 51 populations used for the 50 NicheNet time intervals:

```bash
python reproduction/admouse/nichenet/interpolate.py \
  --data-dir data/admouse --output-dir outputs/admouse_nichenet_populations --device cuda:0
```

Reconstruct gene expression from those populations and prepare the microglial
gene sets for adjacent intervals:

```bash
python reproduction/admouse/nichenet/prepare.py \
  --data-dir data/admouse --states outputs/admouse_nichenet_populations/slice_data \
  --output-dir outputs/admouse_nichenet
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

For the saved paper NicheNet populations, use `data/admouse/nichenet/slice_data`
as `--states` and omit the interpolation command. The 0.05 spacing is the
simulation output interval for NicheNet. Figure 6 uses a 0.1 interval.
