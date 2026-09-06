# Zebrafish populations, growth and gene dynamics

This tutorial recalculates S31, S32, S35 and S38 from the trained model and
aligned cells. It first generates the numerical results, then draws the paper
figures. The dynamical model is not retrained.

## Download the model and data

From the [source checkout](../../installation.md), run:

```bash
python -m CytoBridge.datasets zebrafish --kind analysis --output-dir .
```

This creates `data/zebrafish/aligned.h5ad`, `data/zebrafish/model/` and the
edge classifier. The aligned file contains expression features, coordinates,
time points and cell-type annotations.

## Calculate populations and growth

```bash
python scripts/run_zebrafish_paper_downstream.py \
  --aligned-h5ad data/zebrafish/aligned.h5ad \
  --model-dir data/zebrafish/model \
  --output-dir outputs/zebrafish_analysis \
  --shared-cache-dir outputs/zebrafish_analysis/classifier_cache \
  --stage s22,growth \
  --video-formats none \
  --device cuda:0
```

`s22` is the analysis program's original name for the population calculation
now shown in S31. It starts with the 563 observed cells at time zero and
simulates through time four. This particular analysis keeps the population
size fixed. The learned drift, interactions, score and diffusion remain active.
S32 evaluates the growth head on the observed cells at each measured stage.

The command uses `cb.tl.run_interpolation_workflow` for the simulation and
`cb.tl.evaluate_growth_by_timepoint` for growth. A cell-type classifier is fitted
on the first run and reused from `classifier_cache` on subsequent runs.

The results needed for plotting are:

- `s22/observed_reference_states/`: observed coordinates and cell types.
- `s22/global_t0_fixed_population_states/`: newly simulated coordinates and predicted cell types.
- `growth/growth_per_cell.csv`: raw growth values evaluated by the model.

Each state folder has an `index.json` listing the time points and their NPZ files.

## Draw S31 and S32

```bash
python -m reproduction.zebrafish.plot_population_growth \
  --run-dir outputs/zebrafish_analysis \
  --aligned-h5ad data/zebrafish/aligned.h5ad \
  --output-dir outputs/zebrafish_population_figures
```

This reads the preceding outputs, arranges the observed and simulated
populations, and scales growth by the within-time 5th and 95th percentiles.
It writes two PDF/PNG figure pairs and the plotted arrays and growth tables.

## Calculate YSL gene dynamics

S35 follows the descendants of the 29 YSL cells observed at time zero. It uses
a continuous simulation with learned population weights, not a new set of
YSL cells selected at each time point.

```bash
python -m reproduction.zebrafish.ysl_gene_dynamics \
  --data-dir data/zebrafish \
  --output-dir outputs/zebrafish_ysl_genes \
  --device cuda:0
```

This calls `cb.tl.simulate_sde_points` for the weighted trajectory and
`cb.tl.simulate_sde_points_split_from_x0` for a second realization of growth
through population resampling. Both start with the same cells and use the
paper's learned-edge rule within the observed expression-state range.

For each initial-YSL descendant, the code reconstructs gene expression using
the PCA loadings and center stored in the aligned H5AD. It clips negative
values to zero before averaging. The weighted trajectory uses normalized
population weights within the lineage. The resampled trajectory uses an
ordinary mean across descendants.

As in the original analysis, the 250 genes are selected by their larger
temporal variance across the two growth representations. S35 displays the
weighted trajectory, with each gene centered and scaled across time. The
command writes `lineage_gene_expression.csv`, `temporal_expression.csv`,
`temporal_zscores.csv`, both simulated trajectories, and the new S35 PDF/PNG.

## Calculate the expression-reconstruction comparison

```bash
python scripts/run_zebrafish_paper_downstream.py \
  --aligned-h5ad data/zebrafish/aligned.h5ad \
  --model-dir data/zebrafish/model \
  --output-dir outputs/zebrafish_analysis \
  --shared-cache-dir outputs/zebrafish_analysis/classifier_cache \
  --stage s25 \
  --device cuda:0
```

`s25` is the original analysis name for the reconstruction comparison in S38.
It also calculates an interval-specific gene analysis, starting from each observed stage
and simulates to the next half-time point. Growth is active. At measured times
it uses the observed cell-type annotations. At intermediate times it uses the
classifier to select YSL cells.

`cb.tl.summarize_temporal_gene_patterns` maps the expression-state PCs back to
genes using the stored PCA loadings and center, clips negative reconstructed
values to zero, and calculates mean expression for the selected cells. It
selects 250 genes by their temporal variance. The command also calculates the
observed YSL expression means for comparison with their PCA reconstruction.

It writes `s25/mean_inverse_pca_log1p_clipped.csv`,
`s25/top_variable_genes.csv` and `s25/observed_exact_log1p_anchors.csv`.

## Draw S38

```bash
python -m reproduction.zebrafish.plot_gene_dynamics \
  --run-dir outputs/zebrafish_analysis \
  --output-dir outputs/zebrafish_gene_figures
```

This calculates reconstruction correlations and errors for S38 and writes
the PDF/PNG figure and calculated tables. Use a new output directory when
drawing again.

The [S31–S38 notebook](zebrafish_si_s31_s38.ipynb) displays the paper's recorded
outputs. [Daughter-cell perturbations](zebrafish_daughter_noise.md) has the
simulation and plotting commands for S37.
