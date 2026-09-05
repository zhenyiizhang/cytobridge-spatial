---
orphan: true
---

# Analysis inputs: Supplementary Figures S2–S3: AGIST

The [figure notebook](../../tutorials/paper_figures/agist_figures.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. calculate cell-level velocity agreement (S2)

```text
python scripts/build_agist_velocity_time_cluster_breakdown.py --archive-a <inferred-velocity.npz> --archive-b <generator-velocity.npz> --data-csv <agist-cells.csv> --cluster-assignments <state-clusters.csv> --cluster-diagnostics <cluster-diagnostics.csv> --style <paper-style.json> --output-dir <agist-summary>
```

Start with: `model states plus inferred and generator velocity vectors`

Writes: `velocity_cosine_per_cell_full.csv`

Next: `summarize S2`


This cell-level table comes from the paper evaluation folder. The next step recalculates the summaries shown in S2.



### 2. summarize and draw (S2)

```text
python scripts/execute_paper_notebooks.py --notebook agist_figures --output-dir <notebook-run>
```

Start with: `s2_velocity_cosine_per_cell.csv.gz`

Writes: `S2 summary CSV files and Supplementary_Figure_S2.pdf/.png`




### 3. generate the benchmark (S3)

```text
python -m scripts.run_spatial_synthetic_benchmark generate --data-dir <data> --version spatial_attraction_2d_gene_2d_space_v8_balanced_joint_interaction --n-particles 400 --interaction-strength 0.5 --gene-interaction-gain 3.0
```

Start with: `declared simulator parameters and fixed seeds`

Writes: `attractive_observed.h5ad; attractive_fixed_reference.npz; no_interaction_fixed_reference.npz; manifest.json`

Next: `train S3`




### 4. fit the model used for S3 (S3)

```text
python -m scripts.train_spatial_synthetic_realdata_epochs --data-dir <data> --output-root <run>/training --config configs/spatial_synthetic_attraction_realdata_epochs.yaml --device cuda
```

Start with: `attractive_observed.h5ad; manifest.json; training YAML`

Writes: `training/model/Score_Refine/best_model.pth; training/model/adata.h5ad; training/training_manifest.json`

Next: `evaluate S3`




### 5. run five-seed evaluation (S3)

```text
python -m scripts.run_spatial_synthetic_benchmark evaluate --data-dir <data> --model-dir <run>/training/model --stage Score_Refine --evaluation-dir <run>/evaluation_fixed400_five_seed --seeds 1,4,8,32,256 --device cuda
```

Start with: `fixed references; Score_Refine checkpoint; model config`

Writes: `dense_rollout_seed_1.npz; growth_mass_metrics.csv; interaction_radial_curve.csv; interaction_ablation_metrics.csv; acceptance.json`

Next: `draw S3`




### 6. draw the manuscript figure (S3)

```text
python -m scripts.reporting.build_v11b_main_figure --run-root <run> --pdf-output <output.pdf>
```

Start with: `training/model/adata.h5ad; evaluation_fixed400_five_seed/*.csv; dense_rollout_seed_1.npz`

Writes: `manuscript_figure/finite_range_cell_cell_attraction_benchmark.pdf/.png`
