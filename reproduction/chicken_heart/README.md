# Chicken-heart development: trajectories, cell types, and growth

These four notebooks calculate daily populations, lineage transitions,
interaction networks, growth, and velocity from the trained chicken-heart model.
They are portable copies of the collaborator's `celltypecorrected` notebooks.
For an introduction to the Python API, start with the
[chicken-heart tutorial](../../docs/tutorials/dataset_workflows/chicken_heart.ipynb).

## Inputs

Extract `chicken_heart_model.zip`, `chicken_heart_analysis_data.zip`, and
`chicken_heart_training_inputs.zip` into one project folder. The last archive
provides original coordinates and section metadata for the supplementary
alignment plots. None of these notebooks trains a model.

```text
my_project/data/chicken_heart/
├── aligned.h5ad
├── workflow.json
├── model/
├── edge_classifier/
├── classifier_cache/classifier_resmlp_432f09f20ff65c0d.pt
└── raw/
    ├── heart_pp.h5ad
    └── chicken_heart_spatial_merged_with_meta.h5ad
```

Install the package and notebook dependencies from the source checkout:

```bash
python -m pip install -e '.[all]' nbformat nbclient ipykernel
```

Run the four notebooks in order:

```bash
python scripts/run_chicken_heart_daily_notebooks.py --project-dir my_project
```

The executed notebooks are saved in `my_project/outputs/chicken_heart_paper/notebooks/`.
Figures and tables are in the corresponding folders under
`my_project/outputs/chicken_heart_paper/new_runs_formal_trained/`.
Use `--output-dir` to keep another run separately.

## Calculation order

| Step | Notebook | Reads | Writes |
| --- | --- | --- | --- |
| 1 | [Daily interpolation](notebooks/formal_daily_piecewise_interpolation_celltypecorrected.ipynb) | Aligned data and trained model | Daily H5AD slices, communication slices, and `manifest.json` |
| 2 | [Cell-state trajectories and lineage transitions](notebooks/formal_daily_piecewise_replot_celltypecorrected.ipynb) | Those slices and manifest | Spatial panels, velocity, interaction networks, and lineage plots |
| 3 | [D10 velocity](notebooks/formal_d10_velocity_detail_celltypecorrected.ipynb) | The D10 slice | Regional velocity plots |
| 4 | [Growth and supplementary plots](notebooks/formal_supplementary_replot_celltypecorrected.ipynb) | Daily slices and original section metadata | Growth, alignment, and velocity panels |

For one step, add `--step interpolation`, `--step daily`, `--step d10`, or
`--step supplementary`. The last three read the interpolation results in the
same output directory. When opening a notebook manually, set `REPO_ROOT` to
the source checkout and `PROJECT_DIR` to the folder containing `data/`.

## Model and analysis settings

The notebooks use the chicken-heart model trained at commit `c72e592` and the
classifier named above. The four observed stages D4, D7, D10, and D14 map to
model times 0, 1, 2, and 3. Interpolation restarts at each observed stage and
uses `split_resample_dt=1/12` for the daily output grid.

The port changes file locations and selects the classifier by its exact name.
The copied numerical cells and plotting-function bodies are retained. The
original D10 notebook named its package folder `package-release`. It has
been tested with the same package source as the other three notebooks.

These outputs are kept separately from the assembled manuscript pages.
The current S7–S8 sensitivity figures use the code in
`release_artifacts/chicken_heart_alignment_sensitivity_20260831/`, not the
older alignment code supplied in the collaborator ZIP.
