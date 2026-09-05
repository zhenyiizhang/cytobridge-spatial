# AD 0.05 interpolation to official NicheNet pipeline

This directory is a **code handoff** for the current AD/AdMouse NicheNet
analysis.  The source scripts were copied without editing from the formal
0.05-grid interpolation and `latest_nichenet` analysis roots.  It is not a
standalone raw-data archive: the 51 H5AD state files, trained CytoBridge model,
aligned training H5AD, classifier checkpoint, and the NicheNet R data objects
remain in their provenance roots.

## Run order

1. `interpolation/run_admouse_interpolation_0p05.py`
   Generates 51 model states at `t = 0.00, 0.05, ..., 2.50`, with real observed
   slices replacing the states at `t = 0.00, 1.00, 2.00`.
2. `official_nichenet/01_prepare_model_expression_inputs.py`
   Inverse-projects model state latents into the processed 347-gene space and
   derives per-window Microglia backgrounds, sender genes, and the top 50
   positive reconstructed-change target genes.
3. `official_nichenet/02_run_official_nichenet_51_windows.R`
   Runs official NicheNet `predict_ligand_activities()` and
   `get_weighted_ligand_target_links()` over all 50 adjacent windows.
4. `official_nichenet/03_*` through `08_*` and `official_nichenet/redraw/`
   Produce the official and locally assembled NicheNet visualizations.

## Code layout

- `interpolation/`: formal 0.05-grid CytoBridge rollout script.
- `official_nichenet/`: current authoritative NicheNet preparation, scoring,
  visualization, and redraw scripts.
- `nichenetr_official_R/`: complete local snapshot of the official NicheNet R
  functions used by the workflow.
- `reference_snapshot/`: historical workflow reference files only. They are
  not part of the current formal NicheNet run and must not be substituted for
  `official_nichenet/` scripts.

## Required external inputs and software

All executable scripts derive locations from their own paths; relocating the
`Final_Figures` directory alongside the `admouse_0815` and `nichenet` result
directories therefore does not require editing hard-coded paths. A full rerun requires:

- `../admouse_0815/refs/training/adata.h5ad` and the matched
  model/checkpoint directory;
- `../admouse_0815/refs/accepted_downstream/classifier_cache/`
  classifier checkpoint;
- 51 H5AD state files under
  `../nichenet/new_interpolation/results/interpolation/slice_data/`;
- NicheNet ligand-target and ligand-receptor RDS priors; and
- the CytoBridge Python environment, GPU for interpolation, and the required R
  packages for official NicheNet and plotting.

The final figures and derived tables are stored separately in the parent
`figures/` and `data/` directories.  No original model result or NicheNet score
was changed while creating this code handoff.
