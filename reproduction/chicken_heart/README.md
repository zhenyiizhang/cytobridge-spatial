# Chicken-heart daily analyses

The collaborator archive received on 5 September 2026 contains downstream
analysis notebooks, not a complete training run. Its corrected notebooks refer
to the chicken-heart model trained at commit `c72e592`.

The final dynamical model, score model, and edge predictor are now in
[the model directory](../../release_artifacts/chicken_heart_model_c72e592).
The standard public training tutorial is in
[the dataset notebook](../../docs/tutorials/dataset_workflows/chicken_heart.ipynb).

## Calculation order

| Notebook in the supplied archive | Reads | Writes |
| --- | --- | --- |
| `formal_daily_piecewise_interpolation_celltypecorrected.ipynb` | Aligned H5AD, trained model, edge predictor, corrected cell-type classifier | Daily slice H5ADs, communication slices, and `manifest.json` |
| `formal_daily_piecewise_replot_celltypecorrected.ipynb` | Those daily slices and manifest, the model, anatomical labels, and plotting helpers | Spatial panels, velocity plots, interaction networks, and lineage plots |
| `formal_d10_velocity_detail_celltypecorrected.ipynb` | D10 slice from that manifest, model, and region labels | D10 velocity detail |
| `formal_supplementary_replot_celltypecorrected.ipynb` | Daily slices, model, `heart_pp.h5ad`, merged metadata, and plotting helpers | Growth, alignment, and velocity panels |

The originals are retained in the local project archive. They have not been
added as runnable public tutorials because the helper modules and matching
inputs below are not in the supplied ZIP.

## Files still needed

- `downstream_helpers/heart.py` and `downstream_helpers/heart_lineage_functions.py`.
- The exact `classifier_resmlp_*.pt` selected by the collaborator. The notebook
  selects the first matching filename, so its filename needs to be recorded.
- Daily slice outputs and `manifest.json` from the supplied interpolation notebook.
- `heart_pp.h5ad` and `chicken_heart_spatial_merged_with_meta.h5ad`.
- The resulting panel PDFs/PNGs, to match these calculations to the manuscript.

The D10 notebook also uses a `package-release` source directory, whereas the
other corrected notebooks use the `c72e592` source snapshot.

## Relationship to the paper

The model source matches the retrained run family. Exact panel correspondence
has not yet been established. The daily interpolation uses model times
`1/3, 2/3, 4/3, 5/3, 2.25, 2.5, 2.75` between D4, D7, D10, and D14.
It sets `split_resample_dt=1/12`; the standard configuration uses `0.05`.
This changes the resampling interval as well as the requested output times.

The supplied `alignment_sensitivity/` code predates the corrected S7–S8
analysis. The current S7–S8 plotting code remains in
`release_artifacts/chicken_heart_alignment_sensitivity_20260831/`.

The large H5AD in `veloagent_vs/` is a VeloAgent D14 output. It is not the
CytoBridge training input or a CytoBridge checkpoint.
