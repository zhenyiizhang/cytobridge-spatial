# Figure provenance: chicken-heart alignment sensitivity

## Figure identity

- Manuscript location: Supplementary Figures S7 and S8.
- Dataset: chicken heart, D4, D7, D10, and D14.
- Accepted archive date: 2026-08-31.
- Accepted CytoBridge commit:
  `c72e592d0dea70941bc4971a79c3c903d7454b08`.
- Random seed: 42; `PYTHONHASHSEED=0`.

## Scientific question

The analysis asks whether the fitted alignment and the main downstream vector
and interaction results remain concordant after bounded, stage-specific rigid
perturbations of the input tissue coordinates.

## Source paths

- Accepted alignment input:
  `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-ot-alignment-20260822-f5550e1-r1/result/chicken_heart_ot_aligned.h5ad`
- Input SHA-256:
  `2783cb49fe4df9b7bb99398ced6ac83a998d8708517e1d4698da76435928a993`
- Accepted manuscript run:
  `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-full-ot-20260823-r2`
- Sensitivity-analysis root:
  `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-alignment-sensitivity-audit-20260831-r1`
- Accepted server Python:
  `/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python`

Perturbations were applied to `obsm["spatial_ot_input"]`. The accepted D7
pre-orientation was retained. Every condition repeated alignment, CytoBridge
training, and downstream analysis from the same formal workflow configuration.
No completed run directory was overwritten.

## Conditions shown in S7 and S8

- Unperturbed repeat.
- Translation with maximum displacement of 0.36 or 0.71 median-nearest-neighbor
  units.
- Rotation with maximum absolute angle of 3.1 or 6.2 degrees.
- The corresponding combined translation and rotation conditions.

The full audit also includes stronger perturbations. They are retained as stress
tests but are not displayed and are not part of the paper's bounded robustness
statement.

## Comparison definitions

- Perturbed conditions are compared with the concurrently trained unperturbed
  repeat. The unperturbed repeat is compared with the accepted manuscript run.
- Cells are joined by `obs_names` rather than row position.
- Interaction edges are joined by source and target cell IDs rather than local
  integer indices.
- Aligned-coordinate residuals are calculated after a per-stage proper rigid
  frame adjustment, with reflections excluded, and are divided by the baseline
  section radius.
- Full- and interaction-velocity agreement is the median cell-wise cosine after
  the same frame adjustment.
- Interaction-weight overlap is the normalized weighted Jaccard index over the
  union of source-target cell pairs.

## Results within the displayed range

Across the six perturbation conditions and four stages:

- aligned-coordinate residual was at most 0.0253686066 of section radius
  (2.54% after rounding);
- median full-velocity cosine was at least 0.902656481;
- median interaction-velocity cosine was at least 0.837326159;
- normalized interaction-weight overlap was at least 0.783630291;
- source-target edge-set Jaccard was at least 0.887480776.

Heatmap annotations are rounded for display. A label of `1.00*` means that the
underlying value is below one and rounds to 1.00 at two decimal places.

## Panel-to-source mapping

- S7a: `data/plot_inputs.npz`, keys `source_input_xy` and
  `accepted_aligned_xy`.
- S7b: `data/coordinate_metrics.csv`,
  `data/velocity_metrics_pooled.csv`, and `data/interaction_metrics.csv`.
- S7c: `manifests/input_manifest.json` and
  `manifests/lower_input_manifest.json`.
- S8a-b: `data/plot_inputs.npz`, using the original, perturbed, and aligned
  coordinate arrays for each displayed condition.

The plotting program is `figure_code/plot_heart_alignment_sensitivity.py`.
It reads the archived numerical inputs and calculates every panel; it does not
load an existing PDF or PNG.

## Accepted outputs

- `figures/heart_alignment_sensitivity_S7_final.pdf`
- `figures/heart_alignment_sensitivity_S7_final.png`
- `figures/heart_alignment_sensitivity_S8_final.pdf`
- `figures/heart_alignment_sensitivity_S8_final.png`

Both PDFs are single-page A4 portrait vector files. Arial text is embedded in the
accepted PDFs, and the PNGs were exported at 320 dpi.

S7 contains four rasterized heatmap layers; its text, annotations, axes, bars,
legends, and layout remain vector. S8 contains no raster layer.

## Rebuild

From the archive directory, recreate the accepted figure pages with:

```bash
python figure_code/plot_heart_alignment_sensitivity.py
```

To write to a separate directory while checking the archive, use:

```bash
python figure_code/plot_heart_alignment_sensitivity.py \
  --output-dir /path/to/check
```

The complete preparation, training, comparison, diagnostic, and export command
order is given in `README.md`.
