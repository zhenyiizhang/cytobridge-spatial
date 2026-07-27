# Supplemental centered-shape analysis

`reviewer_shape_analysis.py` is an opt-in supplement to the frozen zebrafish
benchmark. It does **not** replace or modify the primary W1, W2, sliced-W2 or
TMV tables. Its purpose is narrower: separate an error in the predicted
distribution's overall location from an error in its shape.

## Required input and scientific scope

The CLI accepts only a completed `matched_evaluation_manifest.json` produced by
`evaluate_matched_tracks.py`. Before calculation it rechecks the matched run
contract, method registry, prediction inventory, prediction NPZ/summary hashes,
input manifest, truth files and common frozen transform. This guarantees that
the supplement evaluates the same per-method points and weights as the matched
benchmark.

The two tracks are never pooled:

- `loto_transductive_interpolation` is the held-out-stage result. The target
  rows are absent from that fold's training input, but the common corrected
  PCA/spatial representations and the matched `t0+t4` transform make this a
  transductive interpolation diagnostic, not an inductive raw-gene forecast.
- `full_data_in_sample_oracle_control` is an in-sample/oracle reference. It is
  useful as a reconstruction ceiling, but it is not an independent
  generalization result and is not used to rank LOTO.

## Metrics

All calculations use the same frozen transform, parameters and random keys for
every method. Lower is better for every metric.

1. **Centroid error** is the Euclidean distance between the weighted prediction
   centroid and the observed centroid. It measures global displacement.
2. **Centered sliced W2** subtracts each distribution's own centroid and then
   computes sliced W2 on the full empirical supports. It measures residual
   shape rather than translation.
3. **Centered exact W1/W2** applies the same centering before the existing
   matched exact-OT calculation. If either support exceeds
   `--max-ot-points`, “exact” means exact EMD on the deterministic audited
   support sample, as in the existing benchmark.
4. **Covariance Bures distance** compares the full-support weighted population
   covariance matrices. It is the centered Gaussian W2 distance and therefore
   captures second-order spread/orientation, while centered Wasserstein metrics
   can also detect non-Gaussian shape differences.

Prediction weights are normalized to probability mass for these shape metrics;
native total mass remains the separate TMV question. If weights are absent,
prediction rows receive uniform mass. Truth rows receive uniform mass.

Exact duplicate prediction coordinates are collapsed independently in each
transformed feature space and their masses are summed. This preserves the
empirical measure while preventing repeated bootstrap copies from being
misread as additional geometric support. The CSV records original row count,
unique support count, collapsed duplicates, zero-weight rows, raw mass and
effective support size.

## Run

From the repository root:

```bash
OUT=/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/benchmarks/zebrafish-clean-20260718

python scripts/spatiotemporal_benchmark/reviewer_shape_analysis.py \
  --matched-manifest \
    "$OUT/reports/evaluation/matched/matched_evaluation_manifest.json" \
  --output-dir "$OUT/reports/shape-reviewer-v1" \
  --n-projections 1024 \
  --projection-repeats 5 \
  --max-ot-points 800
```

Optional `--methods`, `--tracks` and `--targets` filters must name values
already bound in the matched manifest. Use `--no-plots` for a table-only run.
The output directory must be new or empty.

## Outputs and reading the plots

- `shape_metrics_long.csv`: one row per method, track, target and applicable
  space.
- `shape_metrics_summary.csv`: unweighted mean, target SD and range within each
  method/track/space; no cross-space score or rank.
- `shape_metrics_paired_gaps.csv`: descriptive
  `LOTO - full-data` values for exactly matched targets.
- `plots/*.png` and `plots/*.pdf`: one figure per metric, with separate joint,
  state and spatial panels. Grey is the full-data in-sample control, blue is
  LOTO, dots are targets, bars are target means, and thin lines connect the
  same target.
- `shape_analysis_manifest.json`: hashes of every table and plot, input
  bindings, metric definitions, fixed parameters, software/code versions and
  interpretation limits.

Read centroid and centered metrics together. A small centered distance with a
large centroid error means the predicted cloud has approximately the right
shape but is placed in the wrong location. A small centroid error with a large
centered distance means its average location is right but its internal
distribution is wrong. Covariance Bures alone is insufficient to establish
full distributional agreement, so it is reported beside centered W1/W2 rather
than as a replacement.
