# Chicken-heart alignment sensitivity

This experiment uses translations of 1 and 2 times each section's median nearest-neighbor distance and rotations of 1° and 3°. The combined conditions use 1× with 1° and 2× with 3°. Rotation signs and translation directions differ between stages, following the earlier experiment. There are seven conditions including the unperturbed repeat.

## Calculation

1. Read the accepted chicken-heart input and its raw `spatial_ot_input` coordinates.
2. Perturb the coordinates without changing counts, annotations, observation identities, or the accepted D7 pre-orientation.
3. Repeat preprocessing, spatial alignment, edge-classifier fitting, complete model training, and downstream calculations for every condition with seed 42.
4. Match cells by their observation identifiers. Match directed interaction edges by their source and target identifiers.
5. Compare each perturbed run with the new unperturbed run. Compare the unperturbed repeat with the earlier accepted model.
6. Draw S7/S8 from the resulting coordinate arrays and metric tables.

Coordinate residuals are measured after centering and a proper rotation, without reflection, and normalized by the reference section radius. Velocity values in the figure are median cellwise cosines in the two spatial dimensions. Interaction-weight overlap is the normalized weighted Jaccard index over the union of directed edges.

## Numerical precision

The package retains float64 coordinates during centering and scaling, then creates float32 neural-network inputs. This corrects premature rounding of slide coordinates. The training objective, architecture, seed, and epoch counts are unchanged. Pure translations are removed by the configured section-wise centering. They should therefore give the same normalized coordinates as their corresponding untranslated conditions.

The code change and its base package are recorded in `precision_change.json`. Two regression tests check preservation of raw coordinate precision and invariance to section-wise translation. These and the existing alignment time-mapping tests passed (13 tests).

## Scope of this experiment

The author selected this smaller rotation range after a 6° test produced a D4 reflection during the local-distance/OT alignment stage. A separate precision check removed spurious translation dependence but did not remove that reflection. These earlier results remain in the server directories below. The present figures evaluate small perturbations around the selected section orientations, not robustness to larger rotations.

- Earlier integer experiment: `/home/ubuntu/cytobridge_runs/heart_alignment_integer_20260906`
- Precision and orientation checks: `/home/ubuntu/cytobridge_runs/heart_alignment_precision_20260906`

## Source and code

Server alias: `cytobridge-gpu`.

- Original input: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-ot-alignment-20260822-f5550e1-r1/result/chicken_heart_ot_aligned.h5ad`
- Accepted model: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-full-ot-20260823-r2`
- Experiment: `/home/ubuntu/cytobridge_runs/heart_alignment_small_20260906`
- Permanent model archives: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/heart-alignment-small-20260906-r1`
- Working files: `/dev/shm/cytobridge-heart-alignment-small-20260906-r1`
- Python: `/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python`

`run_experiment.py` generates the inputs, runs the analysis, and archives every completed run. Its `prepare_and_run.py`, `compare_runs.py`, and `export_plot_inputs.py` helpers are included in the server archive. The archive contains the exact package copy. The original input remains in its permanent source directory and need not be duplicated in each archive.

To redraw the figures locally after copying the small `summary` folder and input manifest:

```bash
/opt/anaconda3/bin/python output/heart_alignment_small_20260906/summarize_results.py
/opt/anaconda3/bin/python output/heart_alignment_small_20260906/plot_sensitivity.py
```

The plotting script uses the existing publication-layout helpers in `worktrees/package-final-audit-20260829/reproduction/supplementary_figures`. Its inputs are numerical arrays and CSV tables, not previously rendered figures. It verifies the actual perturbation amplitudes before creating labels.

The initial deliverable is a two-page S7/S8 preview. The larger-rotation figures have not replaced the figures in the SI or response.
