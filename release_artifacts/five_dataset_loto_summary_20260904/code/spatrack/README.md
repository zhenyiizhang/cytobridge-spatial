# SpaTrack comparison for Supplementary Figure S44

This calculation uses the multi-time-point transport solver from [SpaTrack](https://github.com/yzf072/spaTrack), commit `1cc5edec3699f7d8e29663ce3bb0c02cad5600db`. The unchanged numerical module is included as `official_spatrack_utils.py`, with its MIT license.

## Input and calculation

Start with the `inputs` directories produced by the five-dataset benchmark preparation step in the S45 notebook. Each `loto_tN` folder contains `training_reference.npz`, `source_roster.npz`, `truth_tN.npz`, and `manifest.json`.

For a held-out stage, the script takes the nearest observed stage on either side. It uses the same deterministic support of at most 800 cells per stage and the same 5,000 starting cells as the other benchmark methods.

PCA features can be negative. The expression KL cost in the default multi-time-point SpaTrack interface is therefore replaced by Euclidean distance between the shared PCA features. That distance is divided by its maximum over the two training anchors. Each within-stage spatial distance matrix is also divided by its maximum, as in SpaTrack. No features are shifted to make them positive.

The transport marginals, spatial term, and solver follow SpaTrack. Settings are fixed at `alpha=0.1`, `epsilon=0.01`, `rho=infinity`, and 10 outer iterations for every dataset. These settings were specified before evaluating the held-out observations.

The fitted coupling maps each starting cell to the weighted mean of its target-stage cells. Linear interpolation between the source and mapped states gives the withheld-stage prediction. Interpolation follows the existing benchmark time coordinates. The predicted cloud uses uniform weights and does not predict growth. This is a PCA-based benchmark adaptation of SpaTrack, not its default gene-expression/KL analysis.

## Run

From a CytoBridge source checkout with the benchmark dependencies installed. This calculation was run with NumPy 1.26.4 and POT 0.9.7. Use NumPy below 2 because the archived SpaTrack solver uses `np.Inf`.

```bash
python release_artifacts/five_dataset_loto_summary_20260904/code/spatrack/run_benchmark.py \
  --dataset zebrafish \
  --input-root results/benchmark/zebrafish/inputs \
  --targets 1 2 3 \
  --official-source release_artifacts/five_dataset_loto_summary_20260904/code/spatrack/official_spatrack_utils.py \
  --output outputs/spatrack/zebrafish
```

Change `--dataset`, `--input-root`, `--targets`, and `--output` together for each dataset:

| Dataset | Targets |
|---|---|
| zebrafish | 1 2 3 |
| mosta | 1 2 |
| arista | 1 2 3 |
| admouse | 1 |
| chicken_heart | 1 2 |

The output directory must be new. The script writes the coupling, predictions, fitting settings, and repeat-level metrics. Truth is read for scoring only after all predictions for that dataset have been saved.

## Score and redraw

Scoring uses the existing benchmark transformation fitted on the training reference. Sliced-W2 uses all predicted and observed points, 1,024 projections, and five shared projection seeds. The exact W1/W2 calculations use the existing 800-point cap. The archived PCA and spatial representations are fixed across methods and were not refitted within each fold.

To draw S44 from a new run, combine each dataset's `metrics_long.csv` and pass the result to the plotting script:

```python
from pathlib import Path
import pandas as pd

datasets = ["zebrafish", "mosta", "arista", "admouse", "chicken_heart"]
root = Path("outputs/spatrack")
metrics = pd.concat([pd.read_csv(root / d / "metrics_long.csv") for d in datasets])
metrics.to_csv(root / "spatrack_metrics.csv", index=False)
```

```bash
python release_artifacts/five_dataset_loto_summary_20260904/code/plot_five_dataset_loto_wins_summary.py \
  --spatrack-metrics outputs/spatrack/spatrack_metrics.csv \
  --output outputs/spatrack/figure_s44.pdf \
  --table-output-dir outputs/spatrack/tables \
  --manifest outputs/spatrack/figure_manifest.json
```

This recalculates panel a and every dataset panel from numerical results. It does not load a published figure.
