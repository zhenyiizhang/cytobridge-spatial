# SpaTrack comparison for Supplementary Figure S7

This calculation uses the multi-time-point transport solver from [SpaTrack](https://github.com/yzf072/spaTrack), commit `1cc5edec3699f7d8e29663ce3bb0c02cad5600db`. The unchanged numerical module is included as `official_spatrack_utils.py`, with its MIT license.

## Input and calculation

Start with the `inputs` directories produced by the five-dataset benchmark preparation step in the S8 notebook. Each `loto_tN` folder contains `training_reference.npz`, `source_roster.npz`, `truth_tN.npz`, and `manifest.json`.

For a held-out stage, the script takes the nearest observed stage on either side. It uses the same deterministic support of at most 800 cells per stage and the same 5,000 starting cells as the other benchmark methods.

PCA features can be negative. The expression KL cost in the default multi-time-point SpaTrack interface is therefore replaced by Euclidean distance between the shared PCA features. That distance is divided by its maximum over the two training anchors. Each within-stage spatial distance matrix is also divided by its maximum, as in SpaTrack.

The transport marginals, spatial term, and solver follow SpaTrack. Settings are fixed at `alpha=0.1`, `epsilon=0.01`, `rho=infinity`, and 10 outer iterations for every dataset.

The fitted coupling maps each starting cell to the weighted mean of its target-stage cells. Linear interpolation between the source and mapped states gives the withheld-stage prediction. Interpolation follows the existing benchmark time coordinates. The resulting predictions have uniform weights.

## Run

Run the commands below in the `cytobridge-spatial` folder after installing the benchmark dependencies. This calculation used NumPy 1.26.4 and POT 0.9.7. Use NumPy below 2 because this version of the SpaTrack solver uses `np.Inf`.

```bash
python paper_resources/benchmark_controls/code/spatrack/run_benchmark.py \
  --dataset zebrafish \
  --input-root paper_run/data/benchmark/inputs/zebrafish \
  --targets 1 2 3 \
  --official-source paper_resources/benchmark_controls/code/spatrack/official_spatrack_utils.py \
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

The output directory must be new. The script writes the coupling, predictions, fitting settings, and repeat-level metrics. The withheld observations are used to calculate prediction error.

## Score and redraw

Scoring uses the existing benchmark transformation fitted on the training reference. Sliced-W2 uses all predicted and observed points, 1,024 projections, and five shared projection seeds. The exact W1/W2 calculations use the existing 800-point cap. All methods use the shared PCA features and aligned coordinates.

To include new SpaTrack results in S7, average the five projection repeats
and replace the SpaTrack rows of the benchmark table:

```python
from pathlib import Path
import pandas as pd

from reproduction.current_figures import draw_supplementary

datasets = ["zebrafish", "mosta", "arista", "admouse", "chicken_heart"]
root = Path("outputs/spatrack")
metrics = pd.concat([pd.read_csv(root / d / "metrics_long.csv") for d in datasets])
new = metrics.groupby(["dataset", "target", "space"], as_index=False).sliced_w2.mean()
new["method"] = "spatrack_pca"
source = Path("reproduction/benchmark/source_data/target_means.csv")
table = pd.read_csv(source)
table = pd.concat([table[table.method.ne("spatrack_pca")], new], ignore_index=True)
inputs = root / "figure_inputs"
inputs.mkdir(parents=True, exist_ok=True)
table.to_csv(inputs / "loto_target_stage_means_with_spatrack.csv", index=False)
draw_supplementary([7], root / "figures", results_dir=inputs)
```

The benchmark inputs can be downloaded with:

```bash
python -m CytoBridge.datasets benchmark --kind benchmark_inputs.zip --output-dir paper_run
```
