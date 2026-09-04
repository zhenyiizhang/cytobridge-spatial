# LR-prior and interaction ablations

These scripts generate the inference-time comparison in Supplementary Figure S42c,d. The Full and No-LR-prior retraining results in S42a,b are a separate experiment.

## Start with a fitted model

Use the `training` directory written by the dataset workflow and the matching unified-benchmark input manifest. The manifest's `full_data` split supplies the initial population and observed targets. Its aligned input must match the training record. See the dataset notebooks for preprocessing and training, and `scripts/spatiotemporal_benchmark/run_unified_benchmark.py prepare` for creating the benchmark inputs.

Run this command from the repository root, replacing the model and manifest paths:

```bash
python scripts/paper_figures/interaction_ablation/run_comparison.py \
  --dataset arista \
  --model-dir /path/to/arista/training \
  --input-manifest /path/to/benchmark/arista/inputs/manifest.json \
  --code-root . \
  --output outputs/interaction_inference/arista \
  --seeds 42 43 44 \
  --gpu-metrics \
  --device cuda:0
```

Repeat with the corresponding paths for `zebrafish`, `mosta`, `admouse`, and `chicken_heart`. Each command writes `metrics.csv`, `manifest.json`, and the paired predictions in `seed_42`, `seed_43`, and `seed_44` directories. Use a new output directory for each run.

The two conditions use the same checkpoint, initial particles, and random streams. Only the entire interaction-network output is set to zero. Intrinsic drift, growth, score, and diffusion settings stay unchanged. Growth remains represented by continuous particle weights, without resampling. Models are not retrained. These are reconstructions of observed populations, not held-out predictions.

`weighted_simulation.py` contains the simulation and weighting implementation used for this experiment. `accelerated_metrics.py` evaluates the same POT sliced-W2 calculation in float64 batches on the selected GPU. Omitting `--gpu-metrics` uses the original CPU metric routine. All observed cells and 1,024 projection directions are retained.

## Combine the results and draw S42

The first input below is `paired_target_deltas.csv` from the matched Full/No-LR report. The second is the directory containing the five inference runs just completed.

```bash
python scripts/collect_figure_inputs.py s42 \
  --no-lr-table /path/to/matched-ablation-report/paired_target_deltas.csv \
  --inference-results-dir outputs/interaction_inference \
  --output-dir outputs/s42_inputs
```

```bash
cytobridge figure interaction-ablation \
  --results-dir outputs/s42_inputs \
  --output-dir outputs/s42
```

To redraw the published figure from the numerical tables included with the package:

```bash
cytobridge figure interaction-ablation --output-dir outputs/s42_published
```

This command recalculates the projection means, paired error ratios, target summaries, and error bars. It does not load a pre-rendered figure. The plotting API is `CytoBridge.results.interaction_ablation`.

The exact accepted model paths and configurations are recorded in `CytoBridge/results/data/interaction_ablation/inference_run_manifests.json`. The experiment used package commit `61f0b550678ed75e706638ceb7638a0818b7e033`. The three inference seeds quantify prediction variability for the same fitted checkpoint, not independent training or biological replicates.
