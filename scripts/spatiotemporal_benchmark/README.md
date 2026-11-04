# Spatiotemporal benchmark

The benchmark predicts an intermediate stage after leaving it out of model fitting. Each method uses the same PCA features, aligned coordinates and initial particles. PCA is fitted before the stage is withheld. Predictions are compared with the withheld cells using Sliced-W2 in the output spaces supported by each method.

Follow the [benchmark tutorial](../../docs/tutorials/paper_figures/loto_benchmark.ipynb) to download inputs, run methods and draw the comparisons. `build_inputs.py` prepares the shared input files, `verify_inputs.py` checks them, and `run_unified_benchmark.py` runs the configured methods. Dataset configurations are in `configs/unified_benchmark/`.
