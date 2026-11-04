# Five-dataset benchmark

Supplementary Figures S7 and S8 compare PASTE2 and MIOFlow 2.0.
The errors for all eleven methods and controls are in `source_data/target_means.csv`.

## Draw the paper figures

From the repository root:

```bash
python -m reproduction.current_figures --figures 7 8 --output-dir outputs/benchmark
```

This calculates the comparison ranks and relative errors from the 297 target,
method and output-space measurements. It writes new PDF and PNG figures.

## Repeat the new method comparisons

Download the fixed inputs for all eleven held-out stages:

```bash
python -m CytoBridge.datasets benchmark --output-dir paper_run --kind benchmark_inputs.zip
```

The archive contains training references, starting cells, withheld observations
and manifests under `paper_run/data/benchmark/inputs/`. The withheld stage is
absent from the training reference. Dataset names used in commands are
`chicken_heart`, `mosta`, `arista`, `admouse` and `zebrafish`.

Install the selected method's dependencies in its own environment. The
original implementations and their licenses are retained under
`run_code/official_sources/`. For PASTE2, install
`python -m pip install -e reproduction/benchmark/run_code/official_sources/paste2`.
For MIOFlow 2.0, install
`python -m pip install -e reproduction/benchmark/run_code/official_sources/MIOFlow`.

For example, repeat the AD held-out-stage comparison with PASTE2:

```bash
python reproduction/benchmark/run_code/paste2_final/run_benchmark.py \
  --method paste2 --dataset admouse --target 1 \
  --input-root paper_run/data/benchmark/inputs --output outputs/paste2_admouse
```

For MIOFlow 2.0, use its environment and the corresponding command:

```bash
python reproduction/benchmark/run_code/mioflow2/run_benchmark.py \
  --method mioflow2 --dataset admouse --target 1 \
  --input-root paper_run/data/benchmark/inputs --output outputs/mioflow2_admouse
```

The MIOFlow 2.0 calculation uses an NVIDIA GPU. Its spatial-neighborhood
features inform the predicted gene-expression state. It does not output tissue
coordinates, so it is compared in gene-expression space.

Both commands calculate the predictions and evaluate them using the shared
Sliced-W2 procedure. Parameters are recorded in `protocol.json`, and each run
writes its settings with its results. PASTE2 uses overlap fraction 1, alpha
0.1, squared Euclidean PCA cost, no distance normalization and 200 solver
iterations. Other method calculations are described in the benchmark tutorial.
