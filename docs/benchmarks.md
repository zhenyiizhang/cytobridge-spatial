# Benchmarks

Compare predictions at an omitted time point, or draw the paper comparisons
from the supplied error tables.

## Predict an omitted time point

At each fold, one intermediate stage is excluded from fitting. The predicted
population at that stage is compared with the withheld measurements in the
output spaces available for each method. The comparison includes PASTE2 and MIOFlow 2.0. MIOFlow 2.0 predicts gene-expression states informed by
spatial neighborhoods, so its error is evaluated in gene-expression space.

The [benchmark calculation guide](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/reproduction/benchmark/README.md)
provides the fixed inputs and commands for the two new method comparisons.
The [paper notebooks](tutorials/paper_figures/index.md) calculate the summary
plots from all 297 measurements.

Load the benchmark with `CytoBridge.results.load_loto_benchmark`:

```python
from CytoBridge.results import load_loto_benchmark

benchmark = load_loto_benchmark()
protocol = benchmark.protocol
target_metrics = benchmark.target_means
native_support = benchmark.native_support
paired_ratios = benchmark.paired_ratios
dataset_summary = benchmark.dataset_summary
```

Use `target_metrics` for errors at individual stages and `dataset_summary`
for averages by dataset. The `protocol` dictionary gives:

- the dataset and held-out target-time roster;
- the methods and state spaces available for each method;
- the sliced-W2 projection settings;
- the initial source-population size and predicted-support policy; and
- whether an output is native to a method or produced by its coupling adapter.

A custom results directory must contain
`loto_target_stage_means.csv`, `native_output_support.csv`, and
`protocol.json`:

```python
benchmark = load_loto_benchmark("outputs/loto_tables")
```

Sliced-W2 scaling is fitted using the observed training stages. Each error is
calculated with five sets of projection directions. The available output
spaces are listed in `protocol["spaces_by_method"]`.

`cytobridge workflow --reconstruction-diagnostic` runs an in-sample model
diagnostic and is separate from this protocol.

## Write tables and figures

```python
from pathlib import Path

from CytoBridge.results import (
    plot_loto_benchmark,
    write_loto_benchmark_tables,
)

output_dir = Path("outputs/loto_benchmark")
tables = write_loto_benchmark_tables(benchmark, output_dir)
pdf, png = plot_loto_benchmark(benchmark, output_dir)
```

The command-line wrapper performs the same load, table-writing, and plotting
steps:

```bash
python scripts/results/plot_loto_benchmark.py \
  --output-dir outputs/loto_benchmark
```

Use `--results-dir` to point the wrapper at another compatible table set.

## Other packaged readers

| Topic | Reader | Table or plotting helpers |
| --- | --- | --- |
| Classifier smoothing | `load_classifier_smoothing_results` | `classifier_smoothing_statistics`, `write_classifier_smoothing_tables`, `plot_classifier_smoothing` |
| LR-complex aggregation | `load_lr_complex_aggregation_results` | `summarize_lr_complex_aggregation`, `write_lr_complex_aggregation_tables`, `plot_lr_complex_aggregation` |
| Earlier LR-prior comparison | `load_lr_prior_stvcr_results` | `lr_prior_stvcr_statistics`, `write_lr_prior_stvcr_tables`, `plot_lr_prior_stvcr` |
| Training histories | `load_training_history_results` | `calculate_smoothed_training_history`, `write_training_history_tables`, `plot_training_histories` |
| Training time and memory | `load_full_model_compute_cost` | `format_full_model_compute_cost`, `write_full_model_compute_cost_tables` |

All of these functions are available from `CytoBridge.results`. The
corresponding notebooks under `docs/tutorials/paper_figures/` show complete
reader-to-output examples.
