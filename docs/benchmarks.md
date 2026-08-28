# Benchmarks

CytoBridge includes the result tables used in the paper figures and functions
that read them. The functions check the required columns and calculate the
values passed to the plotting code. Pass a directory to read another completed
run, or omit it to reproduce the paper from the included tables.

## Leave-one-timepoint-out benchmark

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

The returned `protocol` dictionary records:

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

The benchmark is leave-one-timepoint-out. The target slice is omitted while
fitting each run, and transforms are fit from the remaining observed slices.
Projection repeats measure numerical variation in sliced-W2 calculations.
Method/space combinations absent from `protocol["spaces_by_method"]` are not
created by an adapter.

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
| Interaction comparisons | `load_interaction_evidence_results` | `interaction_evidence_statistics`, `write_interaction_evidence_tables`, `plot_interaction_evidence` |
| Training histories | `load_training_history_results` | `calculate_smoothed_training_history`, `write_training_history_tables`, `plot_training_histories` |
| Training time and memory | `load_full_model_compute_cost` | `format_full_model_compute_cost`, `write_full_model_compute_cost_tables` |

All of these functions are available from `CytoBridge.results`. The
corresponding notebooks under `docs/tutorials/paper_figures/` show complete
reader-to-output examples.
