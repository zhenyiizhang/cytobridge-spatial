# Training time and memory

Full-model compute measurements are available through
`CytoBridge.results.load_full_model_compute_cost`. When no directory is given,
the reader loads the table bundled with the installed package.

```python
from CytoBridge.results import (
    format_full_model_compute_cost,
    load_full_model_compute_cost,
)

results = load_full_model_compute_cost()
raw = results.measurements
display_table = format_full_model_compute_cost(results)
measurement_settings = results.manifest["measurement"]
```

The raw table contains one row per dataset with these fields:

| Field | Meaning |
| --- | --- |
| `time_points_used_for_training` | number of observed training times |
| `training_time_point_labels` | labels for those observed times |
| `observed_cells_or_spots` | number of observations used for fitting |
| `training_time_seconds` | elapsed time for `TrainingPipeline.train` |
| `peak_host_memory_mib` | process maximum resident set size sampled after training |
| `peak_gpu_allocation_mib` | largest PyTorch allocation across training stages |

The measurement settings in the manifest include the hardware, number of
training stages, included timing operations, and display units.

Training time includes stage preparation, optimizer setup, epochs, checkpoint
selection, and checkpoint writing. It excludes preprocessing, prediction,
evaluation, downstream analysis, and AnnData serialization.

## Write the display tables

```python
from pathlib import Path

from CytoBridge.results import write_full_model_compute_cost_tables

paths = write_full_model_compute_cost_tables(
    results,
    Path("outputs/full_model_compute_cost"),
)
```

`format_full_model_compute_cost` converts seconds to minutes and MiB to GiB for
display without changing the raw measurements.

## Training histories

The training-history reader exposes the six stages separately:

```python
from CytoBridge.results import (
    calculate_smoothed_training_history,
    load_training_history_results,
)

history_results = load_training_history_results()
smoothed = calculate_smoothed_training_history(history_results)
```

Neural-ODE and score-matching stages optimize different objectives. Plot and
summarize each stage on its own loss scale.

## Measurement limits

- Each packaged row describes one measured full-model run rather than an
  average across repeated fits.
- Wall time and memory depend on hardware, software versions, dataset size,
  and configuration.
- Peak GPU allocation is PyTorch allocation, not total device usage.
- CPU maximum resident set size is a process-lifetime value and can include
  memory retained from earlier training stages.
