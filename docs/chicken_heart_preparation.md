# Prepare the chicken-heart counts

This page is for retraining the paper's chicken-heart model. To analyse the
downloaded model, start with the [analysis notebook](tutorials/dataset_workflows/chicken_heart.ipynb)
instead.

Extract `chicken_heart_training_inputs.zip` into your project folder. The archive
contains the four 10x count matrices, matching annotations, and the reference
coordinates used to select the tissue sections.

## Combine counts and annotations

```python
from pathlib import Path
import CytoBridge as cb

PROJECT_DIR = Path(".").resolve()
raw = PROJECT_DIR / "data" / "chicken_heart" / "raw"
prepared = PROJECT_DIR / "outputs" / "chicken_heart_input"
prepared.mkdir(parents=True, exist_ok=True)
reference_input = prepared / "chicken_heart_reference_input.h5ad"

cb.pp.prepare_chicken_heart_input(
    raw_dir=raw / "GSE149457_RAW",
    metadata_h5ad=raw / "chicken_heart_spatial_merged_with_meta.h5ad",
    aligned_reference_h5ad=raw / "heart_aligned_all_timepoints.h5ad",
    output_h5ad=reference_input,
    output_table=prepared / "model_input.csv",
    manifest_path=prepared / "preparation.json",
    graph_database=cb.pp.bundled_graph_database_path("chicken_heart"),
    repair_legacy_d7_left_right=False,
)
```

The returned input joins raw counts to the section annotations and selects the
spots used at D4, D7, D10, and D14. The preparation record lists the selected
observations and their count sources.

## Prepare coordinates for alignment

```python
cb.pp.prepare_chicken_heart_ot_input(
    input_h5ad=reference_input,
    output_h5ad=prepared / "input.h5ad",
    output_table=prepared / "ot_input.csv",
    manifest_path=prepared / "ot_input.json",
)
```

This prepares the original tissue coordinates, including the D7 orientation,
for a new alignment fit. It writes `outputs/chicken_heart_input/input.h5ad`.

## Train with the paper's settings

From `PROJECT_DIR`, run:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad outputs/chicken_heart_input/input.h5ad \
  --output-dir outputs/chicken_heart_trained --device cuda
```

This command preprocesses expression, fits spatial alignment and the LR edge
predictor, trains the dynamical model, and calculates the configured analyses.
The two preparation calls above assemble the study input. They have not already
performed those model-training steps.
