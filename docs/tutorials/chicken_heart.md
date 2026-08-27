# Developing chicken heart

**Notebook:** {download}`chicken_heart.ipynb <dataset_workflows/chicken_heart.ipynb>`

The notebook uses the installed package for the GSE149457 D4/D7/D10/D14
workflow. It keeps data preparation, a new training run, an existing model, and
downstream output in separate directories. All execution switches are off in
the checked-in copy.

## Inputs

Data preparation needs the four raw 10x matrices, the metadata H5AD, and the
reference alignment H5AD. To reuse a fitted model, provide its aligned H5AD and
model directory instead. The ligand-receptor table is loaded from the
`chicken_heart` package preset.

## Prepare and validate data

The preparation calls are public `CytoBridge.pp` APIs:

```python
from pathlib import Path

import CytoBridge as cb

database = cb.pp.bundled_graph_database_path("chicken_heart")
cb.pp.prepare_chicken_heart_input(
    raw_dir=Path("inputs/GSE149457_RAW"),
    metadata_h5ad=Path("inputs/chicken_heart_spatial_merged_with_meta.h5ad"),
    aligned_reference_h5ad=Path("inputs/heart_aligned_all_timepoints.h5ad"),
    output_h5ad=Path("outputs/prepared/chicken_heart_reference_input.h5ad"),
    output_table=Path("outputs/prepared/model_input.csv"),
    manifest_path=Path("outputs/prepared/preparation.json"),
    graph_database=database,
    repair_legacy_d7_left_right=False,
)
cb.pp.prepare_chicken_heart_ot_input(
    input_h5ad=Path("outputs/prepared/chicken_heart_reference_input.h5ad"),
    output_h5ad=Path("outputs/prepared/chicken_heart_ot_input.h5ad"),
    output_table=Path("outputs/prepared/chicken_heart_ot_input.csv"),
    manifest_path=Path("outputs/prepared/ot_input.json"),
)
```

The optional legacy repair remains `False` for current reference coordinates.
The OT adapter applies the fixed D7 raw-coordinate preorientation used by the
package preset and leaves D4, D10, and D14 raw coordinates unchanged.

Use `validate_prepared_chicken_heart_input` for the reference-count intermediate
and `validate_chicken_heart_ot_input` for the workflow input.

## Run the workflow

`load_workflow_config("chicken_heart")` supplies the alignment, graph,
training, and downstream settings. The notebook first renders a plan. Training
uses `TRAIN_OUTPUT_DIR`; an existing checkpoint stays in `MODEL_DIR`.

The equivalent command for a deliberate new run is:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad outputs/prepared/chicken_heart_ot_input.h5ad \
  --output-dir outputs/chicken_heart_training \
  --device cuda
```

## Outputs

The workflow writes the aligned H5AD under `preprocess/`, model files under
`training/`, and tables, figures, generated slices, and `summary.json` under
`downstream/`.
