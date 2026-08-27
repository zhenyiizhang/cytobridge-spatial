#!/usr/bin/env python3
"""Build the five public dataset tutorial notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat

from CytoBridge.results.reproduction_chains import (
    describe_dataset_artifact_chain,
    describe_dataset_paper_chain,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
OWN_DATA_NOTEBOOK = ROOT / "docs" / "tutorials" / "your_data.ipynb"


@dataclass(frozen=True)
class Tutorial:
    preset: str
    title: str
    raw_filename: str
    figure_links: tuple[tuple[str, str, str], ...]


TUTORIALS = (
    Tutorial(
        "zebrafish",
        "Zebrafish embryogenesis",
        "zebrafish_raw.h5ad",
        (
            (
                "Supplementary Figures S31–S38",
                "zebrafish_si_s31_s38.ipynb",
                "zebrafish-si",
            ),
            (
                "Supplementary Figure S43",
                "zebrafish_attention.ipynb",
                "zebrafish-attention",
            ),
        ),
    ),
    Tutorial(
        "mosta",
        "MOSTA mouse organogenesis",
        "mosta_raw.h5ad",
        (
            ("Main Figure 4", "main_figure_4.ipynb", "main-figure-4"),
            (
                "Supplementary Figures S11–S18",
                "mosta_figures.ipynb",
                "mosta-reference-pages",
            ),
        ),
    ),
    Tutorial(
        "arista",
        "ARISTA salamander brain regeneration",
        "arista_raw.h5ad",
        (
            (
                "Main Figure 5",
                "main_figure_5.ipynb",
                "main-figure-5-reference",
            ),
            (
                "Supplementary Figures S19–S24",
                "arista_figures.ipynb",
                "arista-lr",
            ),
            (
                "Supplementary Figure S42",
                "arista_local_domains.ipynb",
                "arista-local-domains",
            ),
        ),
    ),
    Tutorial(
        "admouse",
        "AD mouse brain",
        "admouse_raw.h5ad",
        (
            (
                "Interaction-prior ablation",
                "lr_prior_ablation_stvcr.ipynb",
                "interaction-evidence",
            ),
            ("Five-dataset benchmark", "loto_benchmark.ipynb", "loto-benchmark"),
            (
                "Training histories",
                "training_histories.ipynb",
                "training-histories",
            ),
        ),
    ),
    Tutorial(
        "chicken_heart",
        "Developing chicken heart",
        "chicken_heart_raw.h5ad",
        (
            (
                "Interaction-prior ablation",
                "lr_prior_ablation_stvcr.ipynb",
                "interaction-evidence",
            ),
            ("Five-dataset benchmark", "loto_benchmark.ipynb", "loto-benchmark"),
            (
                "Training histories",
                "training_histories.ipynb",
                "training-histories",
            ),
        ),
    ),
)


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def route_cells(rows: list[dict[str, str]], *, include_paper: bool) -> list:
    cells = []
    for number, row in enumerate(rows, start=1):
        paper = f"\n**Paper:** {row['paper_part']}\n" if include_paper else ""
        cells.append(
            markdown(
                f"""
### Step {number}: {row['step']}
{paper}
```text
{row['code_or_command']}
```

**Reads:** `{row['reads']}`

**Writes:** `{row['writes']}`

**Next:** `{row['next_step']}`

**Availability:** {row['availability']}
"""
            )
        )
    return cells


def build_notebook(tutorial: Tutorial):
    preset = tutorial.preset
    aligned_name = f"{preset}_aligned.h5ad"
    chicken_setup = ""
    chicken_preparation = []
    if preset == "chicken_heart":
        chicken_setup = """

import CytoBridge as cb

RAW_10X_DIR = Path("data/GSE149457_RAW")
METADATA_H5AD = Path("data/chicken_heart_spatial_merged_with_meta.h5ad")
REFERENCE_ALIGNMENT_H5AD = Path("data/heart_aligned_all_timepoints.h5ad")
PREPARATION_DIR = RAW_H5AD.parent / "chicken_heart_preparation"
RUN_RAW_DATA_ASSEMBLY = False
"""
        chicken_preparation = [
            markdown(
                """
### Assemble the chicken-heart H5AD

The downloaded 10x matrices are first matched to the reference spot roster.
The second call writes the `spatial_ot_input` coordinates expected by the
alignment preset.
"""
            ),
            code(
                """
if RUN_RAW_DATA_ASSEMBLY:
    PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
    reference_input = PREPARATION_DIR / "chicken_heart_reference_input.h5ad"
    cb.pp.prepare_chicken_heart_input(
        raw_dir=RAW_10X_DIR,
        metadata_h5ad=METADATA_H5AD,
        aligned_reference_h5ad=REFERENCE_ALIGNMENT_H5AD,
        output_h5ad=reference_input,
        output_table=PREPARATION_DIR / "model_input.csv",
        manifest_path=PREPARATION_DIR / "preparation.json",
        graph_database=cb.pp.bundled_graph_database_path(PRESET),
        repair_legacy_d7_left_right=False,
    )
    cb.pp.prepare_chicken_heart_ot_input(
        input_h5ad=reference_input,
        output_h5ad=RAW_H5AD,
        output_table=PREPARATION_DIR / "chicken_heart_ot_input.csv",
        manifest_path=PREPARATION_DIR / "ot_input.json",
    )
else:
    print("Raw-data assembly is off. Set RUN_RAW_DATA_ASSEMBLY = True to run it.")
"""
            ),
        ]
    figure_lines = "\n".join(
        f"- [{label}](../paper_figures/{target})"
        for label, target, _workflow in tutorial.figure_links
    )
    artifact_route_cells = route_cells(
        describe_dataset_artifact_chain(preset), include_paper=False
    )
    paper_route_cells = route_cells(
        describe_dataset_paper_chain(preset), include_paper=True
    )
    cells = [
        markdown(
            f"""
# {tutorial.title}

This notebook runs the packaged `{preset}` workflow from data preparation
through downstream analysis. Edit the paths in **Setup**, then enable the run
switches for the steps you need. The saved outputs below come from the packaged
preset and do not require the external dataset.
"""
        ),
        markdown("## Setup"),
        code(
            f"""
from pathlib import Path

import pandas as pd
from IPython.display import display

from CytoBridge.workflow import (
    WorkflowOptions,
    build_workflow_plan,
    load_workflow_config,
    render_workflow_plan,
    run_workflow,
)
PRESET = {preset!r}
RAW_H5AD = Path("data/{tutorial.raw_filename}")
OUTPUT_DIR = Path("tutorial_outputs/{preset}")
ALIGNED_H5AD = OUTPUT_DIR / "preprocess" / {aligned_name!r}
MODEL_DIR = OUTPUT_DIR / "training"
{chicken_setup}

RUN_PREPARATION = False
RUN_PREPROCESS_AND_TRAIN = False
RUN_DOWNSTREAM = False
"""
        ),
        code(
            """
config, preset_source = load_workflow_config(PRESET)
dataset = config["dataset"]
scientific = config["scientific"]
downstream = config["downstream"]

pd.DataFrame(
    {
        "setting": [
            "dataset",
            "preset",
            "raw time column",
            "cell annotation",
            "observed training times",
            "classifier neighbors",
        ],
        "value": [
            dataset["display_name"],
            preset_source,
            config["preprocess"]["time_key"],
            dataset["annotation_key"],
            ", ".join(map(str, downstream["observed"])),
            scientific["classifier_k"],
        ],
    }
)
"""
        ),
        markdown(
            """
## Files passed from one step to the next

These are the handoffs used below. Training reads the aligned H5AD and edge
predictor written by preprocessing. Downstream analysis then reads that same
aligned H5AD together with the complete training directory.
"""
        ),
        *artifact_route_cells,
        markdown(
            """
## Data preparation

The preset records the count layer, time mapping, spatial coordinates, and
alignment settings used for this dataset. The plan below shows the input and
output paths before any long-running work starts.
"""
        ),
        *chicken_preparation,
        code(
            """
preparation_options = WorkflowOptions(
    input_h5ad=RAW_H5AD,
    output_dir=OUTPUT_DIR,
    steps=("preprocess",),
)
preparation_plan = build_workflow_plan(
    config,
    source=preset_source,
    options=preparation_options,
)
print(render_workflow_plan(preparation_plan))
"""
        ),
        code(
            """
if RUN_PREPARATION:
    if not RAW_H5AD.is_file():
        raise FileNotFoundError(f"Update RAW_H5AD before preprocessing: {RAW_H5AD}")
    preparation_result = run_workflow(config, options=preparation_options)
    preparation_result
else:
    print("Data preparation is off. Set RUN_PREPARATION = True to run it.")
"""
        ),
        markdown(
            """
## Training

The full training run starts from the raw H5AD, writes the aligned data, fits
the interaction edge predictor when the preset requires one, and trains the
CytoBridge model. A production run requires a CUDA-capable environment.
"""
        ),
        code(
            """
training_options = WorkflowOptions(
    input_h5ad=RAW_H5AD,
    output_dir=OUTPUT_DIR,
    steps=("preprocess", "train"),
    train=True,
)
training_plan = build_workflow_plan(
    config,
    source=preset_source,
    options=training_options,
)
print(render_workflow_plan(training_plan))
"""
        ),
        code(
            """
if RUN_PREPROCESS_AND_TRAIN:
    if not RAW_H5AD.is_file():
        raise FileNotFoundError(f"Update RAW_H5AD before training: {RAW_H5AD}")
    training_result = run_workflow(config, options=training_options)
    training_result
else:
    print("Training is off. Set RUN_PREPROCESS_AND_TRAIN = True to start it.")
"""
        ),
        markdown(
            """
## Downstream analysis

Downstream analysis uses the aligned H5AD and fitted model from the training
directory. The dataset preset supplies the interpolation times, classifier
settings, trajectory simulation, growth analysis, and ligand–receptor options.
"""
        ),
        code(
            """
downstream_options = WorkflowOptions(
    aligned_h5ad=ALIGNED_H5AD,
    model_dir=MODEL_DIR,
    output_dir=OUTPUT_DIR,
    steps=("downstream",),
)
downstream_plan = build_workflow_plan(
    config,
    source=preset_source,
    options=downstream_options,
)
print(render_workflow_plan(downstream_plan))
"""
        ),
        code(
            """
if RUN_DOWNSTREAM:
    missing = [path for path in (ALIGNED_H5AD, MODEL_DIR) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained artifacts: {missing}")
    downstream_result = run_workflow(config, options=downstream_options)
    downstream_result
else:
    print("Downstream analysis is off. Set RUN_DOWNSTREAM = True to run it.")
"""
        ),
        markdown(
            f"""
## Paper figures

The steps below show exactly where this dataset's standard downstream output
continues into manuscript calculations. A step marked `provenance break` means
that a related calculation exists but the manuscript page cannot yet be traced
to one exact command and input set.

{figure_lines}
"""
        ),
        *paper_route_cells,
        markdown("## Saved files"),
        markdown(
            f"""
- Aligned data: `tutorial_outputs/{preset}/preprocess/{aligned_name}`
- Training directory: `tutorial_outputs/{preset}/training`
- Downstream directory: `tutorial_outputs/{preset}/downstream`
"""
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def build_own_data_notebook():
    """Build the short route from an AnnData file to a complete package run."""

    cells = [
        markdown(
            """
# Run CytoBridge on your data

Start with the packaged dataset closest to your experiment, export its config,
and edit the data fields and analysis settings. The same config is then used
for preprocessing, training, downstream calculations, and standard figures.
"""
        ),
        markdown("## Check the input fields"),
        code(
            """
import pandas as pd
from IPython.display import display

pd.DataFrame(
    [
        ("expression", "AnnData X or the count layer named in the config"),
        ("time", "one obs column with a value for every cell"),
        ("cell type", "one obs column used to label generated cells"),
        ("spatial coordinates", "two obs columns or one obsm matrix"),
    ],
    columns=["input", "location"],
)
"""
        ),
        markdown(
            """
The raw count layer, time mapping, cell-type column, and spatial coordinate
columns must agree with the workflow config. Do not rename fields after model
training; the aligned H5AD and model directory are one matched pair.
"""
        ),
        markdown("## Export a starting config"),
        code(
            """
from pathlib import Path

BASE_PRESET = "zebrafish"
CONFIG_PATH = Path("configs/my_dataset.json")
RAW_H5AD = Path("inputs/my_dataset_raw.h5ad")
RUN_ROOT = Path("outputs/my_dataset")

print(
    f"cytobridge workflow --config {BASE_PRESET} "
    f"--export-config {CONFIG_PATH}"
)
"""
        ),
        markdown(
            """
Run the printed command once. In the exported JSON, change these entries before
starting a fit:

- `dataset.name`, `display_name`, and `annotation_key`;
- `preprocess.time_key`, `annotation_source`, count layer, coordinates, and
  `align.time_mapping`;
- `scientific.classifier_k` and the spatial/expression loss weights;
- the training profile, interaction distance, LR database, and predictor
  settings; and
- downstream observed/intermediate times and species tag.

The dataset notebooks show the settings used for the five paper datasets.
"""
        ),
        markdown("## Inspect the run before starting"),
        code(
            """
print(
    "cytobridge workflow "
    f"--config {CONFIG_PATH} --train --input-h5ad {RAW_H5AD} "
    f"--output-dir {RUN_ROOT} --device cuda --dry-run"
)
"""
        ),
        markdown(
            """
The dry run prints the data keys, training profile, preprocessing outputs,
model directory, downstream time grid, and enabled analyses. Fix missing or
incorrect fields in the JSON before removing `--dry-run`.
"""
        ),
        markdown("## Preprocess, train, and run downstream analysis"),
        code(
            """
print(
    "cytobridge workflow "
    f"--config {CONFIG_PATH} --train --input-h5ad {RAW_H5AD} "
    f"--output-dir {RUN_ROOT} --device cuda"
)
"""
        ),
        markdown(
            """
Training is enabled only by `--train`. The command writes the aligned H5AD,
the six-stage model directory, the downstream result folders, a summary file,
and standard PNG/PDF figures under the run root.
"""
        ),
        markdown("## Continue from an existing model"),
        code(
            """
ALIGNED_H5AD = RUN_ROOT / "preprocess" / "my_dataset_aligned.h5ad"
MODEL_DIR = RUN_ROOT / "training"
NEW_DOWNSTREAM = Path("outputs/my_dataset_downstream_rerun")

print(
    "cytobridge workflow "
    f"--config {CONFIG_PATH} --step downstream "
    f"--aligned-h5ad {ALIGNED_H5AD} --model-dir {MODEL_DIR} "
    f"--output-dir {NEW_DOWNSTREAM} --device cuda"
)
"""
        ),
        markdown(
            """
Use a new output directory for a second downstream run. Paper-figure commands
consume their documented compact schemas; they are not a shortcut for turning
an arbitrary new downstream directory into a manuscript page. Use
`cytobridge figure explain <name>` to check that boundary before reusing one.
"""
        ),
        markdown("## Expected output locations"),
        code(
            """
with pd.option_context("display.max_colwidth", None):
    display(
        pd.DataFrame(
            {
                "output": [
                    "aligned data",
                    "model directory",
                    "downstream summary",
                    "standard figures",
                ],
                "path": [
                    RUN_ROOT / "preprocess" / "my_dataset_aligned.h5ad",
                    RUN_ROOT / "training",
                    RUN_ROOT / "downstream" / "summary.json",
                    RUN_ROOT / "downstream" / "figures",
                ],
            }
        )
    )
"""
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for tutorial in TUTORIALS:
        path = NOTEBOOK_DIR / f"{tutorial.preset}.ipynb"
        nbformat.write(build_notebook(tutorial), path)
        print(path.relative_to(ROOT))
    nbformat.write(build_own_data_notebook(), OWN_DATA_NOTEBOOK)
    print(OWN_DATA_NOTEBOOK.relative_to(ROOT))


if __name__ == "__main__":
    main()
