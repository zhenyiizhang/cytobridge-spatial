#!/usr/bin/env python3
"""Build the five public dataset tutorial notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"


@dataclass(frozen=True)
class Tutorial:
    preset: str
    title: str
    raw_filename: str
    figure_links: tuple[tuple[str, str], ...]


TUTORIALS = (
    Tutorial(
        "zebrafish",
        "Zebrafish embryogenesis",
        "zebrafish_raw.h5ad",
        (
            ("Supplementary Figures S31–S38", "zebrafish_si_s31_s38.ipynb"),
            ("Supplementary Figure S43", "zebrafish_attention.ipynb"),
        ),
    ),
    Tutorial(
        "mosta",
        "MOSTA mouse organogenesis",
        "mosta_raw.h5ad",
        (
            ("Main Figure 4", "main_figure_4.ipynb"),
            ("Supplementary Figures S11–S18", "mosta_figures.ipynb"),
        ),
    ),
    Tutorial(
        "arista",
        "ARISTA salamander brain regeneration",
        "arista_raw.h5ad",
        (
            ("Main Figure 5", "main_figure_5.ipynb"),
            ("Supplementary Figures S19–S24", "arista_figures.ipynb"),
            ("Supplementary Figure S42", "arista_local_domains.ipynb"),
        ),
    ),
    Tutorial(
        "admouse",
        "AD mouse brain",
        "admouse_raw.h5ad",
        (
            ("Interaction-prior ablation", "lr_prior_ablation_stvcr.ipynb"),
            ("Five-dataset benchmark", "loto_benchmark.ipynb"),
            ("Training histories", "training_histories.ipynb"),
        ),
    ),
    Tutorial(
        "chicken_heart",
        "Developing chicken heart",
        "chicken_heart_raw.h5ad",
        (
            ("Interaction-prior ablation", "lr_prior_ablation_stvcr.ipynb"),
            ("Five-dataset benchmark", "loto_benchmark.ipynb"),
            ("Training histories", "training_histories.ipynb"),
        ),
    ),
)


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


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
        for label, target in tutorial.figure_links
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
    output_dir=OUTPUT_DIR / "downstream",
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

The figure notebooks load the corresponding packaged result tables and save
the paper PDFs and PNGs:

{figure_lines}
"""
        ),
        markdown("## Saved files"),
        code(
            """
pd.DataFrame(
    {
        "file or directory": [
            "aligned data",
            "training directory",
            "downstream directory",
        ],
        "path": [
            str(ALIGNED_H5AD),
            str(MODEL_DIR),
            str(OUTPUT_DIR / "downstream"),
        ],
    }
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


if __name__ == "__main__":
    main()
