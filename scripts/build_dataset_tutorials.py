#!/usr/bin/env python3
"""Build the five public dataset tutorial notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nbformat

from CytoBridge.results.reproduction_chains import (
    describe_dataset_paper_steps,
    describe_dataset_run_steps,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
OWN_DATA_NOTEBOOK = ROOT / "docs" / "tutorials" / "your_data.ipynb"
SYNTHETIC_NOTEBOOK = (
    ROOT / "docs" / "tutorials" / "data_preparation" / "synthetic_preprocessing.ipynb"
)


@dataclass(frozen=True)
class Tutorial:
    dataset: str
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
                "LR-prior ablation and stVCR comparison",
                "lr_prior_ablation_stvcr.ipynb",
                "lr-prior-stvcr",
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
                "LR-prior ablation and stVCR comparison",
                "lr_prior_ablation_stvcr.ipynb",
                "lr-prior-stvcr",
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


def markdown(text: str, *, cell_id: str | None = None):
    cell = nbformat.v4.new_markdown_cell(text.strip())
    if cell_id is not None:
        cell["id"] = cell_id
    return cell


def code(text: str, *, cell_id: str | None = None):
    cell = nbformat.v4.new_code_cell(text.strip())
    if cell_id is not None:
        cell["id"] = cell_id
    return cell


def _show_placeholders(text: str) -> str:
    """Keep angle-bracket path placeholders visible in rendered Markdown."""

    return re.sub(r"(?<!`)(<[^<>\n]+>)(?!`)", r"`\1`", text)


def route_cells(
    rows: list[dict[str, str]],
    *,
    include_paper: bool,
    include_heading: bool = True,
) -> list:
    cells = []
    for row in rows:
        paper = f" ({row['paper_part']})" if include_paper else ""
        note_text = _show_placeholders(row.get("note", ""))
        note = f"\n{note_text}\n" if note_text else ""
        if row.get("entry_type") == "source":
            entry = (
                "File or directory to use (not a command): "
                f"`{row['code_or_command']}`"
            )
        else:
            language = (
                "python" if row["code_or_command"].startswith("from ") else "bash"
            )
            entry = f"""```{language}
{row['code_or_command']}
```"""
        heading = f"### {row['step']}{paper}\n" if include_heading else ""
        cells.append(
            markdown(
                f"""
{heading}
{entry}

**Input:** {_show_placeholders(row['reads'])}

**Output:** {_show_placeholders(row['writes'])}

**Continue with:** {_show_placeholders(row['next_step'])}
{note}
"""
            )
        )
    return cells


def build_notebook(tutorial: Tutorial):
    dataset_name = tutorial.dataset
    aligned_name = f"{dataset_name}_aligned.h5ad"
    chicken_setup = ""
    chicken_preparation = []
    if dataset_name == "chicken_heart":
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

The public GSE149457 10x matrices contain the raw counts and spot coordinates.
Two paper-retained H5AD files are also required: `METADATA_H5AD` supplies the
spot roster, region labels, and cell-type labels, and
`REFERENCE_ALIGNMENT_H5AD` supplies the matching row order used by the original
paper preparation. These two H5AD files are not generated by the public 10x
download or by the standard workflow.

The first call joins the raw counts to those retained annotations. The second
call starts from the raw coordinates, records them as `spatial_original`, and
writes `spatial_ot_input`; for D7 this applies the recorded 180-degree
pre-orientation before CytoBridge fits a new alignment.

**Input:** `RAW_10X_DIR`, `METADATA_H5AD`, and `REFERENCE_ALIGNMENT_H5AD`

**Output:** `RAW_H5AD`, containing counts, annotations, `spatial_original`, and
`spatial_ot_input`

**Continue with:** the **Run a new dataset from raw counts** command below
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
        graph_database=cb.pp.bundled_graph_database_path(DATASET_CONFIG),
        repair_legacy_d7_left_right=False,
    )
    cb.pp.prepare_chicken_heart_ot_input(
        input_h5ad=reference_input,
        output_h5ad=RAW_H5AD,
        output_table=PREPARATION_DIR / "chicken_heart_ot_input.csv",
        manifest_path=PREPARATION_DIR / "ot_input.json",
    )
else:
    print(
        "Raw-data assembly is off. Add the public 10x matrices and the two "
        "paper-retained H5AD files, then set RUN_RAW_DATA_ASSEMBLY = True."
    )
"""
            ),
        ]
    figure_lines = "\n".join(
        f"- [{label}](../paper_figures/{target})"
        for label, target, _workflow in tutorial.figure_links
    )
    run_rows = describe_dataset_run_steps(dataset_name)
    paper_route_cells = route_cells(
        describe_dataset_paper_steps(dataset_name), include_paper=True
    )
    cells = [
        markdown(
            f"""
# {tutorial.title}

This notebook is a guide to the `{dataset_name}` workflow. It shows how raw
data are mapped into the fields expected by CytoBridge, how a new model run is
passed to downstream analysis, and which later steps start from files saved
for the paper instead of the new run. Edit the paths in **Setup** before
starting.

For a small example that runs on generated data, see [Synthetic
preprocessing](../data_preparation/synthetic_preprocessing.ipynb). The [data and
checkpoint guide](../../data_checkpoints.md) lists inputs distributed outside
the package.
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
DATASET_CONFIG = {dataset_name!r}
RAW_H5AD = Path("data/{tutorial.raw_filename}")
OUTPUT_DIR = Path("tutorial_outputs/{dataset_name}")
PREPROCESS_ONLY_DIR = Path("tutorial_outputs/{dataset_name}_preprocess_only")
DOWNSTREAM_RERUN_DIR = Path("tutorial_outputs/{dataset_name}_downstream_rerun")
ALIGNED_H5AD = OUTPUT_DIR / "preprocess" / {aligned_name!r}
MODEL_DIR = OUTPUT_DIR / "training"
{chicken_setup}

RUN_TRAINING = False
RUN_PREPROCESS_ONLY = False
RUN_DOWNSTREAM = False
"""
        ),
        code(
            """
config, config_source = load_workflow_config(DATASET_CONFIG)
dataset = config["dataset"]
scientific = config["scientific"]
downstream = config["downstream"]
preprocess = config["preprocess"]
align = preprocess["align"]

spatial_obs_keys = align.get("spatial_obs_keys")
if spatial_obs_keys:
    spatial_source = ", ".join(f"obs[{key!r}]" for key in spatial_obs_keys)
else:
    spatial_source = f"obsm[{align.get('input_spatial_key', 'spatial')!r}]"

pd.DataFrame(
    {
        "setting": [
            "dataset",
            "configuration",
            "raw time column",
            "raw annotation column",
            "aligned annotation column",
            "raw count layer",
            "raw spatial coordinates",
            "aligned spatial coordinates",
            "model time values",
            "classifier neighbors",
        ],
        "value": [
            dataset["display_name"],
            config_source,
            preprocess["time_key"],
            preprocess["annotation_source"],
            dataset["annotation_key"],
            align.get("expression_layer", "X"),
            spatial_source,
            f"obsm[{dataset['spatial_key']!r}]",
            ", ".join(map(str, align["time_mapping"].values())),
            scientific["classifier_k"],
        ],
    }
)
"""
        ),
        markdown(
            """
## Start a new model run

The dataset configuration records the count layer, time mapping, spatial
coordinates, alignment settings, and model settings. Start here when fitting a
new model. The command reads the raw H5AD, writes the aligned H5AD, fits the
ligand--receptor edge predictor when the model uses one, trains CytoBridge, and
runs the analyses selected in the configuration. No separate preprocessing
command is needed first.
"""
        ),
        *chicken_preparation,
        route_cells([run_rows[0]], include_paper=False)[0],
        markdown(
            """
The next two cells show the same operation through the Python API. Leave
`RUN_TRAINING = False` when reading the documentation; set it to `True` only
after the paths above point to your data. The compact table shows the file
used by each step; the complete package plan is stored in `training_plan_text`
if you want to print it in Jupyter.
"""
        ),
        code(
            """
training_options = WorkflowOptions(
    input_h5ad=RAW_H5AD,
    output_dir=OUTPUT_DIR,
    train=True,
)
training_plan = build_workflow_plan(
    config,
    source=config_source,
    options=training_options,
)
training_plan_text = render_workflow_plan(training_plan)
pd.DataFrame(
    [
        {
            "step": "preprocess",
            "input": RAW_H5AD,
            "output": ALIGNED_H5AD,
        },
        {
            "step": "train",
            "input": ALIGNED_H5AD,
            "output": MODEL_DIR,
        },
        {
            "step": "downstream",
            "input": f"{ALIGNED_H5AD} + {MODEL_DIR}",
            "output": OUTPUT_DIR / "downstream",
        },
    ]
)
"""
        ),
        code(
            """
if RUN_TRAINING:
    if not RAW_H5AD.is_file():
        raise FileNotFoundError(f"Update RAW_H5AD before training: {RAW_H5AD}")
    training_result = run_workflow(config, options=training_options)
    training_result
else:
    print("Training is off. Set RUN_TRAINING = True to start a new model run.")
"""
        ),
        markdown(
            """
## Inspect the aligned data without training (optional)

Use this separate command only when you want to examine the aligned H5AD before
committing to a model fit. It writes to `PREPROCESS_ONLY_DIR` and does not fit
an edge predictor or a CytoBridge model. It is not an earlier step in the model
run above; when you are ready to train, use the first command from the raw H5AD.
"""
        ),
        route_cells(
            [run_rows[1]], include_paper=False, include_heading=False
        )[0],
        code(
            """
preprocess_only_options = WorkflowOptions(
    input_h5ad=RAW_H5AD,
    output_dir=PREPROCESS_ONLY_DIR,
    steps=("preprocess",),
)
preprocess_only_plan = build_workflow_plan(
    config,
    source=config_source,
    options=preprocess_only_options,
)
preprocess_only_plan_text = render_workflow_plan(preprocess_only_plan)
pd.DataFrame(
    [
        {
            "step": "preprocess only",
            "input": RAW_H5AD,
            "output": PREPROCESS_ONLY_DIR / "preprocess" / ALIGNED_H5AD.name,
        }
    ]
)
"""
        ),
        code(
            """
if RUN_PREPROCESS_ONLY:
    if not RAW_H5AD.is_file():
        raise FileNotFoundError(f"Update RAW_H5AD before preprocessing: {RAW_H5AD}")
    preprocess_only_result = run_workflow(
        config,
        options=preprocess_only_options,
    )
    preprocess_only_result
else:
    print(
        "Preprocessing-only run is off. Set RUN_PREPROCESS_ONLY = True "
        "to write an aligned H5AD without training."
    )
"""
        ),
        markdown(
            """
## Run downstream analysis again (optional)

The first command already runs downstream analysis. Use this section only when
you want to repeat it from the same aligned H5AD and fitted model. The rerun
writes to `DOWNSTREAM_RERUN_DIR`, leaving the original results unchanged.
"""
        ),
        route_cells(
            [run_rows[2]], include_paper=False, include_heading=False
        )[0],
        code(
            """
downstream_options = WorkflowOptions(
    aligned_h5ad=ALIGNED_H5AD,
    model_dir=MODEL_DIR,
    output_dir=DOWNSTREAM_RERUN_DIR,
    steps=("downstream",),
)
downstream_plan = build_workflow_plan(
    config,
    source=config_source,
    options=downstream_options,
)
downstream_plan_text = render_workflow_plan(downstream_plan)
pd.DataFrame(
    [
        {
            "step": "downstream",
            "input": f"{ALIGNED_H5AD}; {MODEL_DIR}",
            "output": DOWNSTREAM_RERUN_DIR / "downstream",
        }
    ]
)
"""
        ),
        code(
            """
if RUN_DOWNSTREAM:
    missing = [path for path in (ALIGNED_H5AD, MODEL_DIR) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing aligned data or model directory: {missing}")
    downstream_result = run_workflow(config, options=downstream_options)
    downstream_result
else:
    print("Downstream rerun is off. Set RUN_DOWNSTREAM = True to repeat it.")
"""
        ),
        markdown(
            f"""
## Paper figures

The headings state exactly where each calculation starts:

- **Continue from the model run above** reads the `OUTPUT_DIR` created here.
- **Start from the paper's saved files** reads the tables, arrays, or models
  retained from the exact paper analysis. It does not read the current
  `OUTPUT_DIR` unless the step says so.
- **Required paper files not included** names an input or page builder that is
  not shipped in this repository; no command is shown in its place.

Commands beginning with `python scripts/...` or `python -m scripts...` must be
run from the root of a cloned source repository. Each step names its input,
output, and the notebook that continues from it.

{figure_lines}
"""
        ),
        *paper_route_cells,
        markdown("## Saved files"),
        markdown(
            f"""
- Aligned data: `tutorial_outputs/{dataset_name}/preprocess/{aligned_name}`
- Training directory: `tutorial_outputs/{dataset_name}/training`
- Downstream directory: `tutorial_outputs/{dataset_name}/downstream`
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
    """Build the short route from an AnnData file to a package run."""

    cells = [
        markdown(
            """
# Run CytoBridge on your data

If you are fitting a model from raw data, use the single `--train` command in
this notebook. It performs preprocessing first, then training and downstream
analysis. You do not need to run a separate preprocessing command.

Choose the included dataset that uses the most similar species, count layer,
time layout, and spatial coordinates. Export its configuration, then change the
field names and analysis settings for your AnnData object. The same edited file
is used for preprocessing, training, and downstream analysis.

If you want to see preprocessing run before using a real file, begin with the
[small generated example](data_preparation/synthetic_preprocessing.ipynb).
"""
        ),
        markdown("## Choose an example configuration"),
        code(
            """
from pathlib import Path

import pandas as pd
from IPython.display import display

from CytoBridge.workflow import load_workflow_config

STARTING_CONFIG = "zebrafish"
CONFIG_PATH = Path("configs/my_dataset.json")
RAW_H5AD = Path("inputs/my_dataset_raw.h5ad")
RUN_ROOT = Path("outputs/my_dataset")
CUSTOM_LR_DATABASE = None  # or Path("inputs/my_ligand_receptor_table.csv")
RUN_WORKFLOW = False

CONFIG_TO_REVIEW = CONFIG_PATH if CONFIG_PATH.is_file() else STARTING_CONFIG
config, config_source = load_workflow_config(CONFIG_TO_REVIEW)
preprocess = config["preprocess"]
align = preprocess["align"]
dataset = config["dataset"]

spatial_obs_keys = align.get("spatial_obs_keys")
if spatial_obs_keys:
    spatial_source = ", ".join(f"obs[{key!r}]" for key in spatial_obs_keys)
else:
    spatial_source = f"obsm[{align.get('input_spatial_key', 'spatial')!r}]"

pd.DataFrame(
    {
        "field in the example": [
            "raw counts",
            "raw time",
            "raw annotation",
            "raw spatial coordinates",
            "aligned annotation",
            "aligned spatial coordinates",
        ],
        "AnnData location": [
            f"layers[{align.get('expression_layer', 'counts')!r}]",
            f"obs[{preprocess['time_key']!r}]",
            f"obs[{preprocess['annotation_source']!r}]",
            spatial_source,
            f"obs[{dataset['annotation_key']!r}]",
            f"obsm[{dataset['spatial_key']!r}]",
        ],
    }
)
"""
        ),
        markdown(
            """
The table initially shows the Zebrafish example. After exporting and editing
`configs/my_dataset.json`, rerun the notebook: it will load that file and show
your field names instead.
"""
        ),
        markdown(
            """
## Check the raw AnnData layout

CytoBridge expects observations in rows and genes in columns. Before running
the workflow, check that:

- `obs_names` are unique;
- the configured count layer contains finite, non-negative integer counts;
- the configured time and annotation columns exist in `obs` and contain no
  missing values;
- every source time appears in `preprocess.align.time_mapping` and every
  observed model time appears in the data;
- the configured spatial input contains two finite coordinates for every
  observation; and
- gene names match the symbols in the ligand-receptor CSV when that analysis is
  enabled.

The [small generated example](data_preparation/synthetic_preprocessing.ipynb)
constructs an AnnData object with this layout and runs preprocessing on it.
"""
        ),
        markdown(
            """
Export the example configuration:

```bash
cytobridge workflow --config zebrafish \\
  --export-config configs/my_dataset.json
```

In the exported JSON, change these exact fields:

- `dataset.name` and `dataset.annotation_key`;
- `preprocess.time_key` and `preprocess.annotation_source`;
- `preprocess.align.expression_layer`, `spatial_obs_keys` or
  `input_spatial_key`, and `time_mapping`;
- `scientific.classifier_k`, `alpha_spatial`, and `alpha_express`;
- `train.interaction_cutoff` and the training configuration when your model
  settings differ; and
- `downstream.observed`, `downstream.interpolated`, and
  `downstream.preferred_species_tag`.

The example configuration selects an LR database included with CytoBridge. To
use your own CSV instead, do not put its local path in `train.graph_database`.
Pass the file with `--graph-database` when fitting the edge predictor and with
`--lr-database` for downstream ligand--receptor analysis. The CSV must contain
`ligand` and `receptor` columns.

The five dataset notebooks display the exact raw and aligned fields used by
their included configurations.

Keep `steps.default` unchanged and `preprocess.enabled` set to `true` if you
want the single command below to run preprocessing, training, and downstream
analysis in order.
"""
        ),
        markdown(
            """
## Review the planned steps

```bash
cytobridge workflow --config configs/my_dataset.json --train \\
  --input-h5ad inputs/my_dataset_raw.h5ad \\
  --output-dir outputs/my_dataset --device cuda --check
```

`--check` shows the selected steps, settings, and intended paths without
starting the calculation. It does **not** open the H5AD or verify its columns,
layers, coordinates, or values. Those checks run when preprocessing starts, so
review the table above and inspect your AnnData before running the next command.
"""
        ),
        markdown(
            """
## Preprocess, train, and run downstream analysis

```bash
cytobridge workflow --config configs/my_dataset.json --train \\
  --input-h5ad inputs/my_dataset_raw.h5ad \\
  --output-dir outputs/my_dataset --device cuda
```

Training starts only when `--train` is present. With the exported configuration
unchanged, this command runs preprocessing, training, and downstream analysis
in order. It writes the aligned H5AD, model directory, result folders, summary
file, and PNG/PDF figures under `outputs/my_dataset`.

If you need your own LR table, use this complete version of the same command:

```bash
cytobridge workflow --config configs/my_dataset.json --train \\
  --input-h5ad inputs/my_dataset_raw.h5ad \\
  --graph-database inputs/my_ligand_receptor_table.csv \\
  --lr-database inputs/my_ligand_receptor_table.csv \\
  --output-dir outputs/my_dataset --device cuda
```
"""
        ),
        code(
            """
from CytoBridge.workflow import WorkflowOptions, run_workflow

if RUN_WORKFLOW:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Export and edit the configuration before starting: {CONFIG_PATH}"
        )
    if not RAW_H5AD.is_file():
        raise FileNotFoundError(f"Update RAW_H5AD before starting: {RAW_H5AD}")
    run_config, _ = load_workflow_config(CONFIG_PATH)
    run_options = WorkflowOptions(
        input_h5ad=RAW_H5AD,
        output_dir=RUN_ROOT,
        graph_database=CUSTOM_LR_DATABASE,
        lr_database=CUSTOM_LR_DATABASE,
        device="cuda",
        train=True,
    )
    run_result = run_workflow(run_config, options=run_options)
    run_result
else:
    print(
        "The full run is off. Update the paths, then set RUN_WORKFLOW = True "
        "to preprocess, train, and run downstream analysis."
    )
"""
        ),
        markdown(
            """
## Continue from an existing model

Use the aligned H5AD and model directory from the same run:

```bash
cytobridge workflow --config configs/my_dataset.json --step downstream \\
  --aligned-h5ad outputs/my_dataset/preprocess/my_dataset_aligned.h5ad \\
  --model-dir outputs/my_dataset/training \\
  --output-dir outputs/my_dataset_downstream_rerun --device cuda
```

Use a new output directory for a second downstream calculation. The
paper-figure commands state whether they read this new run or files retained
from the paper analysis. A paper redraw command does not automatically analyze
this new directory.
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


def build_synthetic_preprocessing_notebook():
    """Build a small, fully executable preprocessing example."""

    cells = [
        markdown(
            """
# CytoBridge synthetic preprocessing

This notebook creates a small spatial count matrix and runs
`CytoBridge.pp.preprocess`. It requires Python 3.10 or later, AnnData, and
`CytoBridge[preprocess]` installed in the current Jupyter kernel.

The example uses generated data. Replace the data-construction cell with a
dataset loader that provides raw counts, a time column, cell-type labels, and
spatial coordinates.
""",
            cell_id="intro",
        ),
        code(
            """
from __future__ import annotations

from io import BytesIO
import platform
from importlib.metadata import version

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from IPython.display import Image, display

import CytoBridge

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
    }
)

SEED = 42
rng = np.random.default_rng(SEED)
environment = {
    "cytobridge": version("CytoBridge"),
    "python": platform.python_version(),
    "seed": SEED,
}
environment
""",
            cell_id="setup",
        ),
        markdown(
            """
## 1. Create the input AnnData object

The input contains non-negative integer counts in `layers['counts']`, stage and
annotation columns in `obs`, and two spatial coordinates in
`obsm['spatial']`. The assertions check the array type, range, and dimensions
before preprocessing.
""",
            cell_id="raw-input",
        ),
        code(
            """
n_cells, n_genes = 72, 40
stages = np.repeat(np.array(["E0", "E1", "E2"]), n_cells // 3)
counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
counts[stages == "E1", :5] += 2
counts[stages == "E2", 5:10] += 3
counts[:, 39] = 0  # deterministic low-information marker used in the PCA feature check
spatial = np.column_stack(
    [
        np.linspace(0.0, 1.0, n_cells),
        rng.normal(0.0, 0.08, size=n_cells),
    ]
).astype(np.float32)

obs = pd.DataFrame(
    {
        "stage": stages,
        "Annotation": np.where(np.arange(n_cells) % 2 == 0, "TypeA", "TypeB"),
    },
    index=[f"Cell{i:03d}" for i in range(n_cells)],
)
var = pd.DataFrame(index=[f"Gene{i:03d}" for i in range(n_genes)])
adata = AnnData(X=counts.copy(), obs=obs, var=var)
adata.layers["counts"] = counts.copy()
adata.obsm["spatial"] = spatial

assert np.isfinite(counts).all() and (counts >= 0).all()
assert np.allclose(counts, np.rint(counts), rtol=0.0, atol=0.0)
assert adata.obsm["spatial"].shape == (n_cells, 2)
adata.obs.groupby(["stage", "Annotation"], observed=True).size().rename("cells")
""",
            cell_id="make-data",
        ),
        markdown(
            """
## 2. Run preprocessing

`expression_layer='counts'` selects the raw source. Strict validation checks
the selected layer before `normalize_total(target_sum=10000)` and `log1p`.
Highly variable genes define the PCA fit without removing genes from the
expression matrix, and requested features are added to that fit when needed.
""",
            cell_id="preprocessing",
        ),
        code(
            """
processed = CytoBridge.pp.preprocess(
    adata.copy(),
    time_key="stage",
    time_mapping={"E0": 0.0, "E1": 1.0, "E2": 2.0},
    n_top_genes=20,
    n_pcs=8,
    expression_layer="counts",
    raw_count_validation="strict",
    required_latent_features=["Gene000", "Gene039"],
)

{
    "X_shape": processed.X.shape,
    "X_latent_shape": processed.obsm["X_latent"].shape,
    "mapped_times": sorted(
        processed.obs["time_point_processed"].unique().tolist()
    ),
}
""",
            cell_id="run-preprocess",
        ),
        markdown(
            """
## 3. Check preprocessing metadata and arrays

The processed AnnData records the expression source, validation mode,
transformation order, time mapping, PCA feature count, and PCA center in
`uns['preprocess_info']`. The checks below also verify array shapes, finite
values, mapped times, and requested PCA features.
""",
            cell_id="metadata",
        ),
        code(
            """
info = processed.uns["preprocess_info"]
assert info["expression_source"] == "layers['counts']"
assert info["raw_count_validation_effective"] == "strict"
assert info["transformation_sequence"] == ["normalize_total", "log1p"]
assert processed.obsm["X_latent"].shape == (n_cells, 8)
assert processed.var["pca_center"].shape == (n_genes,)
assert np.isfinite(processed.obsm["X_latent"]).all()
assert np.isfinite(processed.var["pca_center"].to_numpy()).all()
assert np.allclose(processed.obsm["X_latent"].mean(axis=0), 0.0, atol=1e-5)
assert sorted(processed.obs["time_point_processed"].unique().tolist()) == [
    0.0,
    1.0,
    2.0,
]
assert all(
    bool(processed.var.loc[name, "highly_variable"])
    for name in ["Gene000", "Gene039"]
)

preprocess_summary = {
    "expression_source": info["expression_source"],
    "raw_count_validation": info["raw_count_validation_effective"],
    "transformations": info["transformation_sequence"],
    "n_latent_fit_features": info["n_latent_fit_features"],
    "latent_shape": processed.obsm["X_latent"].shape,
    "latent_all_finite": bool(np.isfinite(processed.obsm["X_latent"]).all()),
    "mapped_times": sorted(
        processed.obs["time_point_processed"].unique().tolist()
    ),
    "required_features_in_pca": ["Gene000", "Gene039"],
}
preprocess_summary
""",
            cell_id="inspect-metadata",
        ),
        markdown(
            """
## 4. Plot the processed coordinates

The left panel shows the spatial coordinates supplied in the input AnnData.
The right panel uses the first two columns of the `X_latent` matrix produced by
the preprocessing call above.
""",
            cell_id="plot-result",
        ),
        code(
            """
stage_colors = {"E0": "#59616A", "E1": "#07838B", "E2": "#D28C3C"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

for stage in stage_colors:
    mask = processed.obs["stage"].astype(str).to_numpy() == stage
    axes[0].scatter(
        processed.obsm["spatial"][mask, 0],
        processed.obsm["spatial"][mask, 1],
        s=22,
        color=stage_colors[stage],
        linewidth=0,
        label=stage,
    )
    axes[1].scatter(
        processed.obsm["X_latent"][mask, 0],
        processed.obsm["X_latent"][mask, 1],
        s=22,
        color=stage_colors[stage],
        linewidth=0,
        label=stage,
    )

for letter, ax in zip(("a", "b"), axes):
    ax.text(
        -0.15,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set(
    title="Input spatial coordinates",
    xlabel="Spatial coordinate 1",
    ylabel="Spatial coordinate 2",
)
axes[1].set(
    title="Processed latent coordinates",
    xlabel="PC 1",
    ylabel="PC 2",
)
axes[1].legend(frameon=False, title="Stage", markerscale=0.9)
fig.tight_layout()
image_buffer = BytesIO()
fig.savefig(image_buffer, format="png", dpi=144, bbox_inches="tight")
plt.close(fig)
display(Image(data=image_buffer.getvalue()))
""",
            cell_id="plot-processed",
        ),
        markdown(
            """
## 5. Validate reuse of a processed object

Passing the transformed matrix through the default preprocessing path a second
time raises a `ValueError`. Start a new preprocessing run from the raw count
layer.
""",
            cell_id="input-validation",
        ),
        code(
            """
try:
    CytoBridge.pp.preprocess(
        processed.copy(),
        time_key="stage",
        n_top_genes=20,
        n_pcs=8,
    )
except ValueError as exc:
    message = str(exc)
    assert "double-transform" in message
    print(message.splitlines()[0])
else:
    raise AssertionError("Expected preprocessing to reject transformed X")
""",
            cell_id="double-transform-check",
        ),
        markdown(
            """
## Outputs

`processed` contains the normalized expression matrix, `obsm['X_latent']`,
PCA metadata, mapped numeric times, and the original spatial coordinates.
`preprocess_summary` collects the fields checked in this example, and the plot
above is drawn directly from `processed`.
""",
            cell_id="outputs",
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
        path = NOTEBOOK_DIR / f"{tutorial.dataset}.ipynb"
        nbformat.write(build_notebook(tutorial), path)
        print(path.relative_to(ROOT))
    nbformat.write(build_own_data_notebook(), OWN_DATA_NOTEBOOK)
    print(OWN_DATA_NOTEBOOK.relative_to(ROOT))
    SYNTHETIC_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_synthetic_preprocessing_notebook(), SYNTHETIC_NOTEBOOK)
    print(SYNTHETIC_NOTEBOOK.relative_to(ROOT))


if __name__ == "__main__":
    main()
