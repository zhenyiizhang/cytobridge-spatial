# Chicken-heart alignment sensitivity

This analysis compares the four chicken-heart sections under coordinate
perturbations. The [tutorial](../../../docs/tutorials/paper_figures/chicken_heart_alignment.md)
shows how to train the models, calculate agreement and draw S12–S13.

## Reproduce the figures

Run from the repository root:

```bash
python -m pip install numpy pandas scipy matplotlib pymupdf
python reproduction/chicken_heart/alignment_sensitivity/plot_sensitivity.py \
  --output-dir outputs/heart_alignment
```

The command draws both figures from the numerical inputs below and writes PDF
and PNG files. A two-page PDF is also written beside the output directory. Use
Arial to match the manuscript. No previously rendered image is used to calculate
or draw a panel.

| Panels | Input | Calculation |
| --- | --- | --- |
| S7a, S8 | `summary/plot_inputs.npz` | Plot the original, perturbed and aligned coordinates, matching cells by observation identifier |
| S7b, coordinates | `summary/coordinate_metrics.csv` | Residual after centering and a proper rotation, divided by the reference section radius |
| S7b, velocity | `summary/velocity_metrics_pooled.csv` | Median cellwise cosine for the full and interaction velocity in physical space |
| S7b, interaction weights | `summary/interaction_metrics.csv` | Normalized weighted Jaccard overlap over the union of directed edges |
| S7c | `input_manifest.json` | Applied displacement magnitude and absolute rotation angle for each stage |

`plotting/heart.py` contains the panel layout. `plot_sensitivity.py` reads the
actual perturbation amplitudes, selects the current control, and applies the
accepted labels and styling. `summarize_results.py` recalculates the ranges
reported in the SI and response.

## How the inputs were calculated

The calculation used the same 3,550 spots at D4, D7, D10 and D14 in every
condition. Perturbations changed `obsm['spatial_ot_input']`, retaining counts,
annotations, cell identities and the D7 pre-orientation. For each condition,
preprocessing, alignment, edge-classifier fitting, complete model training and
downstream analysis were repeated with seed 42.

To calculate new inputs, start with `data/chicken_heart/aligned.h5ad` from the
chicken-heart download. It contains counts and the pre-oriented section
coordinates in `spatial_ot_input`. The command below trains the reference
model and calculates its downstream fields. If you already ran this same
workflow, use that output directory as `--reference-run` instead.

```bash
PYTHONHASHSEED=0 python -m CytoBridge.cli workflow \
  --config CytoBridge/workflow_configs/chicken_heart.json --train \
  --input-h5ad data/chicken_heart/aligned.h5ad \
  --output-dir outputs/chicken_heart_reference --device cuda:0
```

Prepare the six perturbed inputs, then run the unperturbed repeat and six
perturbed conditions. Training is substantial. Each condition repeats
alignment, edge prediction, model training and downstream calculation.

```bash
python reproduction/chicken_heart/alignment_sensitivity/calculate.py prepare \
  --input-h5ad data/chicken_heart/aligned.h5ad --output-dir outputs/heart_sensitivity
```

```bash
python reproduction/chicken_heart/alignment_sensitivity/calculate.py train \
  --output-dir outputs/heart_sensitivity --device cuda:0
```

The default runs all seven conditions sequentially. To distribute them across
GPUs, run separate commands with `--variant translate_low` (or another
condition listed by `--help`) and the desired `--device`. Each condition writes
to its own directory under `outputs/heart_sensitivity/runs/`.

Calculate coordinate, velocity and attention agreement from those new runs:

```bash
python reproduction/chicken_heart/alignment_sensitivity/calculate.py compare \
  --output-dir outputs/heart_sensitivity --reference-run outputs/chicken_heart_reference
```

Draw both figures using the newly calculated tables and coordinate arrays:

```bash
python reproduction/chicken_heart/alignment_sensitivity/plot_sensitivity.py \
  --results-dir outputs/heart_sensitivity/summary \
  --manifest outputs/heart_sensitivity/input_manifest.json \
  --output-dir outputs/heart_sensitivity/figures
```

`calculate.py` calls the original calculations in `server_code/`, with the
input and output paths supplied above:

1. `prepare_and_run.py` reads the original H5AD and generates perturbed inputs.
   `calculate.py` sets the 1×/2× translations and 1°/3° rotations, then trains
   the seven conditions.
2. Each workflow writes `preprocess/chicken_heart_aligned.h5ad`, the trained
   models under `training/`, and velocity and attention arrays under `downstream/`.
3. `compare_runs.py` reads those outputs and creates the comparison CSV tables.
4. `export_plot_inputs.py` collects coordinates in the same observation order
   for plotting.

The original run settings are recorded in `experiment.json` and
`input_manifest.json`. The saved numerical results reproduce the paper's
reported values. Repeating model training gives a new set of estimates.
