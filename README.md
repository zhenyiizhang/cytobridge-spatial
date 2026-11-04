# CytoBridge

CytoBridge reconstructs cell-state and spatial dynamics from transcriptomic
measurements collected at several times. Use it to simulate intermediate
cell populations and examine their growth and cell–cell interactions.

[Documentation](https://cytobridge-spatial.readthedocs.io/) ·
[Tutorials](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/index.html) ·
[API reference](https://cytobridge-spatial.readthedocs.io/en/latest/api/index.html) ·
[Issues](https://github.com/zhenyiizhang/cytobridge-spatial/issues)

## Installation

CytoBridge supports Python 3.10 and 3.11.

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial
python -m pip install -e '.[spatial,velocity]'
```

This combination supports the default spatial workflow from preprocessing and
training through velocity analysis and interactive downstream figures.

If you only need part of the package, install the matching optional
dependencies:

```bash
python -m pip install -e '.[preprocess]'
python -m pip install -e '.[train]'
python -m pip install -e '.[plot]'
python -m pip install -e '.[notebook,velocity]'
```

Install a PyTorch build that matches the CUDA version available on the target
machine.

Check the installation without starting a workflow:

```bash
cytobridge --version
cytobridge doctor
```

## Quick start

Start with the [chicken-heart analysis tutorial](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/dataset_workflows/chicken_heart.html).
It loads the trained model, computes velocity, growth, and attention, then
plots the returned arrays and tables. Each step uses the public Python API.

To work with your own experiment, follow
[Train a model](https://cytobridge-spatial.readthedocs.io/en/latest/training.html).
That guide explains expression preprocessing, spatial alignment, LR graph
construction, and model fitting as separate Python calls, then shows how to
continue with analysis.

For a CPU example without external data, run the
[preprocessing notebook](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/data_preparation/synthetic_preprocessing.html).

## Tutorials and paper figures

Choose a dataset in the [analysis tutorials](docs/tutorials/dataset_workflows/index.md)
and download its [data and trained model](docs/data_checkpoints.md). The
notebooks show how to calculate velocities and growth, simulate cell
populations, and examine cell–cell interactions. Open them under
`docs/tutorials/` and run the cells in order.

To reproduce a figure from the manuscript or Supplementary Information,
find it in the [paper figure guide](docs/tutorials/paper_figures/index.md).
Each guide takes you through the calculations and plotting for that figure.
The [data page](docs/data_checkpoints.md) includes the AGIST simulations,
LARRY and scNT model comparisons, and the five-dataset benchmark.

Use the [API reference](https://cytobridge-spatial.readthedocs.io/en/latest/api/index.html)
to look up individual functions.

## Command-line use

The workflow command runs the steps selected by a dataset configuration.
For example, with the prepared chicken-heart counts:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad outputs/chicken_heart_input/input.h5ad \
  --output-dir outputs/chicken_heart_trained --device cuda
```

See [input preparation](docs/chicken_heart_preparation.md) for the two calls that
produce that input, or [configure a workflow](docs/tutorials/your_data.ipynb) for
another experiment. The Python tutorials are the starting point for changing
individual analyses.

## Data and checkpoints

Download the aligned AnnData file and trained model for the dataset you want
to analyze. The [download guide](https://cytobridge-spatial.readthedocs.io/en/latest/data_checkpoints.html)
also lists the original studies and the inputs for training on raw counts.

## Repository layout

```text
cytobridge-spatial/
├── CytoBridge/          # package source
├── docs/                # Read the Docs source and published notebooks
├── scripts/             # command-line workflow helpers
├── tests/               # package, notebook, and release tests
├── pyproject.toml
└── .readthedocs.yaml
```

## Documentation

Read the Docs builds the documentation from this repository. To build the same
pages locally:

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

To run a notebook locally, follow the data and hardware requirements given
at its start.

## Development

Install the development dependencies and run the test suite:

```bash
python -m pip install -e '.[all,docs]'
python -m pytest -q
```

Changes to result readers should include portable tests and a notebook that
runs from the installed package. Plotting modules should keep Matplotlib
imports local so that `import CytoBridge` remains lightweight.

## Citation

If you use CytoBridge, cite the associated manuscript and software release.
The machine-readable citation is in [`CITATION.cff`](CITATION.cff).

## License

CytoBridge is distributed under the GNU General Public License v3.0 or later.
See [`LICENSE`](LICENSE).
