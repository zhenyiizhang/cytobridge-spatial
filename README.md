# CytoBridge

CytoBridge is a Python package for modelling spatiotemporal single-cell and
spatial transcriptomics data. It provides preprocessing, spatial alignment,
interaction-graph construction, dynamical-model training, interpolation, and
downstream analysis through one package API.

[Documentation](https://cytobridge-spatial.readthedocs.io/) ·
[Tutorials](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/index.html) ·
[API reference](https://cytobridge-spatial.readthedocs.io/en/latest/api/index.html) ·
[Issues](https://github.com/zhenyiizhang/cytobridge-spatial/issues)

The repository contains the installable package, dataset tutorials, paper
figure notebooks, tests, and the source used to build Read the Docs. Large input
datasets and model checkpoints are distributed separately; their expected
formats are listed in the
[data and checkpoint guide](https://cytobridge-spatial.readthedocs.io/en/latest/data_checkpoints.html).

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

The notebooks in `docs/tutorials/dataset_workflows/` introduce the analysis
functions using chicken heart, MOSTA, ARISTA, AD mouse, and Zebrafish data.
Their displayed outputs come from executing the code on each dataset's trained
model. Training is not repeated when you run these notebooks.
Choose a dataset in the [tutorial index](docs/tutorials/dataset_workflows/index.md)
and download its external AnnData files and checkpoints using the
[data and checkpoint guide](docs/data_checkpoints.md).

The [paper figure notebooks](docs/tutorials/paper_figures/index.md) are a separate
collection for reproducing the manuscript's particular comparisons and panel
layouts. They identify their numerical inputs and plotting code. Where a page
uses existing vector panels or a completed figure, it says so.

The [API reference](https://cytobridge-spatial.readthedocs.io/en/latest/api/index.html)
groups functions by analysis. The
[figure source index](docs/paper_reproduction.md) records the scripts and data
used for the paper panels.

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

Raw datasets, processed AnnData files, and trained checkpoints are not stored
in the wheel. The documentation records:

- dataset accessions and required AnnData fields;
- preprocessing and coordinate conventions;
- checkpoint directory layout and package compatibility;
- processed tables included for figure reproduction.

See the
[data and checkpoint guide](https://cytobridge-spatial.readthedocs.io/en/latest/data_checkpoints.html)
before running a dataset notebook.

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

Generated figures, large result directories, H5AD files, and checkpoints are
not part of the source distribution.

## Documentation

Read the Docs builds the documentation from this repository. To build the same
pages locally:

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

Read the Docs renders the committed notebook outputs. CI executes the small
examples and checks notebook structure and outputs. Study-data notebooks are
executed separately on a machine with their data and models.

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
