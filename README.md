# CytoBridge

CytoBridge is a Python package for modelling spatiotemporal single-cell and
spatial transcriptomics data. It provides preprocessing, spatial alignment,
interaction-graph construction, dynamical-model training, interpolation, and
downstream analysis through one package API.

[Documentation](https://cytobridge-spatial.readthedocs.io/) ·
[Tutorials](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/index.html) ·
[API reference](https://cytobridge-spatial.readthedocs.io/en/latest/api/index.html) ·
[Issues](https://github.com/zhenyiizhang/cytobridge-spatial/issues)

The repository contains the installable package, dataset workflows, paper
figure notebooks, tests, and the source used to build ReadTheDocs. Large input
datasets and model checkpoints are distributed separately; their expected
formats are listed in the
[data and checkpoint guide](https://cytobridge-spatial.readthedocs.io/en/latest/data_checkpoints.html).

## Installation

CytoBridge supports Python 3.10 and 3.11.

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial
python -m pip install -e '.[all]'
```

Smaller dependency profiles are available when the full scientific stack is
not needed:

```bash
python -m pip install -e '.[preprocess]'
python -m pip install -e '.[train]'
python -m pip install -e '.[plot]'
python -m pip install -e '.[notebook]'
```

Install a PyTorch build that matches the CUDA version available on the target
machine.

Check the installation without starting a workflow:

```bash
cytobridge --version
cytobridge doctor
```

## Quick start

Five spatial dataset presets are included: Zebrafish, MOSTA, ARISTA, AD mouse,
and developing chicken heart. Inspect a preset before supplying data:

```bash
cytobridge workflow --list-configs
cytobridge workflow --config zebrafish --dry-run
```

Run downstream analysis from an aligned AnnData file and a trained model:

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad /path/to/zebrafish_aligned.h5ad \
  --model-dir /path/to/zebrafish_model \
  --output-dir /path/to/zebrafish_results \
  --device cuda
```

Start preprocessing and model training explicitly with `--train`:

```bash
cytobridge workflow --config mosta --train \
  --input-h5ad /path/to/mosta_raw_counts.h5ad \
  --output-dir /path/to/mosta_run \
  --device cuda
```

The workflow command uses the same package functions exposed under
`CytoBridge.pp`, `CytoBridge.tl`, and `CytoBridge.pl`. See the
[quickstart](https://cytobridge-spatial.readthedocs.io/en/latest/quickstart.html) for
the input keys and optional ligand–receptor and gene-dynamics steps.

## Tutorials and paper figures

The documentation separates runnable material into four groups:

- data preparation;
- dataset workflows;
- paper figure reproduction;
- benchmarks.

Dataset workflows show the package calls needed for a complete analysis and
expect external AnnData files and checkpoints. Paper figure notebooks use
small processed tables included with the package, recalculate panel data, and
write the corresponding PDF, PNG, and CSV files. They do not retrain a model.

The [paper reproduction index](docs/paper_reproduction.md) maps Main Figures
1–6, Supplementary Figures S1–S39, Supplementary Tables 1–2, and the zebrafish
videos to their notebooks, scripts, processed inputs, and external-data
requirements. The source checkout also contains the complete MOSTA and ARISTA
figure releases under `release_artifacts/`; these larger releases are not
installed with the wheel.

Start with the
[tutorial index](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/index.html)
or browse the notebooks in `docs/tutorials/`.

## Data and checkpoints

Raw datasets, processed AnnData files, and trained checkpoints are not stored
in the wheel. The documentation records:

- dataset accessions and required AnnData fields;
- preprocessing and coordinate conventions;
- checkpoint directory layout and package compatibility;
- compact processed tables included for figure reproduction.

See the
[data and checkpoint guide](https://cytobridge-spatial.readthedocs.io/en/latest/data_checkpoints.html)
before running a dataset notebook.

## Repository layout

```text
cytobridge-spatial/
├── CytoBridge/          # package source
├── docs/                # ReadTheDocs source and canonical notebooks
├── scripts/             # command-line workflow helpers
├── tests/               # package, notebook, and release tests
├── pyproject.toml
└── .readthedocs.yaml
```

Generated figures, large result directories, H5AD files, and checkpoints are
not part of the source distribution.

## Documentation

ReadTheDocs builds the documentation from this repository. To build the same
pages locally:

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

ReadTheDocs renders the committed notebook outputs; CI executes the compact
notebooks separately.

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
