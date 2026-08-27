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

For another dataset, export the closest preset, edit the data columns and
analysis settings, and dry-run the edited JSON:

```bash
cytobridge workflow --config zebrafish --export-config configs/my_dataset.json
cytobridge workflow --config configs/my_dataset.json --train \
  --input-h5ad inputs/my_dataset.h5ad \
  --output-dir outputs/my_dataset --device cuda --dry-run
```

The [own-data tutorial](https://cytobridge-spatial.readthedocs.io/en/latest/tutorials/your_data.html)
lists the fields that must be changed before training.

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

The tutorial index covers data preparation, five dataset tutorials, paper
figures, and benchmarks. Each dataset notebook follows the same path from setup
through preprocessing, training, downstream analysis, figure code, and saved
files. Paper-figure notebooks state whether they recalculate values, redraw
prepared summaries, assemble panel files, or export a reference page.

Installed figure commands use the same public result APIs as the notebooks:

```bash
cytobridge figure list
cytobridge figure explain zebrafish-si
cytobridge figure arista-lr --output-dir outputs/arista_lr
```

The explanation reports the input level, upstream entry, plotting command, and
limit of the selected figure workflow. Numerical-redraw notebooks preview the
PNG created by their own plotting cell. Assembly and reference-export pages
identify the PDF or image used as a source.

Complete dataset runs use external AnnData files and checkpoints; their names,
formats, and expected locations are documented in the input guide.

The [paper reproduction index](docs/paper_reproduction.md) maps Main Figures
1–6, Supplementary Figures S1–S43, Supplementary Tables 1–2, and the zebrafish
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

Read the Docs builds the documentation from this repository. To build the same
pages locally:

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

Read the Docs renders the committed notebook outputs. CI also runs every
published notebook from top to bottom.

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
