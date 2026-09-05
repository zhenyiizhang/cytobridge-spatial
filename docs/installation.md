# Installation

Use Python 3.10 or 3.11. The installation below includes the libraries needed
for the spatial analysis and training tutorials, together with JupyterLab.

## Install from GitHub

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial
python -m pip install -e '.[spatial,velocity,notebook]'
```

Open JupyterLab from your project folder to run a notebook:

```bash
jupyter lab
```

## Installation options

| Extra | Use it for |
| --- | --- |
| base | AnnData, arrays, tables, configuration, metrics |
| `preprocess` | raw-count preprocessing and spatial alignment |
| `train` | dynamical-model fitting |
| `graph` | PyTorch Geometric interaction models |
| `spatial` | preprocessing, training, and interaction-graph stages |
| `plot` | static and interactive figures |
| `velocity` | velocity analysis plus the scVelo and Plotly dependencies used by the default spatial downstream workflow |
| `notebook` | JupyterLab dataset tutorials; combine with `velocity` to run their downstream cells |
| `docs` | local ReadTheDocs build tools |
| `all` | the complete supported stack |

The default `cytobridge workflow ... --train` command continues through
downstream analysis, so install `spatial` and `velocity` together. The
`spatial` extra alone is sufficient when explicitly running only preprocessing,
training, and interaction-graph APIs.

The dataset tutorials use an NVIDIA GPU. Install the PyTorch CUDA build that
matches your machine. The small preprocessing example also runs on a CPU.

## Verify the environment

```bash
cytobridge --version
cytobridge doctor
cytobridge workflow --list-configs
```

`cytobridge doctor` checks which optional libraries are installed without
importing them or modifying data.

## Build the documentation locally

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

The documentation build mocks optional scientific imports only while reading
API signatures. Running an analysis still requires the corresponding
installation option.
