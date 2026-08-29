# Installation

CytoBridge requires Python 3.10 or 3.11. The base install contains stable data,
configuration, and metric dependencies; heavier scientific stacks are selected
with extras.

## Install from GitHub

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial
git switch release/cytobridge-reproducible-20260812
python -m pip install -e '.[spatial,velocity]'
```

Once version 1.5 is published, the normal installation will be:

```bash
python -m pip install 'CytoBridge[spatial,velocity]>=1.5,<1.6'
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

GPU-specific PyTorch wheels are an environment decision. Install the CPU or
CUDA build appropriate for your machine before or together with a
Torch-dependent extra.

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
