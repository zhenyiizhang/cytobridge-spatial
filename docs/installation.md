# Installation

CytoBridge requires Python 3.10 or 3.11. The base install contains stable data,
configuration, and metric dependencies; heavier scientific stacks are selected
with extras.

## Release candidate from GitHub

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial
git switch release/cytobridge-reproducible-20260812
python -m pip install -e '.[all]'
```

Once version 1.5 is published, the normal installation will be:

```bash
python -m pip install 'CytoBridge[all]>=1.5,<1.6'
```

## Dependency profiles

| Extra | Use it for |
| --- | --- |
| base | AnnData, arrays, tables, configuration, metrics |
| `preprocess` | raw-count preprocessing and spatial alignment |
| `train` | dynamical-model fitting |
| `graph` | PyTorch Geometric interaction models |
| `spatial` | preprocessing + training + graph workflow |
| `plot` | static and interactive figures |
| `velocity` | velocity/terminal-state analyses |
| `notebook` | executable JupyterLab dataset tutorials, including model/graph/plot dependencies |
| `docs` | local ReadTheDocs build tools |
| `all` | the complete supported stack |

GPU-specific PyTorch wheels are an environment decision. Install the CPU or
CUDA build appropriate for your machine before or together with a
Torch-dependent extra.

## Verify the environment

```bash
cytobridge --version
cytobridge doctor
cytobridge workflow --list-configs
```

`cytobridge doctor` is read-only. It reports dependency availability without
importing the scientific stacks or modifying data.

## Build the documentation locally

```bash
python -m pip install -e '.[docs]'
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

The documentation build mocks optional scientific imports only while reading
API signatures. Running an analysis still requires its matching dependency
profile.
