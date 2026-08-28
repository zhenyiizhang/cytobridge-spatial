```{image} _static/cytobridge_logo.svg
:alt: CytoBridge
:class: cytobridge-home-logo
```

# CytoBridge

CytoBridge learns continuous cell-state and spatial dynamics from transcriptomic
snapshots collected at discrete time points. It provides preprocessing,
training, trajectory simulation, growth analysis, and time-resolved
cell-cell interaction analysis in one Python package.

```{image} _static/cytobridge_figure1.png
:alt: Overview of the CytoBridge model and downstream analyses
:class: cytobridge-home-figure
```

## Installation

Install CytoBridge and check the optional dependencies needed for your analysis.

[Installation guide](installation.md)

## Get started

Run a small example, then move to the notebook for your own AnnData object.

[Quickstart](quickstart.md) · [Run CytoBridge on your data](tutorials/your_data.ipynb)

## Tutorials

The tutorials contain complete preprocessing, training, downstream, and
paper-figure examples. Code cells are shown with their saved outputs.

[Browse tutorials](tutorials/index.md)

## API

Function and command-line reference for preprocessing, training, downstream
analysis, plotting, and paper figures.

[API reference](api/index.rst)

## Paper figures

Follow the calculations used for the main and supplementary figures, including
the files passed from training to downstream analysis and plotting.

[Paper figure notebooks](tutorials/paper_figures/index.md)

```{toctree}
:maxdepth: 1
:caption: Get started
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Guides
:hidden:

data_checkpoints
downstream
nonspatial_workflows
benchmarks
training_compute
paper_reproduction
limitations
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden:

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

contributing
release_notes
```
