# CytoBridge

CytoBridge learns continuous cell-state and spatial dynamics from transcriptomic
snapshots collected at discrete time points. It provides preprocessing,
training, trajectory simulation, growth analysis, and time-resolved
cell-cell interaction analysis in one Python package.

```{image} _static/cytobridge_figure1.png
:alt: Overview of the CytoBridge model and downstream analyses
:class: cytobridge-home-figure
```

::::{grid} 1 2 3 3
:gutter: 2
:class-container: cytobridge-home-cards

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install CytoBridge and choose the optional dependencies needed for plotting or
spatial analysis.
:::

:::{grid-item-card} Run CytoBridge on your data
:link: tutorials/your_data
:link-type: doc

Start with a counts H5AD. The tutorial prepares the data, trains a model, and
runs the standard analyses.
:::

:::{grid-item-card} Dataset notebooks
:link: tutorials/dataset_workflows/index
:link-type: doc

Repeat the standard workflow for each of the five datasets used in the paper.
:::

:::{grid-item-card} Paper figures
:link: tutorials/paper_figures/index
:link-type: doc

Redraw figures from included numerical files, or view completed pages when the
original layout files are kept separately.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Look up the Python functions and command-line options.
:::

:::{grid-item-card} GitHub
:link: https://github.com/zhenyiizhang/cytobridge-spatial/
:link-type: url

Browse the source code, report a problem, or contribute a change.
:::

::::

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
```
