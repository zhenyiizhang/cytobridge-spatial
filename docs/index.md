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

:::{grid-item-card} Get started
:link: quickstart
:link-type: doc

See the input, run a first analysis, and find the resulting plots.
:::

:::{grid-item-card} Run CytoBridge on your data
:link: training
:link-type: doc

Prepare expression and coordinates, build an LR graph, and fit cell-state dynamics.
:::

:::{grid-item-card} Analysis tutorials
:link: tutorials/dataset_workflows/index
:link-type: doc

Calculate velocities, growth, and interactions with the Python API.
:::

:::{grid-item-card} Paper figures
:link: tutorials/paper_figures/index
:link-type: doc

Find a main or supplementary figure and the code used to calculate and draw it.
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

tutorials/dataset_workflows/index
training
trajectory_analysis

```

```{toctree}
:maxdepth: 1
:caption: Paper reproduction
:hidden:

tutorials/paper_figures/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Python and command line
:hidden:

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

contributing
```
