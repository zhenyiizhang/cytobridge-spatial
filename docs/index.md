# CytoBridge

CytoBridge is a Python package for learning continuous cell-state and spatial
dynamics from snapshot data. The package unifies preprocessing, spatial
alignment, six-stage model fitting, interpolation, velocity, growth, sparse
cell–cell communication, strict ligand–receptor analysis, gene programs, and
manuscript-style visualization behind one public API. Matched cross-method
benchmarks and manuscript-specific perturbations use the same saved outputs but
remain explicit analysis pipelines rather than implicit workflow side effects.

```{important}
The four formal spatial applications use `alpha_express=0.015`,
`alpha_spatial=10`, and seed 42. Zebrafish, MOSTA, and ARISTA use the formal
spatial-domain label setting `k=10`; AD uses `k=1` because stronger spatial
voting suppresses heterogeneous and rare populations.
```

## Start here

::::{grid} 1 2 2 2
:::{grid-item-card} Install and inspect
:link: installation
:link-type: doc
Choose a dependency profile, run `cytobridge doctor`, and inspect a workflow
before starting a long fit.
:::
:::{grid-item-card} Run a dataset workflow
:link: quickstart
:link-type: doc
Use the wheel-bundled presets for Zebrafish, MOSTA, ARISTA, or AD.
:::
:::{grid-item-card} Follow a tutorial
:link: tutorials/index
:link-type: doc
Open the four checked-in notebooks and dataset-specific guides.
:::
:::{grid-item-card} Understand the science
:link: scientific_contract
:link-type: doc
Read the shared contracts, allowed dataset differences, and interpretation
limits.
:::
::::

```{toctree}
:maxdepth: 2
:caption: User guide

installation
data_checkpoints
quickstart
scientific_contract
historical_artifact_compatibility
downstream
benchmarks
training_compute
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Reference and development

api/index
limitations
contributing
release_notes
```

The repository also retains a historical Zebrafish clean-counts audit for
developers. It is excluded from the strict documentation build: the package
workflow, scientific contract, and four dataset guides are the supported public
entry points.
