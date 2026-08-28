# Release notes

## 1.5.0rc1

- Added spatial dataset configurations for Zebrafish, MOSTA, ARISTA, AD
  mouse, and developing chicken heart.
- Added the `cytobridge workflow` command for planning, preprocessing,
  training, and downstream analysis.
- Added chicken-heart raw-count and OT-input preparation tools.
- Added non-spatial Weinreb and scNT workflows with preparation, prior building,
  training, evaluation, and attribution commands.
- Updated the six-stage training path and predictor-gated interaction loading.
- Updated interval simulation, velocity, growth, sparse communication,
  ligand--receptor projection, gene dynamics, and distribution metrics.
- Added `CytoBridge.results` readers, table writers, and plotting functions for
  the example result tables.
- Added leave-one-timepoint-out protocol readers and full-model compute-cost
  readers.
- Added dataset tutorials and paper-figure notebooks to the documentation.
- Added a notebook for configuring and running CytoBridge on a new dataset.
- Added the complete calculation and plotting steps for every paper-figure notebook,
  with executed figure previews in the published documentation.
- Improved bounded-memory interaction inference and optional-dependency
  handling for package and documentation imports.

This is a pre-release version intended for testing before the stable 1.5.0
release.
