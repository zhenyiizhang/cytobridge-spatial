# Dataset tutorials

The checked-in notebooks use the installed package and wheel-bundled presets.
They are committed without outputs so readers can inspect every step and run
them against their own explicit inputs. By default, each notebook uses up to
5,000 generated particles and one midpoint per observed interval. Observed
slices, classifier fitting, and observed-slice velocity still use the supplied
AnnData; the cap is not a whole-workflow downsample. Set
`RUN_FORMAL_SCOPE=True` to select the preset's full time grid and population
policy; this does not turn the notebook into every manuscript-specific analysis.

```{toctree}
:maxdepth: 1

zebrafish
mosta
arista
admouse
```

Release checks parse every notebook and run a small synthetic API-wiring smoke
with training disabled. They verify that the tutorial calls remain
compatible with the installed package; they do not execute a trained checkpoint
or reproduce full-data results.

The notebook links on the dataset pages are direct downloads. The shell paths
shown there assume a Git source checkout; a wheel install provides the runtime
dependencies and package presets, not a new `notebooks/` directory in the
user's current working directory.

All four notebooks implement the same compact sequence: load the formal preset,
optionally fit or load a model, interpolate and classify, summarize composition,
velocity, growth and sparse communication, project strict ligand-receptor
trajectories, and evaluate unwarped distributions. Dataset-specific paper
panels, gene-program selections, perturbations, and cross-method benchmark runs
remain explicit analyses rather than hidden notebook side effects.
