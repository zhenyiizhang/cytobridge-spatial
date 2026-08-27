# Tutorials

The dataset tutorials use the installed package and its bundled workflow
presets. Each dataset page lists the external inputs, the public package calls,
and the files or Python objects produced by the notebook.

```{toctree}
:maxdepth: 1
:caption: Data preparation

../data_checkpoints
data_preparation/synthetic_preprocessing
```

```{toctree}
:maxdepth: 1
:caption: Dataset workflows

zebrafish
mosta
arista
admouse
chicken_heart
```

```{toctree}
:maxdepth: 1
:caption: Dataset notebooks

dataset_workflows/zebrafish
dataset_workflows/mosta
dataset_workflows/arista
dataset_workflows/admouse
dataset_workflows/chicken_heart
```

```{toctree}
:maxdepth: 1
:caption: Paper figures
:glob:

paper_figures/*
zebrafish_videos
```

```{toctree}
:maxdepth: 1
:caption: Benchmarks

../benchmarks
../training_compute
```

The documentation build does not execute notebooks. Notebooks are stored
without outputs and require the packaged result tables or repository release
artifacts named on their page.
