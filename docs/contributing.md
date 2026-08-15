# Contributing and extending CytoBridge

`main` is the accepted release baseline. New work starts on a focused branch,
passes package and documentation tests, and enters `main` through a pull
request.

## Add a dataset

1. Reuse the public preprocessing, fitting, downstream, evaluation, and plotting APIs.
2. Add a small `CytoBridge/workflow_configs/<dataset>.json` preset for keys,
   time grid, cutoffs, particle scope, and supported analyses.
3. Add a training YAML only when the shared model needs dataset-scale values.
4. Add a clean notebook under `notebooks/` and a guide under `docs/tutorials/`.
5. Add focused tests for the preset and any genuinely new adapter behavior.

Do not copy the full training or downstream pipeline into a dataset script.
General scientific fixes belong in the public package so every dataset uses the
same corrected implementation.

## Release checks

```bash
pytest -q
python scripts/smoke_installed_wheel.py
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The five notebook smoke executions use explicit synthetic data and are kept
separate from formal scientific outputs.
