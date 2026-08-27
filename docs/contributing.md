# Contributing

Changes are developed on a focused branch and submitted through a pull
request. Keep unrelated changes in separate branches so tests and discussion
stay scoped to one topic.

## Development environment

Install the package and documentation dependencies in an isolated Python 3.10
or 3.11 environment:

```bash
python -m pip install -e '.[all,docs]'
```

## Add a dataset workflow

1. Reuse the public preprocessing, fitting, downstream, evaluation, and
   plotting APIs.
2. Add a small preset under `CytoBridge/workflow_configs/` for input keys,
   time points, cutoffs, particle settings, and enabled analyses.
3. Add a training YAML only when the shared model needs dataset-specific
   values.
4. Add a notebook and a page under `docs/tutorials/` that list inputs, package
   calls, and outputs.
5. Add focused tests for the preset and for any new adapter behavior.

Avoid copying the full training or downstream pipeline into a dataset script.
Reusable behavior belongs in the package API.

## Documentation

- Use paths such as `inputs/` and `outputs/` in examples.
- Document required fields and array shapes next to the function that uses
  them.
- Keep generated notebook outputs out of source unless a page needs them for
  display.
- Add new public functions to the relevant page under `docs/api/`.
- Build with warnings treated as errors before submitting documentation
  changes.

## Run checks

```bash
pytest -q
python scripts/smoke_installed_wheel.py
sphinx-build -W --keep-going -E -b html docs docs/_build/html
```

Use smaller targeted test selections while developing, then run the complete
suite before opening the pull request.
