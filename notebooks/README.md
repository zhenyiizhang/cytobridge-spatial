# CytoBridge dataset tutorials

These notebooks are the package-facing entry points for the four spatial
applications:

- `01_zebrafish.ipynb`
- `02_mosta.ipynb`
- `03_arista.ipynb`
- `04_admouse.ipynb`

They use the same installed `CytoBridge` APIs and wheel-bundled workflow presets.
Zebrafish, MOSTA, and ARISTA use the manuscript spatial-domain setting `k=10`;
AD uses `k=1`. Quantitative analyses use aligned, pre-warp states. Spatial warp
is reserved for display mosaics, videos, and 3D slice views.

Each notebook defaults to a compact walkthrough: it uses midpoint interpolation
and caps the simulated population so readers can inspect the API without launching
a production-scale job. Set `RUN_FORMAL_SCOPE = True` to select the complete time
grid and population policy from the wheel-packaged workflow preset. The compact
and formal modes share the preset's classifier, SDE, noise, and growth settings;
only workload scope changes.

The notebooks are intentionally committed without outputs. The release smoke
runner checks real package API wiring on small synthetic fixtures. It does not load
the manuscript checkpoints and is not evidence that the full formal analyses ran.
Formal scientific results come from the corresponding full-data runs.

Start with:

```bash
python -m pip install -e '.[all]'
jupyter lab notebooks/01_zebrafish.ipynb
```

Each notebook starts with `RUN_TRAINING = False`. Supply an existing aligned
H5AD and model directory for analysis, or deliberately enable training and
provide the required edge predictor. Current checkpoints embed predictor
weights and remain portable; older current-format checkpoints without those
weights also need an explicit edge-predictor path for downstream loading.

To run the notebook API-wiring smoke:

```bash
python scripts/smoke_dataset_notebooks.py
```
