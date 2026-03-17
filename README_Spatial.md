# CytoBridge Spatial Pipeline (API-First)

This document describes the streamlined spatial workflow after API cleanup.

## 1) Recommended API boundaries

- Package API (`CytoBridge.pp` / `CytoBridge.tl`):
  - `cb.pp.preprocess(...)`
  - `cb.pp.align_spatial(...)`
  - `cb.tl.fit(...)`
- Project workflow scripts (`scripts/`):
  - `scripts/preprocess_pipeline.py` for alignment + graph generation + edge predictor
  - `scripts/run_spatial_training.py` for training runs

## 2) Package-level usage (AnnData-first)

```python
import CytoBridge as cb
from CytoBridge.pp import AlignConfig

cfg = AlignConfig(
    n_pcs=50,
    spatial_dim=2,
    phase1_epochs=1000,
    phase2_epochs=50,
)

adata_aligned = cb.pp.align_spatial(
    adata_or_h5ad="/path/to/data.h5ad",
    time_key="time",
    cfg=cfg,
    output_csv="./results/aligned.csv",      # optional
    output_h5ad="./results/aligned.h5ad",    # optional
    device="cuda",
)

trained = cb.tl.fit(
    adata_aligned,
    config="./CytoBridge/configs/zebrafish_training.yaml",
    device="cuda",
)
```

## 3) Project-level preprocessing script

Use this when you also need interaction graphs and edge predictor training:

```bash
python scripts/preprocess_pipeline.py \
  --data-name zebrafish \
  --h5ad-path /path/to/data.h5ad \
  --time-key time \
  --output-dir ./results/zebrafish_preprocess \
  --device cuda
```

This script produces:

- `*_aligned.csv`
- `*_aligned.h5ad`
- `input_graph/`
- `metadata/`
- `edge_classifier/*_edge_model.pt`

## 4) Notes

- `cb.tl.fit(...)` is the canonical training entrypoint.
- `cb.tl.fit_spatial_csv(...)` and `cb.tl.fit_spatial_h5ad(...)` are deprecated wrappers for backward compatibility.
- `time_point_processed` is preserved/created for training consistency.
