from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_frozen_checkpoint_ablation_reader.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("frozen_ablation_reader", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_frozen_ablation_reader_smoke(tmp_path) -> None:
    rng = np.random.default_rng(4)
    n = 60
    start = rng.normal(size=(n, 6)).astype(np.float32)
    full = start + np.asarray([0.2, 0.1, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    interaction_off = full + rng.normal(scale=0.02, size=full.shape).astype(np.float32)
    lr_gate_off = full + rng.normal(scale=0.05, size=full.shape).astype(np.float32)
    ablation = tmp_path / "ablation"
    ablation.mkdir()
    for name, endpoint in {
        "full": full,
        "interaction_off": interaction_off,
        "lr_gate_off": lr_gate_off,
    }.items():
        np.savez_compressed(
            ablation / f"{name}.npz",
            points=np.stack((start, endpoint)),
        )
    manifest = {
        "matched_controls": {"end_time": 4.0},
        "input": {
            "resolved_time_key": "time_numeric",
            "spatial_key": "spatial_aligned",
            "annotation_key": "Annotation",
        },
        "checkpoint": {
            "weight_sha256": "weight-hash",
            "score_sha256": "score-hash",
        },
    }
    (ablation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "cohort_row": np.arange(n),
            "cell_type": np.where(np.arange(n) % 2, "Type B", "Type A"),
        }
    ).to_csv(ablation / "initial_cohort.csv", index=False)

    target = rng.normal(loc=(0.15, 0.05), scale=1.0, size=(90, 2))
    adata = ad.AnnData(
        X=rng.normal(size=(90, 4)),
        obs=pd.DataFrame(
            {
                "time_numeric": np.full(90, 4.0),
                "Annotation": np.where(np.arange(90) % 3, "Type A", "Type B"),
            },
            index=[f"target-{i}" for i in range(90)],
        ),
    )
    adata.obsm["spatial_aligned"] = target
    h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(h5ad)
    output = tmp_path / "reader"

    module = _load_module()
    assert (
        module.main(
            [
                "--ablation-dir",
                str(ablation),
                "--adata",
                str(h5ad),
                "--output-dir",
                str(output),
                "--density-bins",
                "30",
                "--arrow-cells",
                "30",
            ]
        )
        == 0
    )
    for stem in (
        "01_endpoint_tissue_density",
        "02_paired_spatial_displacement",
        "03_cell_type_territories",
    ):
        assert (output / f"{stem}.png").is_file()
        assert (output / f"{stem}.pdf").is_file()
    summary = json.loads((output / "reader_summary.json").read_text())
    assert summary["semantic_guardrails"]["edge_density_matched"] is False
    assert summary["n_fixed_t3_cells"] == n
    assert (output / "START_HERE_CN.md").is_file()
    assert (output / "cell_type_paired_effects.csv").is_file()
