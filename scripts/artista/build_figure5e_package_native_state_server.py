#!/usr/bin/env python3
"""Build Figure 5e growth/interaction state from one package-native ARISTA run.

The numerical contract is intentionally small and public-API only:

* every requested slice is read from the fresh package downstream;
* the current-format model is loaded from the same training directory;
* growth is evaluated with ``model.predict_growth``;
* interaction magnitude is evaluated with
  ``CytoBridge.tl.compute_velocity_components``;
* no display registration, old result, or historical numeric table enters the
  calculation.

The resulting grouped CSV is consumed by the locked legacy Figure 5e renderer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--times", default=",".join(str(value) for value in DEFAULT_TIMES))
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def _parse_times(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--times resolved to an empty list")
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError("--times must be unique and ascending")
    return result


def _atomic_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = _parser().parse_args()
    slice_dir = args.slice_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = _atomic_output_dir(args.output_dir)
    times = _parse_times(args.times)

    import anndata as ad
    import torch
    import CytoBridge as cb

    first = ad.read_h5ad(slice_dir / f"time_{_safe_time_name(times[0])}.h5ad")
    if first.X.ndim != 2 or first.X.shape[1] < 3:
        raise ValueError(f"Unexpected model-state shape: {first.X.shape}")
    dim = int(first.X.shape[1])
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=dim,
        device=args.device,
    )
    model = loaded.model
    interaction_cutoff = float(
        getattr(getattr(model, "interaction_net", None), "cutoff", 1000.0)
    )

    rows: list[pd.DataFrame] = []
    slice_inputs: dict[str, dict[str, object]] = {}
    for time_value in times:
        slice_path = slice_dir / f"time_{_safe_time_name(time_value)}.h5ad"
        if not slice_path.exists():
            raise FileNotFoundError(slice_path)
        adata = first if np.isclose(time_value, times[0]) else ad.read_h5ad(slice_path)
        if args.annotation_key not in adata.obs.columns:
            raise KeyError(f"{slice_path} lacks obs[{args.annotation_key!r}]")
        values = np.asarray(adata.X, dtype=np.float32)
        if values.shape[1] != dim:
            raise ValueError(
                f"Inconsistent state dimension at t={time_value}: {values.shape[1]} != {dim}"
            )
        components = cb.tl.compute_velocity_components(
            data=values,
            time_value=float(time_value),
            model=model,
            interaction_m=int(args.interaction_m),
            interaction_threshold=interaction_cutoff,
            device=args.device,
            spatial_dim=2,
        )
        with torch.no_grad():
            value_tensor = torch.as_tensor(values, dtype=torch.float32, device=args.device)
            time_tensor = torch.full(
                (values.shape[0], 1),
                float(time_value),
                dtype=torch.float32,
                device=args.device,
            )
            growth = (
                model.predict_growth(t=time_tensor, x=value_tensor)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        interaction = np.linalg.norm(
            np.asarray(components["interaction"], dtype=np.float32), axis=1
        )
        if not np.all(np.isfinite(growth)) or not np.all(np.isfinite(interaction)):
            raise ValueError(f"Non-finite Figure 5e values at t={time_value}")
        labels = adata.obs[args.annotation_key].astype(str).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "time": float(time_value),
                    "celltype": labels,
                    "growth": growth,
                    "interaction": interaction,
                }
            )
        )
        slice_inputs[str(float(time_value))] = {
            **_record(slice_path),
            "n_cells": int(values.shape[0]),
            "state_dim": dim,
        }

    raw = pd.concat(rows, ignore_index=True)
    grouped = (
        raw.groupby(["time", "celltype"], observed=True, sort=True)
        .agg(
            growth_mean=("growth", "mean"),
            interaction_mean=("interaction", "mean"),
            n=("growth", "size"),
        )
        .reset_index()
        .sort_values(["time", "celltype"], kind="mergesort")
        .reset_index(drop=True)
    )
    if int(grouped["n"].sum()) != len(raw):
        raise AssertionError("Grouped Figure 5e counts do not recover all rows")
    if set(grouped["time"].astype(float)) != set(times):
        raise AssertionError("Grouped Figure 5e table lost a time point")

    raw_path = output_dir / "figure5e_growth_interaction_by_cell.csv"
    grouped_path = output_dir / "figure5e_growth_interaction_by_celltype.csv"
    manifest_path = output_dir / "manifest.json"
    raw.to_csv(raw_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    manifest = {
        "schema": "arista.figure5e.package_native_state.v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "Growth and interaction magnitudes are recalculated from the fresh "
            "package-native ARISTA model and its own nine downstream states."
        ),
        "calculation": {
            "growth": "model.predict_growth",
            "interaction": "L2 norm of CytoBridge.tl.compute_velocity_components()['interaction']",
            "interaction_m": int(args.interaction_m),
            "interaction_cutoff": interaction_cutoff,
            "device": args.device,
            "display_inputs_used": False,
            "historical_numeric_inputs_used": False,
        },
        "model": {
            "directory": str(model_dir),
            "weight_stage": loaded.weight_stage,
            "score_stage": loaded.score_stage,
            "weight": _record(Path(loaded.weight_path)),
            "score": None if loaded.score_path is None else _record(Path(loaded.score_path)),
            "state_dim": dim,
        },
        "slice_inputs": slice_inputs,
        "qa": {
            "times": times,
            "n_raw_rows": int(len(raw)),
            "n_grouped_rows": int(len(grouped)),
            "n_celltypes": int(grouped["celltype"].nunique()),
            "counts_by_time": {
                str(float(time)): int(count)
                for time, count in raw.groupby("time", sort=True).size().items()
            },
            "growth_range": [float(raw["growth"].min()), float(raw["growth"].max())],
            "interaction_range": [
                float(raw["interaction"].min()),
                float(raw["interaction"].max()),
            ],
            "finite": True,
        },
        "outputs": {
            "raw": _record(raw_path),
            "grouped": _record(grouped_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["qa"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
