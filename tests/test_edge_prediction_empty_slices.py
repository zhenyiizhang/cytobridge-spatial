from __future__ import annotations

import gzip
import pickle

import numpy as np
import pandas as pd

from CytoBridge.pp.edge_prediction import train_edge_predictor


def _write_adjacency(path, edges) -> None:
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump([edges], handle)


def test_train_edge_predictor_accepts_empty_time_slice(tmp_path) -> None:
    data_name = "example"
    graph_dir = tmp_path / "graphs"
    _write_adjacency(
        graph_dir / f"{data_name}_t0" / f"{data_name}_t0_adjacency_records",
        [],
    )
    _write_adjacency(
        graph_dir / f"{data_name}_t1" / f"{data_name}_t1_adjacency_records",
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
            [5, 0],
            [0, 5],
        ],
    )

    rows = []
    for time in (0, 1):
        for node in range(6):
            rows.append(
                {
                    "samples": time,
                    "spatial_x": float(node),
                    "spatial_y": float(time),
                    "latent_0": float(node % 2),
                }
            )
    feature_csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(feature_csv, index=False)

    result = train_edge_predictor(
        data_name=data_name,
        feature_csv_path=str(feature_csv),
        graph_input_dir=str(graph_dir),
        output_model_path=str(tmp_path / "edge_predictor.pt"),
        epochs=1,
        batch_size=8,
        distance_threshold=10.0,
        device="cpu",
        num_workers=0,
        random_seed=42,
    )

    assert (tmp_path / "edge_predictor.pt").is_file()
    assert (tmp_path / "edge_predictor.pt.meta.json").is_file()
    assert result["model_path"] == str(tmp_path / "edge_predictor.pt")
