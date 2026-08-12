from __future__ import annotations

import gzip
import pickle

import pandas as pd
import pytest

from CytoBridge.pp.edge_prediction import train_edge_predictor


def _write_adjacency(path, edges) -> None:
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump([edges], handle)


def _write_feature_csv(path) -> None:
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
    pd.DataFrame(rows).to_csv(path, index=False)


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

    feature_csv = tmp_path / "features.csv"
    _write_feature_csv(feature_csv)

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
    assert result["positive_edge_deduplication"] == {
        "unit": "directed_cell_pair_per_time_slice",
        "raw": 13,
        "unique": 12,
        "duplicates_removed": 1,
        "by_time": [
            {
                "time_value": 0.0,
                "raw": 0,
                "unique": 0,
                "duplicates_removed": 0,
            },
            {
                "time_value": 1.0,
                "raw": 13,
                "unique": 12,
                "duplicates_removed": 1,
            },
        ],
    }


@pytest.mark.parametrize(
    ("second_slice_edges", "detail"),
    (
        (None, "missing"),
        ([[0, 6]], "out of bounds"),
    ),
)
def test_train_edge_predictor_rejects_missing_or_mismatched_observed_slice(
    tmp_path,
    second_slice_edges,
    detail,
) -> None:
    data_name = "example"
    graph_dir = tmp_path / "graphs"
    _write_adjacency(
        graph_dir / f"{data_name}_t0" / f"{data_name}_t0_adjacency_records",
        [[0, 1], [1, 0]],
    )
    if second_slice_edges is not None:
        _write_adjacency(
            graph_dir / f"{data_name}_t1" / f"{data_name}_t1_adjacency_records",
            second_slice_edges,
        )
    feature_csv = tmp_path / "features.csv"
    _write_feature_csv(feature_csv)

    with pytest.raises(ValueError) as error:
        train_edge_predictor(
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

    message = str(error.value)
    assert (
        "Every observed nonempty time slice needs a matching interaction graph"
        in message
    )
    assert "time_idx=1, time_value=1, n_nodes=6" in message
    assert detail in message
