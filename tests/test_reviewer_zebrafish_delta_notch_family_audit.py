from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from reviewer_zebrafish_ccc import delta_notch_family_audit as AUDIT  # noqa: E402


def _adata(
    obs_names: list[str],
    stages: list[float],
    labels: list[str],
    annotations: list[str],
) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "time_point_processed": stages,
            "time": labels,
            "Annotation": annotations,
        },
        index=obs_names,
    )
    return ad.AnnData(
        X=np.zeros((len(obs_names), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["dummy"]),
    )


def _edge(**updates: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "stage": 4.0,
        "stage_label": "24hpf",
        "grouping_seed": 101,
        "source_index": 1,
        "target_index": 2,
        "source_index_stage": 0,
        "target_index_stage": 1,
        "sender_type": "A",
        "receiver_type": "B",
        "attention_abs_mean": 2.0,
        "edge_message_norm_joint": 3.0,
    }
    row.update(updates)
    return pd.DataFrame([row])


def _observed_from(data: ad.AnnData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_index": np.arange(data.n_obs, dtype=int),
            "obs_name": data.obs_names.astype(str),
            "stage": data.obs["time_point_processed"].to_numpy(float),
            "stage_label": data.obs["time"].astype(str).to_numpy(),
            "cell_type": data.obs["Annotation"].astype(str).to_numpy(),
        }
    )


def test_nonempty_output_fails_closed_and_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    sentinel = output / "old.txt"
    sentinel.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        AUDIT.prepare_output(output, overwrite=False)
    assert sentinel.read_text(encoding="utf-8") == "old"

    AUDIT.prepare_output(output, overwrite=True)
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_explicit_stage_local_indices_are_preferred_and_crosschecked() -> None:
    data = _adata(
        ["early", "source", "target", "late"],
        [0.0, 4.0, 4.0, 0.0],
        ["5.25hpf", "24hpf", "24hpf", "5.25hpf"],
        ["X", "A", "B", "X"],
    )
    stage_mask = np.asarray([False, True, True, False])

    resolved, metadata = AUDIT.resolve_edge_stage_indices(
        _edge(),
        data,
        stage_mask,
        stage=4.0,
        stage_label="24hpf",
        time_key="time_point_processed",
        annotation_key="Annotation",
        observed_cells=_observed_from(data),
    )

    assert resolved["_source_stage_index"].tolist() == [0]
    assert resolved["_target_stage_index"].tolist() == [1]
    assert metadata["stage_local_columns_present"] is True
    assert metadata["global_index_order_assumed_without_validation"] is False
    assert metadata["mode"] == "explicit_stage_local_columns"


def test_stage_local_and_global_disagreement_is_rejected() -> None:
    data = _adata(
        ["early", "source", "target"],
        [0.0, 4.0, 4.0],
        ["5.25hpf", "24hpf", "24hpf"],
        ["X", "A", "B"],
    )
    stage_mask = np.asarray([False, True, True])

    with pytest.raises(ValueError, match="stage-local indices disagree"):
        AUDIT.resolve_edge_stage_indices(
            _edge(source_index=2),
            data,
            stage_mask,
            stage=4.0,
            stage_label="24hpf",
            time_key="time_point_processed",
            annotation_key="Annotation",
            observed_cells=_observed_from(data),
        )


def test_missing_stage_local_indices_map_by_observed_cell_names_after_reorder() -> None:
    # H5AD order intentionally differs from the attribution global-index order.
    data = _adata(
        ["cell_c", "cell_a", "cell_b"],
        [4.0, 4.0, 4.0],
        ["24hpf", "24hpf", "24hpf"],
        ["C", "A", "B"],
    )
    edges = _edge(source_index=0, target_index=1).drop(
        columns=["source_index_stage", "target_index_stage"]
    )
    observed = pd.DataFrame(
        {
            "global_index": [0, 1, 2],
            "obs_name": ["cell_a", "cell_b", "cell_c"],
            "stage": [4.0, 4.0, 4.0],
            "stage_label": ["24hpf", "24hpf", "24hpf"],
            "cell_type": ["A", "B", "C"],
        }
    )

    resolved, metadata = AUDIT.resolve_edge_stage_indices(
        edges,
        data,
        np.ones(3, dtype=bool),
        stage=4.0,
        stage_label="24hpf",
        time_key="time_point_processed",
        annotation_key="Annotation",
        observed_cells=observed,
    )

    assert resolved["_source_stage_index"].tolist() == [1]
    assert resolved["_target_stage_index"].tolist() == [2]
    assert metadata["mode"] == "observed_cells_obs_name_to_h5ad"
    assert metadata["observed_cells_used"] is True


def test_missing_stage_local_indices_without_observed_mapping_fail_closed() -> None:
    data = _adata(
        ["cell_a", "cell_b"],
        [4.0, 4.0],
        ["24hpf", "24hpf"],
        ["A", "B"],
    )
    edges = _edge(source_index=0, target_index=1).drop(
        columns=["source_index_stage", "target_index_stage"]
    )

    with pytest.raises(ValueError, match="observed-cells mapping is required"):
        AUDIT.resolve_edge_stage_indices(
            edges,
            data,
            np.ones(2, dtype=bool),
            stage=4.0,
            stage_label="24hpf",
            time_key="time_point_processed",
            annotation_key="Annotation",
            observed_cells=None,
        )


def test_four_pair_builder_and_context_mass_preserve_score_semantics() -> None:
    raw = pd.concat(
        [
            _edge(
                source_index=0,
                target_index=1,
                source_index_stage=0,
                target_index_stage=1,
                sender_type="Nervous System",
                receiver_type="Nervous System",
                attention_abs_mean=2.0,
                edge_message_norm_joint=3.0,
            ),
            _edge(
                source_index=0,
                target_index=1,
                source_index_stage=0,
                target_index_stage=1,
                sender_type="Nervous System",
                receiver_type="Nervous System",
                attention_abs_mean=4.0,
                edge_message_norm_joint=5.0,
                grouping_seed=202,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "source_index_stage": "_source_stage_index",
            "target_index_stage": "_target_stage_index",
        }
    )
    activities = {
        "dla": np.asarray([1.0, 0.0]),
        "dld": np.asarray([0.5, 0.0]),
        "notch1a": np.asarray([0.0, 1.0]),
        "notch3": np.asarray([0.0, 0.25]),
    }

    family = AUDIT.build_pair_edges(raw, activities)
    mass = AUDIT.context_mass_table(family)

    assert set(family["pair"]) == {
        "dla->notch1a",
        "dla->notch3",
        "dld->notch1a",
        "dld->notch3",
    }
    row = family.loc[family["pair"].eq("dla->notch1a")].iloc[0]
    assert row.lr_only == pytest.approx(1.0)
    assert row.attention_lr == pytest.approx(3.0)
    assert row.exact_message_lr == pytest.approx(4.0)
    core = mass.loc[
        mass["pair"].eq("four_pair_family")
        & mass["context"].eq("core_neural_to_core_neural")
    ]
    assert np.allclose(core["mass_fraction"], 1.0)


def test_global_q95_activity_uses_positive_values_and_clips() -> None:
    values = np.asarray([0.0, 1.0, 1.0, 1.0, 10.0])
    result = AUDIT.global_q95_activity(values)

    assert result[0] == 0.0
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert result[-1] == 1.0
