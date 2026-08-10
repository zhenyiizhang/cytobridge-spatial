from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from reviewer_zebrafish_ccc import (
    jam_trained_init_random_control as CONTROL,
)  # noqa: E402


def _reordered_h5ad() -> ad.AnnData:
    # Attribution order is source, target, other; H5AD order intentionally differs.
    obs = pd.DataFrame(
        {
            "time_point_processed": [3.0, 3.0, 3.0],
            "time": ["18hpf", "18hpf", "18hpf"],
            "Annotation": ["Somite", "Somite", "Somite"],
        },
        index=["other", "source", "target"],
    )
    expression = np.asarray(
        [
            [0.0, 0.0],  # other
            [1.0, 0.0],  # source: jam2a+
            [0.0, 1.0],  # target: jam3b+
        ],
        dtype=np.float32,
    )
    return ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=["jam2a", "jam3b"]),
    )


def _write_observed(path: Path) -> Path:
    pd.DataFrame(
        {
            "global_index": [0, 1, 2],
            "obs_name": ["source", "target", "other"],
            "stage": [3.0, 3.0, 3.0],
            "stage_label": ["18hpf", "18hpf", "18hpf"],
            "cell_type": ["Somite", "Somite", "Somite"],
        }
    ).to_csv(path, index=False)
    return path


def _write_edges(path: Path) -> Path:
    pd.DataFrame(
        {
            "stage": [3.0],
            "stage_label": ["18hpf"],
            "grouping_seed": [101],
            "source_index": [0],
            "target_index": [1],
            "sender_type": ["Somite"],
            "receiver_type": ["Somite"],
            "attention_abs_mean": [0.75],
        }
    ).to_csv(path, index=False)
    return path


def test_obs_name_mapping_defeats_h5ad_index_order_trap(tmp_path: Path) -> None:
    data = _reordered_h5ad()
    observed_path = _write_observed(tmp_path / "observed_cells.csv")
    edge_path = _write_edges(tmp_path / "edge_controls_seed_101.csv.gz")
    mapping, _, metadata = CONTROL.observed_cell_mapping(
        data,
        observed_path,
        time_key="time_point_processed",
        time_label_key="time",
        annotation_key="Annotation",
    )
    edges, _ = CONTROL.load_condition_edges(
        edge_path,
        "trained",
        data,
        mapping,
        stage=3.0,
        stage_label="18hpf",
        grouping_seed=101,
        time_key="time_point_processed",
        annotation_key="Annotation",
    )

    assert edges.loc[0, "source_h5ad_index"] == 1
    assert edges.loc[0, "target_h5ad_index"] == 2
    assert metadata["global_h5ad_row_order_assumed_without_validation"] is False
    compatibility = CONTROL.jam_compatibility(
        edges["source_h5ad_index"].to_numpy(int),
        edges["target_h5ad_index"].to_numpy(int),
        CONTROL.positive_gene(data, "jam2a"),
        CONTROL.positive_gene(data, "jam3b"),
    )
    assert bool(compatibility.loc[0, "jam_compatible"])


def test_type_pair_rank_zero_completes_full_directed_annotation_square() -> None:
    edges = pd.DataFrame(
        {
            "sender_type": ["A", "B"],
            "receiver_type": ["B", "A"],
            "attention_abs_mean": [4.0, 2.0],
        }
    )

    result = CONTROL.complete_type_pair_ranks(
        edges, ["A", "B"], condition="trained"
    ).set_index(["sender_type", "receiver_type"])

    assert len(result) == 4
    assert result.loc[("A", "B"), "rank_over_n"] == "1/4"
    assert result.loc[("B", "A"), "rank_over_n"] == "2/4"
    for pair in (("A", "A"), ("B", "B")):
        assert result.loc[pair, "raw_attention_mean"] == 0.0
        assert result.loc[pair, "n_directed_edges"] == 0
        assert bool(result.loc[pair, "zero_completed_no_edge"])
        assert result.loc[pair, "rank_over_n"] == "3/4"
        assert result.loc[pair, "rank_tie_count"] == 2


def _scaffold(target: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_global_index": [0],
            "target_global_index": [target],
            "source_h5ad_index": [0],
            "target_h5ad_index": [target],
            "source_obs_name": ["source"],
            "target_obs_name": [f"target_{target}"],
            "sender_type": ["Somite"],
            "receiver_type": ["Somite"],
        }
    )


def test_directed_scaffold_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="Directed edge scaffold mismatch"):
        CONTROL.validate_same_scaffold(
            {
                "trained": _scaffold(1),
                "pre_interaction": _scaffold(2),
                "random": _scaffold(1),
            }
        )


def test_jam_compatibility_accepts_both_heterophilic_orientations_only() -> None:
    jam2a = np.asarray([True, False, True, False])
    jam3b = np.asarray([False, True, True, False])

    result = CONTROL.jam_compatibility(
        source=np.asarray([0, 1, 2, 3]),
        target=np.asarray([1, 0, 3, 2]),
        jam2a_positive=jam2a,
        jam3b_positive=jam3b,
    )

    assert result["jam_compatible"].tolist() == [True, True, False, False]
    assert result["jam_compatible_orientation"].tolist() == [
        "source_jam2a_target_jam3b",
        "source_jam3b_target_jam2a",
        "none",
        "none",
    ]


@pytest.mark.parametrize(
    ("pre_interaction_option", "deprecated_alias_used"),
    [
        ("--pre-interaction-edges", False),
        ("--init-edges", True),
    ],
)
def test_end_to_end_writes_canonical_pre_interaction_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pre_interaction_option: str,
    deprecated_alias_used: bool,
) -> None:
    obs = pd.DataFrame(
        {
            "time_point_processed": [3.0] * 4,
            "time": ["18hpf"] * 4,
            "Annotation": ["Somite"] * 4,
        },
        index=[f"cell_{index}" for index in range(4)],
    )
    data = ad.AnnData(
        X=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        obs=obs,
        var=pd.DataFrame(index=["jam2a", "jam3b"]),
    )
    h5ad = tmp_path / "data.h5ad"
    data.write_h5ad(h5ad)
    observed = tmp_path / "observed_cells.csv.gz"
    pd.DataFrame(
        {
            "global_index": np.arange(4),
            "obs_name": data.obs_names.astype(str),
            "stage": [3.0] * 4,
            "stage_label": ["18hpf"] * 4,
            "cell_type": ["Somite"] * 4,
        }
    ).to_csv(observed, index=False)
    scaffold = pd.DataFrame(
        {
            "stage": [3.0] * 4,
            "stage_label": ["18hpf"] * 4,
            "grouping_seed": [101] * 4,
            "source_index": [0, 2, 0, 3],
            "target_index": [1, 1, 3, 2],
            "sender_type": ["Somite"] * 4,
            "receiver_type": ["Somite"] * 4,
        }
    )
    paths: dict[str, Path] = {}
    for condition, attention in {
        "trained": [4.0, 3.0, 2.0, 1.0],
        "pre_interaction": [1.0, 2.0, 3.0, 4.0],
        "random": [2.0, 4.0, 1.0, 3.0],
    }.items():
        path = tmp_path / f"{condition}.csv.gz"
        scaffold.assign(attention_abs_mean=attention).to_csv(path, index=False)
        paths[condition] = path
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jam_trained_init_random_control.py",
            "--h5ad",
            str(h5ad),
            "--observed-cells",
            str(observed),
            "--trained-edges",
            str(paths["trained"]),
            pre_interaction_option,
            str(paths["pre_interaction"]),
            "--random-edges",
            str(paths["random"]),
            "--output-dir",
            str(output),
        ],
    )

    CONTROL.main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["n_stage_scaffold_edges"] == 4
    assert manifest["counts"]["n_somite_somite_scaffold_edges"] == 4
    assert manifest["guardrails"]["raw_attention_scale_compared_across_models"] is False
    compatibility = manifest["cli_compatibility"]
    assert compatibility["canonical_argument"] == "--pre-interaction-edges"
    assert compatibility["deprecated_alias"] == "--init-edges"
    assert compatibility["deprecated_alias_used"] is deprecated_alias_used
    assert compatibility["required_checkpoint"] == "Refine/best_model.pth"
    assert (
        compatibility["forbidden_checkpoint_for_this_control"]
        == "Init_interaction/best_model.pth"
    )
    delta = pd.read_csv(
        output / "tables" / "trained_pre_interaction_edge_percentile_delta.csv.gz"
    )
    assert "trained_minus_pre_interaction_attention_percentile" in delta
    assert "init_attention_percentile" not in delta
    assert not any("raw" in column.casefold() for column in delta.columns)
    summary = pd.read_csv(
        output / "tables" / "jam_compatibility_percentile_summary.csv"
    )
    assert set(summary["condition"]) == {
        "trained",
        "pre_interaction",
        "random",
    }
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Refine/best_model.pth" in readme
    assert "must not" in readme
    assert "Init_interaction/best_model.pth" in readme


def test_canonical_pre_interaction_argument_has_priority_over_deprecated_alias(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.csv"
    deprecated = tmp_path / "deprecated.csv"
    resolved, metadata = CONTROL.resolve_pre_interaction_edges(
        argparse.Namespace(
            pre_interaction_edges=canonical,
            init_edges=deprecated,
        )
    )

    assert resolved == canonical
    assert metadata["deprecated_alias_provided"] is True
    assert metadata["deprecated_alias_ignored_because_canonical_was_provided"] is True
    assert metadata["deprecated_alias_used"] is False
