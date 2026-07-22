from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "paper_style_positive_consistency.py"
)
SPEC = importlib.util.spec_from_file_location("paper_style_positive_consistency", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _grid() -> pd.DataFrame:
    rows = []
    for stage, n_types in enumerate((7, 7, 11, 14, 19)):
        for sender in range(n_types):
            for receiver in range(n_types):
                rows.append(
                    {
                        "stage": float(stage),
                        "sender_type": f"T{sender:02d}",
                        "receiver_type": f"T{receiver:02d}",
                    }
                )
    result = pd.DataFrame(rows)
    assert len(result) == 776
    return result


def test_loader_uses_eq8_ctps_instead_of_legacy_count(tmp_path: Path) -> None:
    comparison, cag_dir, cellchat_dir = (
        tmp_path / "comparison",
        tmp_path / "cag",
        tmp_path / "cellchat",
    )
    for directory in (comparison, cag_dir, cellchat_dir):
        directory.mkdir()
    grid = _grid()
    canonical = []
    for offset, view_id in enumerate(
        (
            "cytobridge__trained__attention",
            "cytobridge__trained__exact_message",
            "commot__project_lr",
            "cellchat__project_lr",
        )
    ):
        frame = grid.copy()
        frame["view_id"] = view_id
        frame["native_score"] = np.arange(len(frame), dtype=float) + offset
        canonical.append(frame)
    pd.concat(canonical, ignore_index=True).to_csv(
        comparison / "canonical_type_pair_scores.csv.gz", index=False, compression="gzip"
    )
    cag = grid.copy()
    cag["cellagentchat_significant_score_sum_mean"] = np.arange(len(cag)) / 10
    cag["cellagentchat_raw_score_sum_mean"] = np.arange(len(cag)) / 5
    cag["cellagentchat_native_primary_mean"] = 999.0
    cag.to_csv(cag_dir / "cellagentchat_type_pair_scores.csv", index=False)
    (cag_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    cellchat = grid.copy()
    cellchat["score"] = np.arange(len(cellchat)) / 7
    cellchat.to_csv(
        cellchat_dir / "cellchat_type_pair_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    (cellchat_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    scores, provenance = analysis.load_scores(comparison, cag_dir, cellchat_dir)
    assert len(scores) == 776
    assert scores["cellagentchat_ctps"].max() == pytest.approx(77.5)
    assert scores["legacy_significant_lr_count"].eq(999).all()
    assert "cellagentchat_manifest" in provenance


def test_consensus_distinguishes_external_and_self_included_designs() -> None:
    scores = _grid()
    base = np.arange(len(scores), dtype=float)
    for index, column in enumerate(
        (
            "cytobridge_attention",
            "cytobridge_exact_message",
            "commot",
            "cellagentchat_ctps",
            "cellagentchat_continuous",
            "cellchat_trimean",
            "cellchat_truncatedmean",
        )
    ):
        scores[column] = base + index
        scores[f"{column}_rank"] = scores.groupby("stage")[column].rank(pct=True)
    scores["external_native_consensus"] = scores[
        ["commot_rank", "cellagentchat_ctps_rank", "cellchat_trimean_rank"]
    ].mean(axis=1)
    scores["external_threshold_relaxed_consensus"] = scores[
        ["commot_rank", "cellagentchat_continuous_rank", "cellchat_truncatedmean_rank"]
    ].mean(axis=1)

    by_stage, summary = analysis.consensus_metrics(scores)
    external = summary.loc[
        summary["design"].eq("external_only_native_primary")
        & summary["target"].eq("CytoBridge attention")
    ].iloc[0]
    article = summary.loc[
        summary["design"].eq("article_style_all_method_native_primary")
        & summary["target"].eq("CytoBridge attention")
    ].iloc[0]
    assert external["consensus_includes_target"] == np.bool_(False)
    assert article["consensus_includes_target"] == np.bool_(True)
    assert external["mean_stage_spearman"] == pytest.approx(1.0)
    assert by_stage["stage"].nunique() == 5


def test_top_overlap_expands_boundary_ties_and_reports_random_enrichment() -> None:
    scores = pd.DataFrame(
        {
            "stage": [0.0] * 6,
            "sender_type": [f"S{i}" for i in range(6)],
            "receiver_type": ["R"] * 6,
            "cytobridge_attention": [5, 4, 4, 1, 0, 0],
            "cytobridge_exact_message": [5, 4, 4, 1, 0, 0],
            "external_native_consensus": [5, 4, 4, 1, 0, 0],
            "external_threshold_relaxed_consensus": [5, 4, 4, 1, 0, 0],
            "commot": [5, 4, 4, 1, 0, 0],
            "cellagentchat_ctps": [5, 4, 4, 1, 0, 0],
            "cellagentchat_continuous": [5, 4, 4, 1, 0, 0],
            "cellchat_trimean": [5, 4, 4, 1, 0, 0],
            "cellchat_truncatedmean": [5, 4, 4, 1, 0, 0],
        }
    )
    by_stage, _ = analysis.top_overlap(scores, top_fraction=1 / 3)
    row = by_stage.loc[
        by_stage["target"].eq("CytoBridge attention")
        & by_stage["reference"].eq("External native consensus")
    ].iloc[0]
    assert row["top_k_requested"] == 2
    assert row["target_set_size_after_boundary_ties"] == 3
    assert row["intersection"] == 3
    assert row["overlap_enrichment_over_random"] == pytest.approx(2.0)


def test_top_set_does_not_promote_zero_ties_to_top_signals() -> None:
    frame = pd.DataFrame({"score": [0.0, 0.0, 0.0, np.nan]})
    assert analysis._top_set(frame, "score", requested=2) == set()
